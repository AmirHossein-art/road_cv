import csv
import gc
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics.models.sam import SAM3SemanticPredictor


# =============================================================================
# 1. Project paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "sam3.pt"
IMAGE_DIR = PROJECT_ROOT / "guardrail_benchmark"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sam3_guardrail_seg_debug"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "summary_semantic_debug.csv"


# =============================================================================
# 2. Test configuration
# =============================================================================

# فعلاً فقط سه تصویر پردازش می‌شوند.
# پس از موفقیت تست، مقدار را None قرار بده.
TEST_IMAGE_LIMIT = None

CONFIDENCE_THRESHOLD = 0.25

# SAM3 stride برابر 14 دارد.
# 644 مضرب 14 است و هشدار تبدیل 640 به 644 را حذف می‌کند.
IMAGE_SIZE = 644

OVERLAY_ALPHA = 0.55

# برای تست اولیه فقط دو concept مشخص استفاده می‌کنیم.
# ترتیب این لیست بسیار مهم است.
PROMPTS = [
    "concrete barrier",  # class_id = 0
    "metal guardrail",   # class_id = 1
]

# mapping کلاس‌ها بر اساس ترتیب PROMPTS
CLASS_GROUP_BY_ID = {
    0: "barrier",
    1: "guardrail",
}

# رنگ‌ها در OpenCV به صورت BGR هستند.
COLOR_MAP = {
    "barrier": (0, 255, 255),   # Yellow
    "guardrail": (255, 0, 0),   # Blue
    "overlap": (0, 0, 255),     # Red
}


# =============================================================================
# 3. Utility functions
# =============================================================================

def get_unique_image_paths(image_dir: Path) -> list[Path]:
    """
    Return unique image paths without the Windows JPG/jpg duplication issue.
    """
    valid_extensions = {".jpg", ".jpeg", ".png"}

    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in valid_extensions
    )


def names_to_text(names_data) -> str:
    """
    Convert result.names to a CSV-safe readable string.
    """
    if isinstance(names_data, dict):
        normalized = {
            str(key): str(value)
            for key, value in names_data.items()
        }
        return json.dumps(normalized, ensure_ascii=False)

    if isinstance(names_data, (list, tuple)):
        return json.dumps(
            [str(value) for value in names_data],
            ensure_ascii=False,
        )

    return str(names_data)


def add_warning_text(image: np.ndarray, message: str) -> np.ndarray:
    """
    Add a visible warning to an output image when masks are empty.
    This prevents an unchanged image from being mistaken for a valid result.
    """
    output = image.copy()

    cv2.rectangle(
        output,
        (10, 10),
        (min(output.shape[1] - 10, 850), 70),
        (0, 0, 0),
        thickness=-1,
    )

    cv2.putText(
        output,
        message,
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    return output


# =============================================================================
# 4. Semantic-style output generation
# =============================================================================

def save_segmentation_outputs(
    result,
    overlay_path: Path,
    label_visual_path: Path,
    label_raw_path: Path,
    alpha: float = OVERLAY_ALPHA,
) -> dict:
    """
    Create and save three outputs:

    1. Colored overlay:
       - concrete barrier = yellow
       - metal guardrail = blue
       - overlapping predictions = red

    2. Colored semantic label visualization.

    3. Raw single-channel label map:
       - 0 = background
       - 1 = concrete barrier
       - 2 = metal guardrail
       - 3 = overlap

    Returns diagnostic statistics for CSV output.
    """

    img = result.orig_img.copy()
    h_img, w_img = img.shape[:2]

    stats = {
        "raw_mask_count": 0,
        "nonempty_mask_count": 0,
        "barrier_instance_count": 0,
        "guardrail_instance_count": 0,
        "unknown_instance_count": 0,
        "barrier_pixels": 0,
        "guardrail_pixels": 0,
        "overlap_pixels": 0,
        "total_positive_pixels_before_merge": 0,
        "mask_min": "",
        "mask_max": "",
    }

    # خروجی‌های خالی اولیه
    raw_label_map = np.zeros((h_img, w_img), dtype=np.uint8)
    label_visual = np.zeros_like(img)

    # -------------------------------------------------------------------------
    # Check whether mask objects exist
    # -------------------------------------------------------------------------

    if getattr(result, "masks", None) is None or result.masks is None:
        warning_image = add_warning_text(
            img,
            "NO MASK OBJECT RETURNED",
        )

        cv2.imwrite(str(overlay_path), warning_image)
        cv2.imwrite(str(label_visual_path), label_visual)
        cv2.imwrite(str(label_raw_path), raw_label_map)

        return stats

    mask_tensor = result.masks.data.detach().cpu()

    if len(mask_tensor) == 0:
        warning_image = add_warning_text(
            img,
            "MASK OBJECT EXISTS, BUT MASK COUNT IS ZERO",
        )

        cv2.imwrite(str(overlay_path), warning_image)
        cv2.imwrite(str(label_visual_path), label_visual)
        cv2.imwrite(str(label_raw_path), raw_label_map)

        return stats

    stats["raw_mask_count"] = len(mask_tensor)
    stats["mask_min"] = float(mask_tensor.min().item())
    stats["mask_max"] = float(mask_tensor.max().item())

    masks = mask_tensor.numpy()

    # -------------------------------------------------------------------------
    # Read class IDs
    # -------------------------------------------------------------------------

    class_ids = np.array([], dtype=np.int64)

    if (
        getattr(result, "boxes", None) is not None
        and result.boxes is not None
        and getattr(result.boxes, "cls", None) is not None
    ):
        class_ids = (
            result.boxes.cls
            .detach()
            .cpu()
            .to(torch.int64)
            .numpy()
        )

    # -------------------------------------------------------------------------
    # Merge valid masks by semantic class
    # -------------------------------------------------------------------------

    barrier_mask = np.zeros((h_img, w_img), dtype=bool)
    guardrail_mask = np.zeros((h_img, w_img), dtype=bool)

    for idx, mask in enumerate(masks):
        if mask.shape != (h_img, w_img):
            mask_resized = cv2.resize(
                mask.astype(np.float32),
                (w_img, h_img),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            mask_resized = mask

        # بعد از اصلاح threshold داخل کتابخانه، mask باید binary باشد.
        # این threshold فقط برای تبدیل مطمئن آرایه به bool است.
        binary_mask = mask_resized > 0.5

        positive_pixels = int(binary_mask.sum())

        stats["total_positive_pixels_before_merge"] += positive_pixels

        # مهم: mask خالی را prediction معتبر حساب نمی‌کنیم.
        if positive_pixels == 0:
            continue

        stats["nonempty_mask_count"] += 1

        if idx < len(class_ids):
            class_id = int(class_ids[idx])
        else:
            class_id = -1

        group_name = CLASS_GROUP_BY_ID.get(class_id)

        if group_name == "barrier":
            barrier_mask |= binary_mask
            stats["barrier_instance_count"] += 1

        elif group_name == "guardrail":
            guardrail_mask |= binary_mask
            stats["guardrail_instance_count"] += 1

        else:
            stats["unknown_instance_count"] += 1

    # -------------------------------------------------------------------------
    # Separate overlap and exclusive areas
    # -------------------------------------------------------------------------

    overlap_mask = barrier_mask & guardrail_mask

    exclusive_barrier = barrier_mask & ~overlap_mask
    exclusive_guardrail = guardrail_mask & ~overlap_mask

    stats["barrier_pixels"] = int(barrier_mask.sum())
    stats["guardrail_pixels"] = int(guardrail_mask.sum())
    stats["overlap_pixels"] = int(overlap_mask.sum())

    # -------------------------------------------------------------------------
    # Create colored overlay
    # -------------------------------------------------------------------------

    overlay = img.copy()

    overlay[exclusive_barrier] = COLOR_MAP["barrier"]
    overlay[exclusive_guardrail] = COLOR_MAP["guardrail"]
    overlay[overlap_mask] = COLOR_MAP["overlap"]

    blended = cv2.addWeighted(
        overlay,
        alpha,
        img,
        1.0 - alpha,
        0,
    )

    # اگر mask object وجود دارد ولی همه maskها خالی‌اند،
    # روی خروجی هشدار واضح درج شود.
    if stats["nonempty_mask_count"] == 0:
        blended = add_warning_text(
            blended,
            "MASK OBJECTS RETURNED, BUT ALL MASKS ARE EMPTY",
        )

    elif (
        stats["barrier_pixels"] == 0
        and stats["guardrail_pixels"] == 0
    ):
        blended = add_warning_text(
            blended,
            "NON-EMPTY MASKS EXIST, BUT CLASS MAPPING FAILED",
        )

    cv2.imwrite(str(overlay_path), blended)

    # -------------------------------------------------------------------------
    # Create colored semantic label visualization
    # -------------------------------------------------------------------------

    label_visual[exclusive_barrier] = COLOR_MAP["barrier"]
    label_visual[exclusive_guardrail] = COLOR_MAP["guardrail"]
    label_visual[overlap_mask] = COLOR_MAP["overlap"]

    cv2.imwrite(str(label_visual_path), label_visual)

    # -------------------------------------------------------------------------
    # Create raw semantic label map
    # -------------------------------------------------------------------------

    raw_label_map[exclusive_barrier] = 1
    raw_label_map[exclusive_guardrail] = 2
    raw_label_map[overlap_mask] = 3

    cv2.imwrite(str(label_raw_path), raw_label_map)

    return stats


# =============================================================================
# 5. Main inference loop
# =============================================================================

def main() -> None:
    # -------------------------------------------------------------------------
    # Validate paths
    # -------------------------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found: {IMAGE_DIR}"
        )

    # -------------------------------------------------------------------------
    # Read images
    # -------------------------------------------------------------------------

    image_paths = get_unique_image_paths(IMAGE_DIR)

    if TEST_IMAGE_LIMIT is not None:
        image_paths = image_paths[:TEST_IMAGE_LIMIT]

    if not image_paths:
        raise RuntimeError(
            f"No images found in: {IMAGE_DIR}"
        )

    print("=" * 70)
    print("SAM3 GUARDRAIL / BARRIER SEGMENTATION TEST")
    print("=" * 70)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Images to process: {len(image_paths)}")

    # -------------------------------------------------------------------------
    # Device information
    # -------------------------------------------------------------------------

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Running on device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            "Total GPU memory:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / (1024 ** 3),
                2,
            ),
            "GB",
        )

    # -------------------------------------------------------------------------
    # Fixed prompt order
    # -------------------------------------------------------------------------

    print("\nPrompt order and expected class IDs:")

    for class_id, prompt in enumerate(PROMPTS):
        print(f"  class_id={class_id}: {prompt}")

    # -------------------------------------------------------------------------
    # Create predictor
    # -------------------------------------------------------------------------

    overrides = {
        "conf": CONFIDENCE_THRESHOLD,
        "task": "segment",
        "mode": "predict",
        "model": str(MODEL_PATH),
        "device": device,
        "quantize": 16 if torch.cuda.is_available() else 32,
        "imgsz": IMAGE_SIZE,
        "save": False,
        "show": False,
        "verbose": False,
    }

    predictor = SAM3SemanticPredictor(overrides=overrides)

    # -------------------------------------------------------------------------
    # CSV fields
    # -------------------------------------------------------------------------

    fieldnames = [
        "image",
        "status",
        "error",
        "prompt_0",
        "prompt_1",
        "result_names",
        "class_ids",
        "num_boxes",
        "raw_mask_count",
        "nonempty_mask_count",
        "barrier_instance_count",
        "guardrail_instance_count",
        "unknown_instance_count",
        "barrier_pixels",
        "guardrail_pixels",
        "overlap_pixels",
        "total_positive_pixels_before_merge",
        "mask_min",
        "mask_max",
        "overlay_path",
        "label_visual_path",
        "label_raw_path",
    ]

    # -------------------------------------------------------------------------
    # Run inference
    # -------------------------------------------------------------------------

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        with torch.inference_mode():
            for image_path in tqdm(
                image_paths,
                desc="Processing images",
            ):
                overlay_path = (
                    OUTPUT_DIR
                    / f"{image_path.stem}_overlay.jpg"
                )

                label_visual_path = (
                    OUTPUT_DIR
                    / f"{image_path.stem}_labels_visual.png"
                )

                label_raw_path = (
                    OUTPUT_DIR
                    / f"{image_path.stem}_labels_raw.png"
                )

                row = {
                    "image": image_path.name,
                    "status": "failed",
                    "error": "",
                    "prompt_0": PROMPTS[0],
                    "prompt_1": PROMPTS[1],
                    "result_names": "",
                    "class_ids": "",
                    "num_boxes": 0,
                    "raw_mask_count": 0,
                    "nonempty_mask_count": 0,
                    "barrier_instance_count": 0,
                    "guardrail_instance_count": 0,
                    "unknown_instance_count": 0,
                    "barrier_pixels": 0,
                    "guardrail_pixels": 0,
                    "overlap_pixels": 0,
                    "total_positive_pixels_before_merge": 0,
                    "mask_min": "",
                    "mask_max": "",
                    "overlay_path": str(overlay_path),
                    "label_visual_path": str(label_visual_path),
                    "label_raw_path": str(label_raw_path),
                }

                results = None
                r_cpu = None

                try:
                    # ---------------------------------------------------------
                    # Set current image and run concept segmentation
                    # ---------------------------------------------------------

                    predictor.set_image(str(image_path))
                    results = predictor(text=PROMPTS)

                    if results is None or len(results) == 0:
                        row["status"] = "no_result"
                        row["error"] = "Predictor returned no Results object."

                        empty_image = cv2.imread(str(image_path))

                        if empty_image is not None:
                            warning_image = add_warning_text(
                                empty_image,
                                "PREDICTOR RETURNED NO RESULTS",
                            )
                            cv2.imwrite(
                                str(overlay_path),
                                warning_image,
                            )

                        writer.writerow(row)
                        csv_file.flush()
                        continue

                    # ---------------------------------------------------------
                    # Transfer result to CPU before visualization
                    # ---------------------------------------------------------

                    r_cpu = results[0].cpu()

                    del results
                    results = None

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # ---------------------------------------------------------
                    # Collect class diagnostics
                    # ---------------------------------------------------------

                    row["result_names"] = names_to_text(
                        getattr(r_cpu, "names", {})
                    )

                    class_ids_list = []

                    if (
                        getattr(r_cpu, "boxes", None) is not None
                        and r_cpu.boxes is not None
                    ):
                        row["num_boxes"] = len(r_cpu.boxes)

                        if getattr(r_cpu.boxes, "cls", None) is not None:
                            class_ids_list = [
                                int(value)
                                for value in (
                                    r_cpu.boxes.cls
                                    .detach()
                                    .cpu()
                                    .tolist()
                                )
                            ]

                    row["class_ids"] = json.dumps(
                        class_ids_list
                    )

                    # ---------------------------------------------------------
                    # Generate semantic outputs and diagnostic statistics
                    # ---------------------------------------------------------

                    stats = save_segmentation_outputs(
                        result=r_cpu,
                        overlay_path=overlay_path,
                        label_visual_path=label_visual_path,
                        label_raw_path=label_raw_path,
                    )

                    row.update(stats)

                    if stats["nonempty_mask_count"] > 0:
                        row["status"] = "success"
                    elif stats["raw_mask_count"] > 0:
                        row["status"] = "all_masks_empty"
                    else:
                        row["status"] = "no_masks"

                    # ---------------------------------------------------------
                    # Print per-image diagnostics
                    # ---------------------------------------------------------

                    tqdm.write("")
                    tqdm.write(f"Image: {image_path.name}")
                    tqdm.write(
                        f"  result.names: {row['result_names']}"
                    )
                    tqdm.write(
                        f"  class_ids: {row['class_ids']}"
                    )
                    tqdm.write(
                        f"  boxes: {row['num_boxes']}"
                    )
                    tqdm.write(
                        f"  raw masks: {row['raw_mask_count']}"
                    )
                    tqdm.write(
                        f"  non-empty masks: "
                        f"{row['nonempty_mask_count']}"
                    )
                    tqdm.write(
                        f"  barrier pixels: "
                        f"{row['barrier_pixels']}"
                    )
                    tqdm.write(
                        f"  guardrail pixels: "
                        f"{row['guardrail_pixels']}"
                    )
                    tqdm.write(
                        f"  overlap pixels: "
                        f"{row['overlap_pixels']}"
                    )
                    tqdm.write(
                        f"  mask min/max: "
                        f"{row['mask_min']} / {row['mask_max']}"
                    )
                    tqdm.write(
                        f"  status: {row['status']}"
                    )

                except Exception as exc:
                    row["status"] = "error"
                    row["error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )

                    print(
                        f"\nError processing {image_path.name}: "
                        f"{row['error']}"
                    )

                finally:
                    writer.writerow(row)
                    csv_file.flush()

                    if r_cpu is not None:
                        del r_cpu

                    if results is not None:
                        del results

                    gc.collect()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Final information
    # -------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST FINISHED")
    print("=" * 70)
    print(f"CSV summary: {CSV_PATH}")
    print(f"Outputs: {OUTPUT_DIR}")

    print("\nExpected interpretation:")
    print(
        "- raw_mask_count > 0 and nonempty_mask_count > 0: "
        "segmentation masks are valid."
    )
    print(
        "- raw_mask_count > 0 and nonempty_mask_count == 0: "
        "mask objects exist, but their pixel data is empty."
    )
    print(
        "- nonempty_mask_count > 0 but barrier/guardrail pixels == 0: "
        "class-ID mapping is incorrect."
    )

    print(
        "\nAfter this three-image test succeeds, set "
        "TEST_IMAGE_LIMIT = None to process the full folder."
    )


if __name__ == "__main__":
    main()