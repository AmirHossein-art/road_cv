import csv
import gc
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

# داخل این پوشه فعلاً حدود 20 تصویر که مطمئنی تابلو دارند قرار بده.
IMAGE_DIR = PROJECT_ROOT / "traffic_sign_benchmark"

# Output dircetion name is sam3_traffic_sign_detection_ImageSize_ConfidenceThreshold
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sam3_traffic_sign_detection_644_0.20"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"
INSTANCE_DIR = OUTPUT_DIR / "instances"

ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
INSTANCE_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SUMMARY_CSV = OUTPUT_DIR / "image_summary.csv"
DETECTIONS_CSV = OUTPUT_DIR / "detections.csv"

# Apparent-size filtering
# این‌ها روی ابعاد تصویر اصلی اعمال می‌شوند، نه IMAGE_SIZE مدل.
MIN_BOX_HEIGHT_RATIO = 0.0125
MIN_BOX_AREA_RATIO = 0.00012

# =============================================================================
# 2. Test configuration
# =============================================================================

TEST_IMAGE_LIMIT = 100

PROMPTS = [
    "traffic sign",
]

CONFIDENCE_THRESHOLD = 0.20

# SAM3 stride برابر 14 دارد و 644 مضرب 14 است.
# برای تست اول روی GTX 1660 Ti امن‌تر از رزولوشن‌های بالاتر است.
IMAGE_SIZE = 644

# ذخیره ماسک هر instance برای استفاده‌های بعدی
SAVE_INSTANCE_MASKS = True

# رنگ bounding box در OpenCV به صورت BGR
BOX_COLOR = (0, 255, 0)

VALID_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


# =============================================================================
# 3. Utility functions
# =============================================================================

def get_unique_image_paths(image_dir: Path) -> list[Path]:
    """
    Return unique image paths without Windows case-sensitivity duplication.
    """
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )


def class_name_from_id(class_id: int) -> str:
    """
    Map SAM3 class ID to the corresponding text prompt.
    """
    if 0 <= class_id < len(PROMPTS):
        return PROMPTS[class_id]

    return f"unknown_class_{class_id}"


def resize_masks_to_image(
    masks: np.ndarray,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    """
    Ensure every instance mask has exactly the same dimensions as the
    original image.

    Output shape:
        [number_of_instances, image_height, image_width]
    """
    if masks is None or len(masks) == 0:
        return np.empty(
            (0, image_height, image_width),
            dtype=np.uint8,
        )

    resized_masks = []

    for mask in masks:
        if mask.shape != (image_height, image_width):
            mask = cv2.resize(
                mask.astype(np.float32),
                (image_width, image_height),
                interpolation=cv2.INTER_NEAREST,
            )

        binary_mask = (mask > 0.5).astype(np.uint8)
        resized_masks.append(binary_mask)

    return np.stack(resized_masks, axis=0)


def draw_detections(
    image: np.ndarray,
    boxes_xyxy: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
) -> np.ndarray:
    """
    Draw bounding boxes, confidence values, and instance numbers.
    """
    annotated = image.copy()

    if len(boxes_xyxy) == 0:
        cv2.rectangle(
            annotated,
            (10, 10),
            (610, 65),
            (0, 0, 0),
            thickness=-1,
        )

        cv2.putText(
            annotated,
            "NO TRAFFIC SIGN DETECTED",
            (25, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        return annotated

    for detection_index, (
        box,
        confidence,
        class_id,
    ) in enumerate(
        zip(boxes_xyxy, confidences, class_ids),
        start=1,
    ):
        x1, y1, x2, y2 = box

        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        class_name = class_name_from_id(int(class_id))

        label = (
            f"#{detection_index} "
            f"{class_name} "
            f"{confidence:.2f}"
        )

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            BOX_COLOR,
            thickness=3,
        )

        text_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            2,
        )

        text_width, text_height = text_size

        label_top = max(
            y1 - text_height - baseline - 8,
            0,
        )

        cv2.rectangle(
            annotated,
            (x1, label_top),
            (
                min(x1 + text_width + 10, annotated.shape[1] - 1),
                y1,
            ),
            (0, 0, 0),
            thickness=-1,
        )

        cv2.putText(
            annotated,
            label,
            (x1 + 5, max(y1 - baseline - 4, text_height + 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            BOX_COLOR,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    return annotated


def extract_result_arrays(
    result,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract boxes, confidence values, class IDs, and masks from one
    Ultralytics Results object.
    """

    boxes_xyxy = np.empty((0, 4), dtype=np.float32)
    confidences = np.empty((0,), dtype=np.float32)
    class_ids = np.empty((0,), dtype=np.int32)
    masks = np.empty((0, 0, 0), dtype=np.uint8)

    if (
        getattr(result, "boxes", None) is not None
        and result.boxes is not None
        and len(result.boxes) > 0
    ):
        boxes_xyxy = (
            result.boxes.xyxy
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        if getattr(result.boxes, "conf", None) is not None:
            confidences = (
                result.boxes.conf
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        if getattr(result.boxes, "cls", None) is not None:
            class_ids = (
                result.boxes.cls
                .detach()
                .cpu()
                .numpy()
                .astype(np.int32)
            )

    if (
        getattr(result, "masks", None) is not None
        and result.masks is not None
        and len(result.masks) > 0
    ):
        masks = (
            result.masks.data
            .detach()
            .cpu()
            .numpy()
        )

    return boxes_xyxy, confidences, class_ids, masks


def filter_near_candidates_by_apparent_size(
    boxes_xyxy: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    masks: np.ndarray,
    image_height: int,
    image_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove detections that appear too small in the original image.

    Important:
        This is only an apparent-size filter, not a true metric-distance filter.

    Returns:
        filtered boxes
        filtered confidences
        filtered class IDs
        filtered masks
        keep indices
    """

    if len(boxes_xyxy) == 0:
        return (
            boxes_xyxy,
            confidences,
            class_ids,
            masks,
            np.empty((0,), dtype=np.int64),
        )

    box_widths = np.maximum(
        boxes_xyxy[:, 2] - boxes_xyxy[:, 0],
        0.0,
    )

    box_heights = np.maximum(
        boxes_xyxy[:, 3] - boxes_xyxy[:, 1],
        0.0,
    )

    box_areas = box_widths * box_heights

    height_ratios = box_heights / float(image_height)

    area_ratios = box_areas / float(
        image_width * image_height
    )

    keep_mask = (
        (height_ratios >= MIN_BOX_HEIGHT_RATIO)
        & (area_ratios >= MIN_BOX_AREA_RATIO)
    )

    keep_indices = np.flatnonzero(keep_mask)

    filtered_boxes = boxes_xyxy[keep_indices]
    filtered_confidences = confidences[keep_indices]
    filtered_class_ids = class_ids[keep_indices]

    if len(masks) == len(boxes_xyxy):
        filtered_masks = masks[keep_indices]
    else:
        filtered_masks = masks

    return (
        filtered_boxes,
        filtered_confidences,
        filtered_class_ids,
        filtered_masks,
        keep_indices,
    )


# =============================================================================
# 4. Main
# =============================================================================

def main() -> None:
    # -------------------------------------------------------------------------
    # Validate paths
    # -------------------------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"SAM3 model not found: {MODEL_PATH}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            "\nTraffic-sign image folder does not exist:\n"
            f"{IMAGE_DIR}\n\n"
            "Create this folder and place test images inside it."
        )

    image_paths = get_unique_image_paths(IMAGE_DIR)

    if TEST_IMAGE_LIMIT is not None:
        image_paths = image_paths[:TEST_IMAGE_LIMIT]

    if not image_paths:
        raise RuntimeError(
            f"No valid images found in: {IMAGE_DIR}"
        )

    # -------------------------------------------------------------------------
    # Device information
    # -------------------------------------------------------------------------

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("=" * 72)
    print("SAM3 TRAFFIC SIGN DETECTION TEST")
    print("=" * 72)
    print(f"Model: {MODEL_PATH}")
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Images to process: {len(image_paths)}")
    print(f"Prompts: {PROMPTS}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

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

    predictor = SAM3SemanticPredictor(
        overrides=overrides
    )

    # -------------------------------------------------------------------------
    # CSV schemas
    # -------------------------------------------------------------------------

    image_summary_fields = [
        "image",
        "width",
        "height",
        "num_detections",
        "max_confidence",
        "average_confidence",
        "annotated_image",
        "instance_file",
        "status",
        "error",
    ]

    detection_fields = [
        "image",
        "detection_index",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "box_width",
        "box_height",
        "box_area",
        "mask_pixels",
        "annotated_image",
        "instance_file",
    ]

    total_detections = 0
    images_with_detections = 0

    # -------------------------------------------------------------------------
    # Process images
    # -------------------------------------------------------------------------

    with open(
        IMAGE_SUMMARY_CSV,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as summary_file, open(
        DETECTIONS_CSV,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as detections_file:

        summary_writer = csv.DictWriter(
            summary_file,
            fieldnames=image_summary_fields,
        )

        detections_writer = csv.DictWriter(
            detections_file,
            fieldnames=detection_fields,
        )

        summary_writer.writeheader()
        detections_writer.writeheader()

        with torch.inference_mode():
            for image_path in tqdm(
                image_paths,
                desc="Detecting traffic signs",
            ):
                annotated_path = (
                    ANNOTATED_DIR
                    / f"{image_path.stem}_sign_boxes.jpg"
                )

                instance_path = (
                    INSTANCE_DIR
                    / f"{image_path.stem}_sign_instances.npz"
                )

                summary_row = {
                    "image": image_path.name,
                    "width": "",
                    "height": "",
                    "num_detections": 0,
                    "max_confidence": "",
                    "average_confidence": "",
                    "annotated_image": str(annotated_path),
                    "instance_file": "",
                    "status": "failed",
                    "error": "",
                }

                results = None
                result_cpu = None

                try:
                    image = cv2.imread(str(image_path))

                    if image is None:
                        raise RuntimeError(
                            f"OpenCV could not read image: {image_path}"
                        )

                    image_height, image_width = image.shape[:2]

                    summary_row["width"] = image_width
                    summary_row["height"] = image_height

                    # ---------------------------------------------------------
                    # SAM3 inference
                    # ---------------------------------------------------------

                    predictor.set_image(str(image_path))

                    results = predictor(
                        text=PROMPTS
                    )

                    if results is None or len(results) == 0:
                        raise RuntimeError(
                            "SAM3 returned no Results object."
                        )

                    result_cpu = results[0].cpu()

                    del results
                    results = None

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # ---------------------------------------------------------
                    # Extract predictions
                    # ---------------------------------------------------------

                    (
                        boxes_xyxy,
                        confidences,
                        class_ids,
                        raw_masks,
                    ) = extract_result_arrays(result_cpu)

                    resized_masks = resize_masks_to_image(
                        masks=raw_masks,
                        image_height=image_height,
                        image_width=image_width,
                    )

                    # Usually boxes and masks have the same count.
                    # We use the minimum only for safe per-instance indexing.
                    aligned_count = min(
                        len(boxes_xyxy),
                        len(confidences),
                        len(class_ids),
                        len(resized_masks),
                    )

                    boxes_xyxy = boxes_xyxy[:aligned_count]
                    confidences = confidences[:aligned_count]
                    class_ids = class_ids[:aligned_count]
                    resized_masks = resized_masks[:aligned_count]

                    if len(resized_masks) > aligned_count:
                        resized_masks = resized_masks[:aligned_count]

                    raw_detection_count = aligned_count

                    (
                        boxes_xyxy,
                        confidences,
                        class_ids,
                        resized_masks,
                        kept_indices,
                    ) = filter_near_candidates_by_apparent_size(
                        boxes_xyxy=boxes_xyxy,
                        confidences=confidences,
                        class_ids=class_ids,
                        masks=resized_masks,
                        image_height=image_height,
                        image_width=image_width,
                    )

                    filtered_detection_count = len(boxes_xyxy)

                    tqdm.write(
                        f"{image_path.name}: "
                        f"raw={raw_detection_count}, "
                        f"kept_after_size_filter={filtered_detection_count}"
                    )

                    # ---------------------------------------------------------
                    # Draw detections
                    # ---------------------------------------------------------

                    annotated = draw_detections(
                        image=image,
                        boxes_xyxy=boxes_xyxy,
                        confidences=confidences,
                        class_ids=class_ids,
                    )

                    if not cv2.imwrite(
                        str(annotated_path),
                        annotated,
                    ):
                        raise RuntimeError(
                            f"Could not save: {annotated_path}"
                        )

                    # ---------------------------------------------------------
                    # Save machine-readable instance data
                    # ---------------------------------------------------------

                    if SAVE_INSTANCE_MASKS:
                        np.savez_compressed(
                            instance_path,
                            image_name=np.asarray(
                                image_path.name
                            ),
                            image_width=np.asarray(
                                image_width,
                                dtype=np.int32,
                            ),
                            image_height=np.asarray(
                                image_height,
                                dtype=np.int32,
                            ),
                            prompts=np.asarray(
                                PROMPTS,
                                dtype="<U64",
                            ),
                            boxes_xyxy=boxes_xyxy.astype(
                                np.float32
                            ),
                            confidences=confidences.astype(
                                np.float32
                            ),
                            class_ids=class_ids.astype(
                                np.int32
                            ),
                            masks=resized_masks.astype(
                                np.uint8
                            ),
                        )

                        summary_row["instance_file"] = str(
                            instance_path
                        )

                    # ---------------------------------------------------------
                    # Write one CSV row per detection
                    # ---------------------------------------------------------

                    for detection_index in range(
                        aligned_count
                    ):
                        x1, y1, x2, y2 = boxes_xyxy[
                            detection_index
                        ]

                        box_width = max(
                            float(x2 - x1),
                            0.0,
                        )

                        box_height = max(
                            float(y2 - y1),
                            0.0,
                        )

                        box_area = (
                            box_width * box_height
                        )

                        if detection_index < len(
                            resized_masks
                        ):
                            mask_pixels = int(
                                resized_masks[
                                    detection_index
                                ].sum()
                            )
                        else:
                            mask_pixels = 0

                        class_id = int(
                            class_ids[detection_index]
                        )

                        detections_writer.writerow({
                            "image": image_path.name,
                            "detection_index": detection_index,
                            "class_id": class_id,
                            "class_name": class_name_from_id(
                                class_id
                            ),
                            "confidence": float(
                                confidences[
                                    detection_index
                                ]
                            ),
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2),
                            "box_width": box_width,
                            "box_height": box_height,
                            "box_area": box_area,
                            "mask_pixels": mask_pixels,
                            "annotated_image": str(
                                annotated_path
                            ),
                            "instance_file": (
                                str(instance_path)
                                if SAVE_INSTANCE_MASKS
                                else ""
                            ),
                        })

                    # ---------------------------------------------------------
                    # Image summary
                    # ---------------------------------------------------------

                    num_detections = aligned_count

                    summary_row["num_detections"] = (
                        num_detections
                    )

                    if num_detections > 0:
                        images_with_detections += 1
                        total_detections += num_detections

                        summary_row["max_confidence"] = float(
                            confidences.max()
                        )

                        summary_row[
                            "average_confidence"
                        ] = float(
                            confidences.mean()
                        )

                        summary_row["status"] = (
                            "detections_found"
                        )

                    else:
                        summary_row["status"] = (
                            "no_detection"
                        )

                    tqdm.write(
                        f"{image_path.name}: "
                        f"{num_detections} detections"
                    )

                except Exception as exc:
                    summary_row["status"] = "error"
                    summary_row["error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )

                    tqdm.write(
                        f"ERROR - {image_path.name}: "
                        f"{summary_row['error']}"
                    )

                finally:
                    summary_writer.writerow(summary_row)

                    summary_file.flush()
                    detections_file.flush()

                    if result_cpu is not None:
                        del result_cpu

                    if results is not None:
                        del results

                    gc.collect()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("TEST FINISHED")
    print("=" * 72)
    print(f"Processed images: {len(image_paths)}")
    print(
        "Images with at least one detection: "
        f"{images_with_detections}"
    )
    print(f"Total detections: {total_detections}")
    print(f"Annotated images: {ANNOTATED_DIR}")
    print(f"Instance files: {INSTANCE_DIR}")
    print(f"Image summary: {IMAGE_SUMMARY_CSV}")
    print(f"Detection table: {DETECTIONS_CSV}")


if __name__ == "__main__":
    main()