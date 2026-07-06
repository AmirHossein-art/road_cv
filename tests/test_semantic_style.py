import cv2
import numpy as np

from pathlib import Path
import csv
import gc

import torch
from ultralytics.models.sam import SAM3SemanticPredictor

def save_semantic_overlay_two_classes(result, output_path, alpha=0.45):
    """
    Semantic-style overlay with 2 colors:
    - barrier / jersey barrier / median barrier -> yellow
    - guardrail / roadside guardrail -> blue
    - no boxes
    - no labels
    """

    img = result.orig_img.copy()

    # اگر mask نداریم، خود تصویر ذخیره شود
    if getattr(result, "masks", None) is None or result.masks is None:
        cv2.imwrite(str(output_path), img)
        return

    if getattr(result, "boxes", None) is None or result.boxes is None:
        cv2.imwrite(str(output_path), img)
        return

    masks = result.masks.data.cpu().numpy()   # shape: [n, h, w]
    boxes = result.boxes

    if masks is None or len(masks) == 0:
        cv2.imwrite(str(output_path), img)
        return

    h_img, w_img = img.shape[:2]

    # دو mask نهایی جدا
    barrier_mask = np.zeros((h_img, w_img), dtype=bool)
    guardrail_mask = np.zeros((h_img, w_img), dtype=bool)

    for idx, mask in enumerate(masks):
        # گرفتن class id
        class_id = None
        if getattr(boxes, "cls", None) is not None and idx < len(boxes.cls):
            class_id = int(boxes.cls[idx].item())

        # resize اگر لازم بود
        if mask.shape != (h_img, w_img):
            mask_resized = cv2.resize(
                mask.astype(np.float32),
                (w_img, h_img),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            mask_resized = mask

        binary_mask = mask_resized > 0.5

        # گروه‌بندی کلاس‌ها
        if class_id in [0, 1, 2, 3]:
            barrier_mask |= binary_mask
        elif class_id in [4, 5]:
            guardrail_mask |= binary_mask

    overlay = img.copy()

    # رنگ‌ها در OpenCV به صورت BGR
    barrier_color = (0, 255, 255)   # زرد
    guardrail_color = (255, 0, 0)   # آبی

    # اول barrier را رنگ کن
    overlay[barrier_mask] = barrier_color

    # بعد guardrail را رنگ کن
    # اگر overlap باشد، guardrail روی barrier می‌نشیند
    overlay[guardrail_mask] = guardrail_color

    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    cv2.imwrite(str(output_path), blended)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "sam3.pt"
IMAGE_DIR = PROJECT_ROOT / "guardrail_benchmark"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sam3_semantic_style"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "summary_semantic_style.csv"

# None یعنی برای همه عکس‌ها خروجی تصویری ذخیره شود
# اگر حافظه یا زمان زیاد شد، مثلاً بگذار 20
SAVE_VISUAL_LIMIT = None


def get_unique_image_paths(image_dir):
    valid_extensions = {".jpg", ".jpeg", ".png"}

    return sorted(
        [
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_extensions
        ]
    )


def summarize_result(r_cpu):
    """
    Extract simple statistics from one Ultralytics Results object.
    These are model-output statistics, not true accuracy metrics.
    """
    num_masks = 0
    num_boxes = 0

    confidences = []
    avg_conf = ""
    max_conf = ""
    min_conf = ""

    total_mask_area_ratio = ""
    largest_mask_area_ratio = ""

    # Boxes and confidence
    if getattr(r_cpu, "boxes", None) is not None and r_cpu.boxes is not None:
        num_boxes = len(r_cpu.boxes)

        if getattr(r_cpu.boxes, "conf", None) is not None:
            confidences = [float(x) for x in r_cpu.boxes.conf.tolist()]

            if confidences:
                avg_conf = sum(confidences) / len(confidences)
                max_conf = max(confidences)
                min_conf = min(confidences)

    # Masks and mask area
    if getattr(r_cpu, "masks", None) is not None and r_cpu.masks is not None:
        masks_data = r_cpu.masks.data  # shape: [n, h, w]
        num_masks = len(masks_data)

        if num_masks > 0:
            h, w = masks_data.shape[-2], masks_data.shape[-1]
            image_area = h * w

            mask_areas = masks_data.float().sum(dim=(1, 2)).tolist()
            mask_area_ratios = [float(a) / image_area for a in mask_areas]

            total_mask_area_ratio = sum(mask_area_ratios)
            largest_mask_area_ratio = max(mask_area_ratios)

    return {
        "num_masks": num_masks,
        "num_boxes": num_boxes,
        "avg_conf": avg_conf,
        "max_conf": max_conf,
        "min_conf": min_conf,
        "total_mask_area_ratio": total_mask_area_ratio,
        "largest_mask_area_ratio": largest_mask_area_ratio,
    }


def main():
    print("Project root:", PROJECT_ROOT)
    print("Model path:", MODEL_PATH)
    print("Image dir:", IMAGE_DIR)
    print("Output dir:", OUTPUT_DIR)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image folder not found: {IMAGE_DIR}")

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    image_paths = get_unique_image_paths(IMAGE_DIR)

    # فعلاً فقط یک عکس برای تست
    image_paths = image_paths[:10]

    print(f"Found {len(image_paths)} unique images.")

    if not image_paths:
        print("No images found.")
        return

    prompts = [
        "concrete barrier",
        "guardrail",
    ]

    overrides = dict(
        conf=0.5,
        task="segment",
        mode="predict",
        model=str(MODEL_PATH),
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        quantize=16 if torch.cuda.is_available() else 32,
        save=False,
        verbose=False,
    )

    predictor = SAM3SemanticPredictor(overrides=overrides)

    rows = []

    for i, image_path in enumerate(image_paths, start=1):
        print(f"[{i}/{len(image_paths)}] Processing: {image_path.name}")

        predictor.set_image(str(image_path))
        results = predictor(text=prompts)

        output_image_path = ""

        if results is not None and len(results) > 0:
            # نتیجه را سریع از GPU به CPU منتقل می‌کنیم
            r_cpu = results[0].cpu()
            
            print("names:", r_cpu.names)
            print("cls ids:", r_cpu.boxes.cls.tolist() if r_cpu.boxes is not None else "No boxes")

            # نسخه GPU را آزاد می‌کنیم
            del results
            torch.cuda.empty_cache()

            stats = summarize_result(r_cpu)

            should_save_visual = SAVE_VISUAL_LIMIT is None or i <= SAVE_VISUAL_LIMIT

            if should_save_visual:
                output_image_path = OUTPUT_DIR / f"{image_path.stem}_semantic_2class.jpg"
                save_semantic_overlay_two_classes(r_cpu, output_image_path)

            del r_cpu

        else:
            stats = {
                "num_masks": 0,
                "num_boxes": 0,
                "avg_conf": "",
                "max_conf": "",
                "min_conf": "",
                "total_mask_area_ratio": "",
                "largest_mask_area_ratio": "",
            }

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        rows.append({
            "image": image_path.name,
            "prompts": " | ".join(prompts),
            "num_masks": stats["num_masks"],
            "num_boxes": stats["num_boxes"],
            "avg_conf": stats["avg_conf"],
            "max_conf": stats["max_conf"],
            "min_conf": stats["min_conf"],
            "total_mask_area_ratio": stats["total_mask_area_ratio"],
            "largest_mask_area_ratio": stats["largest_mask_area_ratio"],
            "output_image": str(output_image_path),
        })

    fieldnames = [
        "image",
        "prompts",
        "num_masks",
        "num_boxes",
        "avg_conf",
        "max_conf",
        "min_conf",
        "total_mask_area_ratio",
        "largest_mask_area_ratio",
        "output_image",
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Done.")
    print(f"Summary saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()