import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics.models.sam import SAM3SemanticPredictor

# 1. Setup Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "sam3.pt"
IMAGE_DIR = PROJECT_ROOT / "guardrail_benchmark"

# Setup Output Directory (Step 3)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sam3_semantic_style"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Prompts
prompts = [
    "concrete barrier",
    "road barrier",
    "jersey barrier",
    "median barrier",
    "guardrail",
    "roadside guardrail",
]

def get_unique_image_paths(image_dir):
    """Fix for Windows case-insensitive duplicate path issue"""
    valid_extensions = {".jpg", ".jpeg", ".png"}
    return sorted(
        [
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_extensions
        ]
    )

def save_semantic_overlay_two_classes(result, output_path, alpha=0.45):
    """
    Semantic-style overlay with 2 classes + 1 overlap class:
    - exclusive barrier -> yellow
    - exclusive guardrail -> blue
    - overlap (both barrier and guardrail) -> red
    """
    img = result.orig_img.copy()

    if getattr(result, "masks", None) is None or result.masks is None:
        cv2.imwrite(str(output_path), img)
        return

    if getattr(result, "boxes", None) is None or result.boxes is None:
        cv2.imwrite(str(output_path), img)
        return

    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes

    if masks is None or len(masks) == 0:
        cv2.imwrite(str(output_path), img)
        return

    h_img, w_img = img.shape[:2]

    barrier_mask = np.zeros((h_img, w_img), dtype=bool)
    guardrail_mask = np.zeros((h_img, w_img), dtype=bool)

    for idx, mask in enumerate(masks):
        class_id = None
        if getattr(boxes, "cls", None) is not None and idx < len(boxes.cls):
            class_id = int(boxes.cls[idx].item())

        if mask.shape != (h_img, w_img):
            mask_resized = cv2.resize(
                mask.astype(np.float32),
                (w_img, h_img),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            mask_resized = mask

        binary_mask = mask_resized > 0.5

        # Class Mapping Logic
        if class_id in [0, 1, 2, 3]:
            barrier_mask |= binary_mask
        elif class_id in [4, 5]:
            guardrail_mask |= binary_mask

    # پیدا کردن نواحی هم‌پوشانی (جایی که هم مانع بتنی و هم گاردریل تشخیص داده شده)
    overlap_mask = barrier_mask & guardrail_mask

    # جدا کردن نواحی اختصاصی (حذف نواحی هم‌پوشانی از ماسک‌های اصلی)
    exclusive_barrier = barrier_mask & ~overlap_mask
    exclusive_guardrail = guardrail_mask & ~overlap_mask

    overlay = img.copy()

    # OpenCV uses BGR: (Blue, Green, Red)
    barrier_color = (0, 255, 255)   # Yellow
    guardrail_color = (255, 0, 0)   # Blue
    overlap_color = (0, 0, 255)     # Red (برای دیدن گیجی مدل)

    overlay[exclusive_barrier] = barrier_color
    overlay[exclusive_guardrail] = guardrail_color
    overlay[overlap_mask] = overlap_color

    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.imwrite(str(output_path), blended)

def main():
    # ۱. گرفتن لیست تمام عکس‌ها بدون مشکل duplicate ویندوز
    image_paths = get_unique_image_paths(IMAGE_DIR)
    if not image_paths:
        print("No images found in the benchmark directory.")
        return
    
    print(f"Found {len(image_paths)} unique images. Starting batch processing...\n")

    # ۲. تنظیم و لود کردن مدل (فقط یک‌بار خارج از حلقه)
    overrides = {
        "model": str(MODEL_PATH),
        "conf": 0.25,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "quantize": 16 if torch.cuda.is_available() else 32,
        "save": False,  # جلوگیری از مصرف VRAM برای رسم پیش‌فرض
        "show": False
    }
    predictor = SAM3SemanticPredictor(overrides=overrides)

    # ۳. شروع حلقه برای پردازش تمام عکس‌ها
    for idx, image_path in enumerate(image_paths):
        print(f"[{idx + 1}/{len(image_paths)}] Processing: {image_path.name}")

        # ارسال عکس جدید به مدل
        predictor.set_image(str(image_path))

        # اجرای Inference
        results = predictor(text=prompts)

        # انتقال به CPU و آزادسازی VRAM
        r_cpu = results[0].cpu()
        del results
        torch.cuda.empty_cache()

        # ذخیره خروجی با استفاده از تابع Overlap ما
        output_image_path = OUTPUT_DIR / f"{image_path.stem}_semantic_2class.jpg"
        save_semantic_overlay_two_classes(r_cpu, output_image_path)

    print("\nBatch processing completed! Check the 'outputs/sam3_semantic_style' folder.")

if __name__ == "__main__":
    main()