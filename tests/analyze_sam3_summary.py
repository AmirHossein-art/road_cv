from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sam3_text_guardrail"
CSV_PATH = OUTPUT_DIR / "summary.csv"

CHART_DIR = OUTPUT_DIR / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(CSV_PATH)

    print("Rows:", len(df))
    print("Images with at least one mask:", (df["num_masks"] > 0).sum())
    print("Detection presence rate:", (df["num_masks"] > 0).mean())

    print("\nBasic statistics:")
    print(df[["num_masks", "num_boxes", "avg_conf", "max_conf", "total_mask_area_ratio"]].describe())

    # نمودار ۱: تعداد mask در هر عکس
    plt.figure(figsize=(12, 5))
    plt.bar(df["image"], df["num_masks"])
    plt.xticks(rotation=90)
    plt.xlabel("Image")
    plt.ylabel("Number of masks")
    plt.title("SAM 3 - Number of Detected Masks per Image")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "num_masks_per_image.png", dpi=200)
    plt.close()

    # نمودار ۲: میانگین confidence در هر عکس
    if "avg_conf" in df.columns:
        plt.figure(figsize=(12, 5))
        plt.bar(df["image"], df["avg_conf"])
        plt.xticks(rotation=90)
        plt.xlabel("Image")
        plt.ylabel("Average confidence")
        plt.title("SAM 3 - Average Confidence per Image")
        plt.tight_layout()
        plt.savefig(CHART_DIR / "avg_conf_per_image.png", dpi=200)
        plt.close()

    # نمودار ۳: درصد مساحت mask نسبت به تصویر
    if "total_mask_area_ratio" in df.columns:
        plt.figure(figsize=(12, 5))
        plt.bar(df["image"], df["total_mask_area_ratio"])
        plt.xticks(rotation=90)
        plt.xlabel("Image")
        plt.ylabel("Total mask area ratio")
        plt.title("SAM 3 - Total Mask Area Ratio per Image")
        plt.tight_layout()
        plt.savefig(CHART_DIR / "mask_area_ratio_per_image.png", dpi=200)
        plt.close()

    # خلاصه متنی
    report_path = OUTPUT_DIR / "quick_report.txt"

    presence_rate = (df["num_masks"] > 0).mean()
    avg_masks = df["num_masks"].mean()
    avg_conf = df["avg_conf"].mean()

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("SAM 3 Preliminary Output Statistics\n")
        f.write("===================================\n\n")
        f.write(f"Number of images: {len(df)}\n")
        f.write(f"Images with at least one detected mask: {(df['num_masks'] > 0).sum()}\n")
        f.write(f"Detection presence rate: {presence_rate:.2%}\n")
        f.write(f"Average number of masks per image: {avg_masks:.2f}\n")
        f.write(f"Average confidence: {avg_conf:.3f}\n\n")
        f.write("Important note:\n")
        f.write(
            "These values are model-output statistics, not true accuracy metrics. "
            "To compute true accuracy, ground-truth annotations are required.\n"
        )

    print(f"Charts saved to: {CHART_DIR}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()