
from pathlib import Path
import pandas as pd

# Project and dataset paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "COD10K-v3"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_manifest(split):
    split_dir = DATASET_ROOT / split

    image_dir = split_dir / "Image"
    mask_dir = split_dir / "GT_Object"
    edge_dir = split_dir / "GT_Edge"

    records = []
    missing_files = []

    # Select only camouflaged images
    for image_path in sorted(image_dir.glob("COD10K-CAM-*.jpg")):
        mask_path = mask_dir / (image_path.stem + ".png")
        edge_path = edge_dir / (image_path.stem + ".png")

        if not mask_path.exists() or not edge_path.exists():
            missing_files.append(image_path.name)
            continue

        records.append({
            "image_path": str(image_path.relative_to(PROJECT_ROOT)),
            "mask_path": str(mask_path.relative_to(PROJECT_ROOT)),
            "edge_path": str(edge_path.relative_to(PROJECT_ROOT))
        })

    if missing_files:
        raise FileNotFoundError(
            f"{split}: Missing masks for {len(missing_files)} images. "
            f"Examples: {missing_files[:5]}"
        )

    df = pd.DataFrame(records)

    output_file = OUTPUT_DIR / f"{split.lower()}_manifest.csv"
    df.to_csv(output_file, index=False)

    print(f"\n{split} manifest created!")
    print("CAM samples:", len(df))
    print("Saved at:", output_file)


if __name__ == "__main__":
    create_manifest("Train")
    create_manifest("Test")
    print("\nDataset preparation completed successfully!")
