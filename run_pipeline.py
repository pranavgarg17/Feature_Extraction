import os
from src.patch_extraction import extract_patches
from src.feature_extraction import CTransPathExtractor
from src.save_utils import save_h5
from src.label_utils import create_labels


def main():

    # Define paths
    raw_dir = "data/raw"
    feature_dir = "data/features"
    label_dir = "data/labels"

    # Initialize feature extractor
    extractor = CTransPathExtractor(device="cpu")

    # Loop through all slides
    for file in os.listdir(raw_dir):

        if not file.endswith(".svs"):
            continue

        slide_path = os.path.join(raw_dir, file)
        slide_name = file.replace(".svs", "")

        print(f"\nProcessing: {slide_name}")

        # ---------------------------
        # Step 1: Patch extraction
        # ---------------------------
        patches, coords = extract_patches(slide_path)

        print("Patches extracted:", len(patches))

        # ---------------------------
        # Step 2: Feature extraction
        # ---------------------------
        features = extractor.extract(patches)

        # ---------------------------
        # Step 3: Save features
        # ---------------------------
        save_path = os.path.join(feature_dir, f"{slide_name}.h5")
        save_h5(features, coords, save_path)

        print(f"Saved: {save_path}")

    # ---------------------------
    # Step 4: Create labels
    # ---------------------------
    create_labels(feature_dir, label_dir)

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()