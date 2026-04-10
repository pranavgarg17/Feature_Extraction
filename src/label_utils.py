import pandas as pd
import os
import random


def clean_slide_id(name):
    """
    Remove extension and UUID if present
    Example:
    TCGA-XX-XXXX...DX1.abc123 → TCGA-XX-XXXX...DX1
    """
    # Remove the saved feature extension first.
    name = name.replace(".h5", "")

    # Drop any trailing UUID-like suffix after the first period so slide IDs
    # match between features and metadata tables.
    name = name.split(".")[0]
    return name


def create_labels(feature_dir, label_dir, real_labels_path=None):
    """
    Create label files required by HistoBistro

    Args:
        feature_dir: folder containing .h5 files
        label_dir: output folder
        real_labels_path: optional path to real labels file (xlsx/csv)

    Outputs:
        - slide.csv
        - clini_table.xlsx
    """

    # Make sure the destination folder exists for the generated label files.
    os.makedirs(label_dir, exist_ok=True)

    # Scan the feature directory and normalize every saved slide ID.
    slide_names = [
        clean_slide_id(f)
        for f in os.listdir(feature_dir)
        if f.endswith(".h5")
    ]

    # Keep slide ordering deterministic across runs.
    slide_names = sorted(slide_names)

    # Build the slide manifest expected by downstream tooling.
    slide_df = pd.DataFrame({
        "FILENAME": slide_names,
        "PATIENT": slide_names
    })

    # Save the slide-to-patient mapping table.
    slide_df.to_csv(os.path.join(label_dir, "slide.csv"), index=False)

    # Either load real molecular labels or synthesize placeholders for testing.
    if real_labels_path is not None:
        print("Using REAL labels")

        # Read either Excel or CSV label files based on the provided extension.
        if real_labels_path.endswith(".xlsx"):
            real_df = pd.read_excel(real_labels_path)
        else:
            real_df = pd.read_csv(real_labels_path)

        # Normalize slide IDs in the metadata file to match feature filenames.
        real_df["slide_id"] = real_df.iloc[:, 0].apply(clean_slide_id)

        # Create direct lookup maps for the mutation labels we need.
        label_map_kras = dict(zip(real_df["slide_id"], real_df["KRAS"]))
        label_map_braf = dict(zip(real_df["slide_id"], real_df["BRAF"]))

        # Populate output labels in the same order as the saved feature files.
        kras_labels = []
        braf_labels = []

        for s in slide_names:
            if s in label_map_kras:
                kras_labels.append(label_map_kras[s])
                braf_labels.append(label_map_braf[s])
            else:
                # Fall back to 0 if a slide is missing from the label source.
                print(f"Warning: Missing label for {s}, assigning 0")
                kras_labels.append(0)
                braf_labels.append(0)

    else:
        # Create dummy binary labels for quick end-to-end pipeline testing.
        print("Using RANDOM labels (only for testing)")
        kras_labels = [random.randint(0, 1) for _ in slide_names]
        braf_labels = [random.randint(0, 1) for _ in slide_names]

    # Assemble the clinical label table expected by HistoBistro.
    clini_df = pd.DataFrame({
        "PATIENT": slide_names,
        "KRAS": kras_labels,
        "BRAF": braf_labels
    })

    # Save the final patient-level label spreadsheet.
    clini_df.to_excel(os.path.join(label_dir, "clini_table.xlsx"), index=False)

    print("✅ Labels created successfully!")
