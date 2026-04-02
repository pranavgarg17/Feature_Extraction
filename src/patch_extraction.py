import openslide
import numpy as np
import cv2


def is_tissue(patch, threshold=0.5):
    """
    Check if patch contains enough tissue using Otsu thresholding

    Args:
        patch (numpy array): RGB image
        threshold (float): minimum tissue ratio required

    Returns:
        True if patch contains tissue
    """

    # Convert RGB → Grayscale
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

    # Apply Otsu threshold
    _, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Invert mask (tissue = white)
    mask = 255 - mask

    # Calculate tissue ratio
    tissue_ratio = np.sum(mask > 0) / mask.size

    return tissue_ratio > threshold


def extract_patches(slide_path, patch_size=256, stride=256, max_patches=2000):
    """
    Extract tissue patches using Otsu masking
    """

    slide = openslide.OpenSlide(slide_path)

    patches = []
    coords = []

    width, height = slide.dimensions

    for x in range(0, width, stride):
        for y in range(0, height, stride):

            patch = slide.read_region((x, y), 0, (patch_size, patch_size))
            patch = np.array(patch)[:, :, :3]

            # 🔥 OTSU FILTER
            if not is_tissue(patch):
                continue

            patches.append(patch)
            coords.append((x, y))

            if len(patches) >= max_patches:
                return patches, coords

    return patches, coords