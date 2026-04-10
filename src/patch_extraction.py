import cv2
import numpy as np
import openslide


# Default extraction settings for ViT-style patch encoders and WSI tissue
# pre-filtering.
DEFAULT_PATCH_SIZE = 224
DEFAULT_TARGET_MPP = 0.5
DEFAULT_TISSUE_THRESHOLD = 0.5
DEFAULT_MASK_DOWNSAMPLE = 32


def _safe_float(value):
    # Convert slide metadata values to floats while tolerating missing keys
    # and malformed strings.
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_slide_mpp(slide):
    """
    Return the native level-0 microns-per-pixel for the slide when possible.
    Falls back to objective power when explicit MPP metadata is missing.
    """

    properties = slide.properties

    # Prefer explicit microns-per-pixel metadata when the scanner provides it.
    mpp_x = _safe_float(properties.get(openslide.PROPERTY_NAME_MPP_X))
    mpp_y = _safe_float(properties.get(openslide.PROPERTY_NAME_MPP_Y))

    if mpp_x and mpp_y:
        return (mpp_x + mpp_y) / 2.0

    # Fall back to objective power when MPP is absent.
    objective_power = _safe_float(
        properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    )
    if objective_power and objective_power > 0:
        # Approximation commonly used in digital pathology:
        # 40x -> 0.25 um/px, 20x -> 0.5 um/px.
        return 10.0 / objective_power

    return None


def generate_tissue_mask(
    slide,
    mask_downsample=DEFAULT_MASK_DOWNSAMPLE,
    min_tissue_saturation=20,
    white_threshold=220,
):
    """
    Build a low-resolution tissue mask by removing white background regions.
    """

    # Pick a coarse slide level so tissue detection runs on a thumbnail rather
    # than on the full-resolution image.
    level = slide.get_best_level_for_downsample(mask_downsample)
    level_downsample = slide.level_downsamples[level]
    level_width, level_height = slide.level_dimensions[level]

    # Read the downsampled overview image used for mask generation.
    thumbnail = slide.read_region((0, 0), level, (level_width, level_height))
    thumbnail = np.array(thumbnail)[:, :, :3]

    # Combine grayscale brightness and HSV saturation to distinguish tissue
    # from mostly white background regions.
    hsv = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2GRAY)

    # Keep non-white, non-empty regions as candidate tissue.
    not_white = gray < white_threshold
    saturated = hsv[:, :, 1] > min_tissue_saturation
    tissue_mask = np.logical_or(not_white, saturated).astype(np.uint8) * 255

    # Clean the mask so tiny isolated artifacts and holes do not affect
    # patch selection.
    kernel = np.ones((3, 3), np.uint8)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)

    return tissue_mask, level_downsample


def patch_has_tissue(mask, x, y, read_size, mask_downsample, threshold):
    """
    Check whether the corresponding region in the low-resolution mask contains
    enough tissue to keep the patch.
    """

    # Map the level-0 patch window into the low-resolution mask coordinates.
    x0 = int(x / mask_downsample)
    y0 = int(y / mask_downsample)
    x1 = int(np.ceil((x + read_size) / mask_downsample))
    y1 = int(np.ceil((y + read_size) / mask_downsample))

    # Clamp the mask window to valid image bounds.
    x0 = max(0, min(x0, mask.shape[1]))
    x1 = max(0, min(x1, mask.shape[1]))
    y0 = max(0, min(y0, mask.shape[0]))
    y1 = max(0, min(y1, mask.shape[0]))

    if x1 <= x0 or y1 <= y0:
        return False

    # Keep the patch only if enough of its area overlaps tissue.
    region = mask[y0:y1, x0:x1]
    tissue_ratio = np.mean(region > 0)
    return tissue_ratio >= threshold


def get_read_size(slide, patch_size, target_mpp):
    # Compute how large a level-0 crop is needed to approximate the requested
    # physical magnification before resizing to the model input size.
    native_mpp = get_slide_mpp(slide)

    if native_mpp:
        scale = target_mpp / native_mpp
        read_size = max(1, int(round(patch_size * scale)))
    else:
        read_size = patch_size

    return read_size


def read_patch_at_target_mpp(slide, x, y, patch_size, target_mpp):
    """
    Read a patch whose physical resolution is as close as possible to the
    requested target MPP, then resize it to the model input size.
    """

    # Read a physically matched crop size from level 0.
    read_size = get_read_size(slide, patch_size, target_mpp)

    patch = slide.read_region((x, y), 0, (read_size, read_size))
    patch = np.array(patch)[:, :, :3]

    # Normalize the output shape so the feature extractor always receives
    # 224x224 RGB patches.
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = cv2.resize(
            patch,
            (patch_size, patch_size),
            interpolation=cv2.INTER_AREA,
        )

    return patch, read_size


def extract_patches(
    slide_path,
    patch_size=DEFAULT_PATCH_SIZE,
    stride=DEFAULT_PATCH_SIZE,
    max_patches=2000,
    target_mpp=DEFAULT_TARGET_MPP,
    tissue_threshold=DEFAULT_TISSUE_THRESHOLD,
    mask_downsample=DEFAULT_MASK_DOWNSAMPLE,
):
    """
    Extract grid-based patches from tissue regions only.

    The pipeline follows three rules:
    1. Build a low-resolution tissue mask to remove white background.
    2. Extract only patches that overlap tissue.
    3. Read patches at approximately 0.5 um/px (about 20x) when metadata allows.
    """

    # Open the whole-slide image once and reuse the handle for mask generation
    # and patch reads.
    slide = openslide.OpenSlide(slide_path)

    try:
        # Build the tissue mask up front so background regions can be skipped
        # cheaply during the grid scan.
        tissue_mask, effective_mask_downsample = generate_tissue_mask(
            slide, mask_downsample=mask_downsample
        )

        # Accumulate extracted RGB patches and their top-left coordinates.
        patches = []
        coords = []

        # Use the level-0 slide dimensions as the grid reference frame.
        width, height = slide.dimensions
        read_size = get_read_size(slide, patch_size, target_mpp)

        # Scan the slide on a regular grid and keep only tissue-containing
        # windows.
        for y in range(0, max(0, height - patch_size + 1), stride):
            for x in range(0, max(0, width - patch_size + 1), stride):
                # Skip border locations where a full physical crop would run
                # outside the slide bounds.
                if x + read_size > width or y + read_size > height:
                    continue

                # Reject background windows using the low-resolution mask before
                # performing the more expensive patch read.
                if not patch_has_tissue(
                    tissue_mask,
                    x,
                    y,
                    read_size=read_size,
                    mask_downsample=effective_mask_downsample,
                    threshold=tissue_threshold,
                ):
                    continue

                # Read and resize the accepted patch for feature extraction.
                patch, _ = read_patch_at_target_mpp(
                    slide,
                    x,
                    y,
                    patch_size=patch_size,
                    target_mpp=target_mpp,
                )

                patches.append(patch)
                coords.append((x, y))

                # Stop early once the requested patch budget is reached.
                if len(patches) >= max_patches:
                    return patches, coords

        return patches, coords
    finally:
        # Always release the slide handle even if extraction fails midway.
        slide.close()
