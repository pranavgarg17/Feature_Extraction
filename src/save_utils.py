import h5py
import numpy as np
import os


def save_h5(features, coords, save_path):
    """
    Save features + coordinates in HDF5 format

    Required format for HistoBistro:
        features → (N, 768)
        coords   → (N, 2)
    """

    # Create directory if not exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with h5py.File(save_path, "w") as f:

        # Save features
        f.create_dataset("features", data=features)

        # Save coordinates
        f.create_dataset("coords", data=np.array(coords))