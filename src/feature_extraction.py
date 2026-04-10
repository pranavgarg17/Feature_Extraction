import torch
import numpy as np
from torchvision import transforms
import sys, os

# Make the local TransPath package importable when the pipeline is run
# from the repository root.
sys.path.append(os.path.abspath("."))

from TransPath.ctran import ctranspath


class CTransPathExtractor:
    def __init__(self, device="cpu"):

        # Store the target device once so both model loading and inference
        # use the same CPU/GPU setting.
        self.device = device

        # Build the CTransPath architecture used for feature extraction.
        self.model = ctranspath()

        # Load the pretrained checkpoint and support both raw state dicts and
        # checkpoints that wrap weights under the "model" key.
        checkpoint = torch.load("ctranspath.pth", map_location=device)
        state_dict = checkpoint.get("model", checkpoint)

        # Load weights non-strictly so small checkpoint/model mismatches are
        # reported instead of crashing immediately.
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

        # Put the model in inference mode on the requested device.
        self.model.to(device)
        self.model.eval()

        # Convert numpy RGB patches into the 224x224 tensor format expected
        # by the backbone.
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def extract(self, patches):

        # Collect one feature vector per input patch.
        features = []

        for patch in patches:
            # Apply the image transform and add a batch dimension for the model.
            img = self.transform(patch).unsqueeze(0).to(self.device)

            # Disable gradients because we only need forward-pass embeddings.
            with torch.no_grad():
                feat = self.model(img)

            # Move the embedding back to CPU numpy format for downstream saving.
            features.append(feat.squeeze().cpu().numpy())

        # Return a single stacked array with shape (num_patches, feature_dim).
        return np.array(features)
