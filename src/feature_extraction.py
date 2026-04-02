import torch
import numpy as np
from torchvision import transforms
import sys, os

# ensure TransPath is visible
sys.path.append(os.path.abspath("."))

from TransPath.ctran import ctranspath


class CTransPathExtractor:
    def __init__(self, device="cpu"):

        self.device = device

        # ✅ Correct model (NO timm)
        self.model = ctranspath()

        # ✅ Load weights
        checkpoint = torch.load("ctranspath.pth", map_location=device)
        state_dict = checkpoint.get("model", checkpoint)

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

        self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def extract(self, patches):

        features = []

        for patch in patches:
            img = self.transform(patch).unsqueeze(0).to(self.device)

            with torch.no_grad():
                feat = self.model(img)

            features.append(feat.squeeze().cpu().numpy())

        return np.array(features)