import os
from collections import OrderedDict
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class BiasConsistencyDetector(nn.Module):
    """
    Detector implementation based on training/detectors/BiasConsistency.py
    from https://github.com/thanhquan123hi1/BiasConsist
    Uses CLIP ViT-L/14 with bias adaptation + normalized features + Linear(1024, 2).
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        feature_dim: int = 1024,
        normalize_eps: float = 1e-6,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.normalize_eps = normalize_eps
        self.clip_model_name = clip_model_name

        try:
            self._processor = CLIPProcessor.from_pretrained(clip_model_name)
        except Exception:
            self._processor = CLIPProcessor.from_pretrained(clip_model_name, local_files_only=True)

        try:
            clip_model = CLIPModel.from_pretrained(clip_model_name)
        except Exception:
            clip_model = CLIPModel.from_pretrained(clip_model_name, local_files_only=True)

        self.backbone = clip_model.vision_model
        self.head = nn.Linear(self.feature_dim, 2)

    def preprocess(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """Preprocess PIL Image into CLIP input tensor [3, 224, 224]."""
        if isinstance(image, Image.Image):
            return self._processor(images=image, return_tensors="pt")["pixel_values"][0]
        return image

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            inputs: Tensor of shape [B, 3, 224, 224]
        Returns:
            logits: Tensor of shape [B, 2]
        """
        outputs = self.backbone(inputs)
        raw_features = outputs.pooler_output
        normalized_features = F.normalize(
            raw_features,
            p=2,
            dim=1,
            eps=self.normalize_eps,
        )
        logits = self.head(normalized_features)
        return logits

    @classmethod
    def load_from_checkpoint(
        cls,
        weights_path: str = "weights/BiasConsist/bias_consistency.pth",
        device: str = "cpu",
        clip_model_name: str = "openai/clip-vit-large-patch14",
    ) -> "BiasConsistencyDetector":
        """Load pretrained detector weights."""
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"BiasConsistency checkpoint not found: {weights_path}")

        detector = cls(clip_model_name=clip_model_name)
        checkpoint = torch.load(weights_path, map_location="cpu")

        if isinstance(checkpoint, dict):
            for subkey in ("state_dict", "model_state_dict", "model", "net"):
                if subkey in checkpoint and isinstance(checkpoint[subkey], (dict, OrderedDict)):
                    checkpoint = checkpoint[subkey]
                    break

        cleaned_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            clean_k = k
            for prefix in ("module.", "model."):
                if clean_k.startswith(prefix):
                    clean_k = clean_k[len(prefix):]
            cleaned_state_dict[clean_k] = v

        msg = detector.load_state_dict(cleaned_state_dict, strict=False)
        print(f"BiasConsistencyDetector loaded from {weights_path}")
        if msg.missing_keys:
            print(f"Missing keys: {len(msg.missing_keys)} (e.g. {msg.missing_keys[:3]})")
        if msg.unexpected_keys:
            print(f"Unexpected keys: {len(msg.unexpected_keys)} (e.g. {msg.unexpected_keys[:3]})")

        detector.eval()
        detector.to(device)
        return detector
