"""
Ultralytics YOLOv8 teacher feature extractor.

Ultralytics YOLO models do not expose `return_features=True` in a stable way.
For distillation, the most version-robust method is to hook the Detect head's
inputs: Detect receives the list of FPN feature maps (typically P3, P4, P5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from Cycle_gan.cyclegan_loader import *
from data.dataset import UnpairedDataset


@dataclass(frozen=True)
class TeacherFeatureSpec:
    # Map keys used by DistillationTrainer to feature indices.
    # Ultralytics YOLOv8 Detect head is typically fed [P3, P4, P5].
    keys: Tuple[str, ...] = ("P3", "P4", "P5")


class TeacherYOLOv8FeatureExtractor(nn.Module):
    """
    Wrapper around an Ultralytics YOLOv8 model that exposes FPN feature maps.

    Usage:
        teacher = TeacherYOLOv8FeatureExtractor(weights="yolov8l.pt").eval()
        out = teacher(pseudo_rgb)  # returns {"fused_features": {"P3":..., ...}}
    """

    def __init__(
        self,
        weights: str = "yolov8l.pt",
        device: Optional[torch.device] = None,
        feature_spec: TeacherFeatureSpec = TeacherFeatureSpec(),
    ) -> None:
        super().__init__()

        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Ultralytics is required to use the YOLOv8 teacher.\n"
                "Install with: pip install ultralytics"
            ) from e

        self.feature_spec = feature_spec

        yolo = YOLO(weights)
        model = yolo.model
        mm = model.model  # the Sequential

        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        self.model = model
        if device is not None:
            self.model.to(device)

        self._cached_feats: Optional[List[torch.Tensor]] = None
        self._register_detect_input_hook()

    def _register_detect_input_hook(self) -> None:
        # Ultralytics YOLO models are usually a Sequential-like `model.model`
        detect = None
        mm = getattr(self.model, "model", None)
        for i, layer in enumerate(mm):
            if type(layer).__name__ == "Detect":
                detect = layer
                break

        if detect is None:
            raise RuntimeError("Could not locate Detect module in Ultralytics model.")

        def _pre_hook(_module, inputs):
            # Detect.forward usually gets one positional arg: list[Tensor]
            if not inputs:
                self._cached_feats = None
                return
            x = inputs[0]
            if isinstance(x, (list, tuple)) and all(isinstance(t, torch.Tensor) for t in x):
                self._cached_feats = list(x)
            else:
                self._cached_feats = None

        detect.register_forward_pre_hook(_pre_hook)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Args:
            x: pseudo-RGB tensor in range [0, 1], shape (B, 3, H, W)

        Returns:
            {"fused_features": {"P3": Tensor, "P4": Tensor, "P5": Tensor}}
        """
        self._cached_feats = None
        _ = self.model(x)

        feats = self._cached_feats
        if feats is None:
            raise RuntimeError(
                "Teacher feature hook did not capture Detect inputs. "
                "Ultralytics version/model structure may differ."
            )

        keys = self.feature_spec.keys
        if len(feats) < len(keys):
            raise RuntimeError(
                f"Expected at least {len(keys)} teacher feature maps, got {len(feats)}."
            )

        fused = {k: feats[i] for i, k in enumerate(keys)}
        return {"fused_features": fused}

    @torch.no_grad()
    def infer_channels(self, input_size: Tuple[int, int] = (640, 640)) -> Dict[str, int]:
        """
        Convenience helper to determine teacher feature channels by running one forward.
        """
        device = next(self.model.parameters()).device
        dummy = torch.zeros(1, 3, input_size[0], input_size[1], device=device)
        out = self.forward(dummy)
        feats = out["fused_features"]
        return {k: int(v.shape[1]) for k, v in feats.items()}
if __name__ == "__main__":
    import config as args
    weights_path = args.cyclegan_weights
    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_nc = 3
    output_nc = 3
    n_residual_blocks = 9
    G = load_cyclegan_generator(weights_path, device=device, input_nc=input_nc, output_nc=output_nc, n_residual_blocks=n_residual_blocks, direction ="B2A")
    # print(G)
    dataset = UnpairedDataset(
        'C:/Project/Project_Detection/FLIR_ADAS_v2/images_rgb_train/data',
        'C:/Project/Project_Detection/FLIR_ADAS_v2/images_thermal_train/data',
        transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize((512, 512)),
        ])
    )
    sample = dataset[250]

    with torch.no_grad():
        input_tensor = sample['B'].unsqueeze(0).to(device)
        input_tensor = gpu_normalize(input_tensor)
        output = G(input_tensor)
    output = (output + 1) / 2
    output = output.clamp(0, 1)

    Teacher = TeacherYOLOv8FeatureExtractor(weights =args.teacher_weights)
  
    features = Teacher(output)



    # output_image = tensor_to_image(output)
    for key,value in features.items():
        print(key,value)

