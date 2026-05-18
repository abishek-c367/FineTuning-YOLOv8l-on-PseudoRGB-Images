"""
CycleGAN generator loader for IR(thermal)->pseudoRGB.

Uses the architecture defined in CycleGAN_Project/src/models/cyclegan.py.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import ToPILImage
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
# Import torchvision.transforms for image transformations
import torchvision.transforms.v2 as transforms
from data.dataset import UnpairedDataset
import config as args
from Cycle_gan.cyclegan import Generator  # type: ignore


def load_cyclegan_generator(
    weights_path: str,
    device: Optional[torch.device] = None,
    input_nc: int = 3,
    output_nc: int = 3,
    n_residual_blocks: int = 9,
    direction: str = "B2A",  # "B2A" = thermal → RGB, "A2B" = RGB → thermal
) -> nn.Module:
    """
    Load a CycleGAN generator and correctly extract its weights.

    Args:
        weights_path: path to checkpoint
        device: torch device
        input_nc: input channels
        output_nc: output channels
        n_residual_blocks: generator depth
        direction:
            "B2A" → G_BA (thermal → RGB)
            "A2B" → G_AB (RGB → thermal)
    """

    # Initialize generator
    G = Generator(input_nc, output_nc, n_residual_blocks=n_residual_blocks)

    ckpt = torch.load(weights_path, map_location="cpu")

    # --------------------------------------------------
    # STEP 1: Extract model_state if full checkpoint
    # --------------------------------------------------
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        print(f"[Loader] Resuming from epoch={ckpt.get('epoch', '?')}, "
              f"global_step={ckpt.get('global_step', '?')}")
        state = ckpt["model_state"]
    else:
        state = ckpt  # already a state_dict

    # --------------------------------------------------
    # STEP 2: Select correct generator
    # --------------------------------------------------
    if direction == "B2A":
        prefix = "G_BA."
    elif direction == "A2B":
        prefix = "G_AB."
    else:
        raise ValueError("direction must be 'B2A' or 'A2B'")

    # Filter only generator weights
    filtered_state = {
        k.replace(prefix, ""): v
        for k, v in state.items()
        if k.startswith(prefix)
    }

    print(f"Using generator: {prefix}")
    # print("Filtered keys:", list(filtered_state.keys())[:10])

    # --------------------------------------------------
    # STEP 3: Load weights
    # --------------------------------------------------
    missing, unexpected = G.load_state_dict(filtered_state, strict=False)

    if missing:
        print("Missing keys:", missing[:5])
    if unexpected:
        print("Unexpected keys:", unexpected[:5])

    # --------------------------------------------------
    # STEP 4: Final setup
    # --------------------------------------------------
    G.eval()
    for p in G.parameters():
        p.requires_grad = False

    if device is not None:
        G.to(device)

    return G
def tensor_to_image(tensor):
    if tensor.dim() == 4:
        tensor = tensor[0]

    tensor = tensor.detach().cpu()

    # Denormalize [-1,1] → [0,1]
    tensor = (tensor + 1) / 2

    tensor = tensor.clamp(0, 1)
    return ToPILImage()(tensor)
import kornia.augmentation as K

gpu_normalize = K.Normalize(
    mean=torch.tensor([0.5, 0.5, 0.5]),
    std=torch.tensor([0.5, 0.5, 0.5]),
)



if __name__ == "__main__":
    # weights_path = "cyclegan_epoch_45.pth"
    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_nc = 3
    output_nc = 3
    n_residual_blocks = 9
    G = load_cyclegan_generator(args.cyclegan_weights, device=device, input_nc=input_nc, output_nc=output_nc, n_residual_blocks=n_residual_blocks, direction ="B2A")
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
    sample = dataset[2505]
    print("sample size before unsqueez: ",sample["B"].shape)
 
    with torch.no_grad():
        input_tensor = sample['B'].unsqueeze(0).to(device)
        print("sample size after unsqueez: ",input_tensor.shape)

        input_tensor = gpu_normalize(input_tensor)
        output = G(input_tensor)

   
    output_image = tensor_to_image(output)

    # #display both image together
    plt.subplot(1, 2, 1)
    plt.imshow(sample['B'].permute(1, 2, 0).numpy())
    plt.subplot(1, 2, 2)
    plt.imshow(output_image)
    plt.show()