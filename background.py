"""BEN2 background removal for the source image."""

import os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors.torch import load_file


REVISION = "e48a20765fb421d19dcdb0bf3cc61e802ca5ec8f"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_background():
    from ben2.model import BEN_Base

    source = os.getenv("HI3D_BACKGROUND_MODEL", "PramaLLC/BEN2")
    path = Path(source)
    if not path.is_dir():
        path = Path(
            snapshot_download(
                source,
                revision=os.getenv("HI3D_BACKGROUND_REVISION", REVISION),
                allow_patterns=["model.safetensors"],
            )
        )
    model = BEN_Base()
    model.load_state_dict(load_file(str(path / "model.safetensors")), strict=True)
    return model.eval()


def remove_background(model, image):
    """Return the image as RGBA with BEN2's alpha matte, at its original size."""
    rgb = image.convert("RGB")
    pixels = np.asarray(rgb.resize((1024, 1024), Image.Resampling.LANCZOS), dtype=np.float32) / 255
    # Normalize and post-process in float16 like upstream's torchvision pipeline; this matches its matte exactly.
    pixels = torch.from_numpy(pixels).permute(2, 0, 1).to("cuda", torch.float16)
    mean = torch.tensor(MEAN, dtype=torch.float16, device="cuda").view(3, 1, 1)
    std = torch.tensor(STD, dtype=torch.float16, device="cuda").view(3, 1, 1)
    alpha = model(((pixels - mean) / std)[None])
    alpha = torch.nn.functional.interpolate(alpha, size=(rgb.height, rgb.width), mode="bilinear")[0, 0]
    alpha = (alpha - alpha.min()) / (alpha.max() - alpha.min())
    matte = Image.fromarray((alpha * 255).cpu().numpy().astype(np.uint8))
    result = rgb.copy()
    result.putalpha(matte)
    return result
