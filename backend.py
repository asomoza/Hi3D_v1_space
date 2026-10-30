"""Hunyuan3D-2mini turbo for the coarse shape and Hi3D v1.0 to refine it, on local CUDA and ZeroGPU."""

import gc
import os
import threading
from contextlib import contextmanager

from runtime import MODE, gpu_task


# Import Spaces before torch so ZeroGPU can install its runtime hooks.
if MODE == "zerogpu":
    import spaces  # noqa: F401
import torch

from diffusers import ModularPipelineBlocks


RESOLUTIONS = (512, 1024, 1536)
_LOCK = threading.RLock()
_coarse = None
_refiners = {}
_background = None


def load_coarse():
    global _coarse
    if _coarse is None:
        blocks = ModularPipelineBlocks.from_pretrained(
            os.getenv("HUNYUAN3D_BLOCKS", "OzzyGT/hunyuan3d_2mini_custom_blocks"),
            revision=os.getenv("HUNYUAN3D_BLOCKS_REVISION"),
            trust_remote_code=True,
            components_repo=os.getenv("HUNYUAN3D_TURBO_MODEL", "OzzyGT/hunyuan3d_2mini_turbo"),
            components_revision=os.getenv("HUNYUAN3D_TURBO_MODEL_REVISION"),
            trust_components_code=True,
        )
        blocks.sub_blocks["denoise"].sub_blocks["denoise"].set_progress_bar_config(desc="Coarse shape")
        pipe = blocks.init_pipeline()
        pipe.load_components(dtype=torch.float16)
        pipe.to("cuda")
        _coarse = pipe
    return _coarse


def _hi3d_blocks(resolution):
    return ModularPipelineBlocks.from_pretrained(
        os.getenv("HI3D_BLOCKS", "OzzyGT/hi3d_v1_custom_blocks"),
        revision=os.getenv("HI3D_BLOCKS_REVISION"),
        trust_remote_code=True,
        components_repo=os.getenv("HI3D_MODEL", "OzzyGT/hi3d_v1"),
        components_revision=os.getenv("HI3D_MODEL_REVISION"),
        trust_components_code=True,
        resolution=resolution,
    )


def load_refiners():
    if _refiners:
        return _refiners
    # The 1536 blockset uses every component; the other two share them.
    full = _hi3d_blocks(1536).init_pipeline()
    full.load_components(dtype=torch.float16)
    full.to("cuda")
    for resolution in RESOLUTIONS:
        if resolution == 1536:
            pipe = full
        else:
            pipe = _hi3d_blocks(resolution).init_pipeline()
            pipe.update_components(**{name: full.components[name] for name in pipe.component_names})
        _refiners[resolution] = pipe
    return _refiners


@contextmanager
def _untracked_progress():
    """Keep first-run download and loading bars out of Gradio's track_tqdm progress."""
    from gradio.context import LocalContext

    token = LocalContext.progress.set(None)
    try:
        yield
    finally:
        LocalContext.progress.reset(token)


def unload():
    global _coarse, _background
    with _LOCK:
        if MODE == "zerogpu":
            return "ZeroGPU releases the GPU after each request."
        _coarse = None
        _refiners.clear()
        _background = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return "Models unloaded."


@gpu_task
@torch.inference_mode()
def infer(image, coarse_mesh, resolution, seed, steps, num_faces, coarse_steps, octree_resolution):
    """The coarse mesh (generated unless given) and its refinement, as dicts with `vertices` and `faces`."""
    with _LOCK:
        with _untracked_progress():
            refiner = load_refiners()[resolution]
        if coarse_mesh is None:
            with _untracked_progress():
                coarse = load_coarse()
            coarse_mesh = coarse(
                image=image,
                num_inference_steps=coarse_steps,
                octree_resolution=octree_resolution,
                generator=torch.Generator().manual_seed(seed),
                output="meshes",
            )[0]
            if coarse_mesh is None:
                raise ValueError("No coarse surface was found. Try another seed or image.")
        refined = refiner(
            image=image,
            mesh=coarse_mesh,
            num_inference_steps=steps,
            num_faces=num_faces,
            generator=torch.Generator().manual_seed(seed),
            output="meshes",
        )[0]
        return coarse_mesh, refined


@gpu_task(duration=20)
@torch.inference_mode()
def remove_background(image):
    global _background
    from background import load_background
    from background import remove_background as matte

    with _LOCK:
        try:
            if _background is None:
                _background = load_background()
            if MODE == "local":
                _background.to("cuda")
            return matte(_background, image)
        finally:
            if MODE == "local":
                if _background is not None:
                    _background.to("cpu")
                gc.collect()
                torch.cuda.empty_cache()


if MODE == "zerogpu":
    from background import load_background

    load_coarse()
    load_refiners()
    _background = load_background().to("cuda")
