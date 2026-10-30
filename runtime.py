"""Runtime selection without importing CUDA or Spaces in local mode."""

import os


def runtime_mode():
    mode = os.getenv("HI3D_RUNTIME", "auto").lower()
    if mode not in {"auto", "local", "zerogpu"}:
        raise ValueError("HI3D_RUNTIME must be auto, local, or zerogpu")
    hosted = os.getenv("SPACES_ZERO_GPU", "").lower() not in {"", "0", "false"}
    if mode == "local" and hosted:
        raise ValueError("Local mode cannot run on ZeroGPU hardware")
    return ("zerogpu" if hosted else "local") if mode == "auto" else mode


MODE = runtime_mode()


def gpu_task(function=None, *, duration=None):
    if function is None:
        return lambda fn: gpu_task(fn, duration=duration)
    if MODE == "zerogpu":
        import spaces

        return spaces.GPU(duration=duration or int(os.getenv("HI3D_GPU_SECONDS", "180")))(function)
    return function
