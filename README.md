---
title: Hi3D
emoji: 🗿
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
python_version: '3.12'
app_file: app.py
pinned: false
license: apache-2.0
short_description: A Gradio space to test Hi3D v1.0 mesh refinement
---
# Hi3D v1.0

Try it online on [Hugging Face Spaces](https://huggingface.co/spaces/OzzyGT/hi3d_v1).

A Gradio showcase of Hi3D v1.0 running on Modular Diffusers custom blocks: it refines a coarse 3D shape guided by the object image. Upload an object image, remove its background if needed, optionally upload a coarse mesh (otherwise Hunyuan3D-2mini turbo generates one), pick the resolution, generate, orbit both meshes, and download the refined GLB. [Powered by Hi3D 1.0](http://hi3d.ai/).

The resolution sets the number of refinement passes: 512 runs one, 1024 two and 1536 three, each on a finer grid. The face count is the target of the final simplification. Uploaded images, meshes and generated files are temporarily cached by Gradio; cache cleanup runs hourly for files older than one hour.

## Run locally

A CUDA GPU is required. Install [uv](https://docs.astral.sh/uv/), then:

```bash
git clone https://github.com/asomoza/Hi3D_v1_space
cd Hi3D_v1_space
uv run app.py
```

The blocks default to `OzzyGT/hi3d_v1_custom_blocks` and the model to `OzzyGT/hi3d_v1`. The Hi3D models load on the first Generate and stay on the GPU, next to Hunyuan3D-2mini turbo when it generates the coarse shape. **Unload models** releases them. Set `HI3D_BLOCKS` and `HI3D_MODEL` to other repository IDs or local folders, `HUNYUAN3D_BLOCKS` and `HUNYUAN3D_TURBO_MODEL` for the coarse model, and the same names with `_REVISION` to pin commits of their custom code. Only configure repositories whose code you trust.

## Provenance

Hi3D is licensed under the Hi3D Open License Agreement Version 1.0. Copyright © 2026 OCEANO HK HOLDING LIMITED. All rights not expressly granted are reserved. The Hi3D trademarks are owned by OCEANO HK HOLDING LIMITED or its applicable affiliate(s). Use must follow its Acceptable Use Policy; see the components repository's `LICENSE` and `NOTICE`. The coarse shapes come from [tencent/Hunyuan3D-2mini](https://huggingface.co/tencent/Hunyuan3D-2mini), licensed under the Tencent Hunyuan 3D 2.0 Community License Agreement, Copyright © 2025 Tencent. All Rights Reserved; powered by Tencent Hunyuan. Background removal uses the BEN2 base model from [PramaLLC/BEN2](https://huggingface.co/PramaLLC/BEN2) (MIT); see `ben2/LICENSE` and `ben2/NOTICE`. App code is [Apache-2.0](LICENSE).
