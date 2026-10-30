import secrets
import tempfile
import time
from pathlib import Path

import gradio as gr
import trimesh
from backend import RESOLUTIONS, infer, remove_background, unload
from mesh import export_glb, to_trimesh
from PIL import ImageOps
from runtime import MODE


DEFAULT_FACES = {512: 1_000_000, 1024: 1_000_000, 1536: 2_000_000}


def load_mesh(path):
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise gr.Error("The coarse mesh file has no triangles.")
    return {"vertices": mesh.vertices, "faces": mesh.faces}


def cache_glb(mesh, name):
    # Gradio copies returned files to its managed cache before this directory is removed.
    with tempfile.TemporaryDirectory(prefix="hi3d_") as directory:
        path = Path(directory) / name
        export_glb(mesh, path)
        from gradio.processing_utils import save_file_to_cache

        return save_file_to_cache(str(path), gr.utils.get_upload_folder())


def generate(
    image,
    coarse_file,
    coarse_was_generated,
    resolution,
    seed,
    randomize,
    steps,
    num_faces,
    coarse_steps,
    octree_resolution,
    progress=gr.Progress(track_tqdm=True),
):
    if image is None:
        raise gr.Error("Upload an image first.")
    resolution = int(resolution)
    seed = secrets.randbelow(2**31) if randomize else int(seed)
    if (
        resolution not in RESOLUTIONS
        or not 0 <= seed < 2**31
        or not 1 <= int(steps) <= 50
        or not 10_000 <= int(num_faces) <= 4_000_000
        or not 1 <= int(coarse_steps) <= 50
        or not 64 <= int(octree_resolution) <= 512
    ):
        raise gr.Error("Generation settings are outside the supported range.")
    image = ImageOps.exif_transpose(image).convert("RGBA")
    if image.getextrema()[3][1] == 0:
        raise gr.Error("The image is fully transparent.")
    coarse_mesh = load_mesh(coarse_file) if coarse_file else None
    generated = coarse_mesh is None
    started = time.monotonic()
    progress(0.02, desc="Generating the coarse shape" if generated else "Refining")
    try:
        coarse_mesh, refined = infer(
            image, coarse_mesh, resolution, seed, int(steps), int(num_faces), int(coarse_steps), int(octree_resolution)
        )
    except ValueError as error:
        raise gr.Error(str(error)) from error
    progress(0.9, desc="Exporting meshes")
    # a generated coarse mesh goes into the coarse mesh input, where an uploaded one is shown
    coarse = cache_glb(to_trimesh(coarse_mesh), "coarse.glb") if generated else gr.update()
    refined_mesh = to_trimesh(refined)
    refined_path = cache_glb(refined_mesh, "refined.glb")
    progress(1, desc="Ready")
    info = f"{resolution}, seed {seed}: {len(refined_mesh.faces):,} triangles in {time.monotonic() - started:.1f} s"
    return (
        coarse,
        generated or coarse_was_generated,
        refined_path,
        gr.update(value=refined_path, interactive=True),
        seed,
        info,
    )


def remove_source_background(image):
    if image is None:
        raise gr.Error("Upload an image first.")
    try:
        return remove_background(image)
    except ValueError as error:
        raise gr.Error(str(error)) from error


CSS = """
#remove-background { background: #2f6f4f; border-color: #3b8762; color: white; }
#remove-background:hover { background: #378159; }
#unload-models { background: #a32626; border-color: #bd3535; color: white; }
#unload-models:hover { background: #bf3030; }
"""


def build_demo():
    with gr.Blocks(title="Hi3D", delete_cache=(3600, 3600)) as demo:
        gr.Markdown(
            "# Hi3D v1.0\nRefines a coarse 3D shape guided by its image, with Modular Diffusers. "
            "[Powered by Hi3D 1.0](http://hi3d.ai/)."
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    run = gr.Button("Generate", variant="primary", scale=3)
                    if MODE == "local":
                        release = gr.Button("Unload models", variant="stop", scale=1, elem_id="unload-models")
                source = gr.Image(type="pil", image_mode="RGBA", label="Object image")
                with gr.Row():
                    remove_bg = gr.Button("Remove background", elem_id="remove-background")
                coarse_file = gr.Model3D(
                    label="Coarse mesh (optional)", height=320, clear_color=(0.08, 0.09, 0.12, 1), interactive=True
                )
                # whether the coarse mesh input holds a generated mesh, which a new image clears
                coarse_generated = gr.State(False)
                gr.Markdown(
                    "Use an object on a transparent background, or remove the background with BEN2. Drop a coarse "
                    "mesh (GLB, OBJ, PLY or STL) to refine it; without one, Hunyuan3D-2mini turbo generates it from "
                    "the image first and shows it here."
                )
                with gr.Accordion("Refinement settings", open=True):
                    resolution = gr.Radio([str(value) for value in RESOLUTIONS], value="1024", label="Resolution")
                    steps = gr.Slider(1, 50, value=25, step=1, label="Steps per pass")
                    num_faces = gr.Number(value=DEFAULT_FACES[1024], precision=0, label="Face count")
                    seed = gr.Number(value=0, precision=0, label="Seed", interactive=False)
                    randomize = gr.Checkbox(value=True, label="Randomize seed")
                with gr.Accordion("Coarse shape settings (Hunyuan3D-2mini turbo)", open=False):
                    coarse_steps = gr.Slider(1, 50, value=30, step=1, label="Steps")
                    octree_resolution = gr.Slider(64, 512, value=380, step=4, label="Octree resolution")
            with gr.Column():
                preview = gr.Model3D(label="Refined mesh", height=640, clear_color=(0.08, 0.09, 0.12, 1))
                info = gr.Markdown("")
                download = gr.DownloadButton(label="Download refined GLB", interactive=False)
                gr.Markdown("The GLB contains the untextured shape only; no input image or generation settings.")
        remove_bg.click(
            remove_source_background,
            source,
            source,
            concurrency_id="model",
            concurrency_limit=1,
            show_progress_on=[source],
            api_name="remove_background",
        )
        source.upload(
            lambda generated: (None, False) if generated else (gr.update(), False),
            coarse_generated,
            [coarse_file, coarse_generated],
            queue=False,
        )
        coarse_file.upload(lambda: False, None, coarse_generated, queue=False)
        coarse_file.clear(lambda: False, None, coarse_generated, queue=False)
        randomize.change(lambda checked: gr.update(interactive=not checked), randomize, seed, queue=False)
        resolution.change(lambda value: DEFAULT_FACES[int(value)], resolution, num_faces, queue=False)
        run.click(
            generate,
            [
                source,
                coarse_file,
                coarse_generated,
                resolution,
                seed,
                randomize,
                steps,
                num_faces,
                coarse_steps,
                octree_resolution,
            ],
            [coarse_file, coarse_generated, preview, download, seed, info],
            show_progress_on=[preview],
            concurrency_limit=1,
            concurrency_id="model",
            api_name="generate",
        )
        if MODE == "local":
            release.click(lambda: gr.Info(unload()), concurrency_id="model", concurrency_limit=1)
    return demo.queue(max_size=8)


if __name__ == "__main__":
    build_demo().launch(css=CSS)
