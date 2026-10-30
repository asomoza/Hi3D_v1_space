import numpy as np
import pyfqmr
import trimesh


PREVIEW_FACES = 200_000


def to_trimesh(mesh):
    result = trimesh.Trimesh(mesh["vertices"], mesh["faces"], process=False)
    result.remove_unreferenced_vertices()
    return result


def preview_mesh(mesh, max_faces=PREVIEW_FACES):
    """A copy simplified to at most `max_faces` faces, light enough for the browser viewer."""
    if len(mesh.faces) <= max_faces:
        return mesh
    simplifier = pyfqmr.Simplify()
    simplifier.setMesh(mesh.vertices, mesh.faces)
    simplifier.simplify_mesh(target_count=max_faces, aggressiveness=5, preserve_border=True, verbose=False)
    vertices, faces, _ = simplifier.getMesh()
    return trimesh.Trimesh(vertices, faces, process=False)


def export_glb(mesh, path):
    """Write the mesh as GLB with a plain gray material and no metadata."""
    mesh = mesh.copy()
    colors = np.tile(np.array([200, 200, 200, 255], np.uint8), (len(mesh.vertices), 1))
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=colors)
    mesh.export(path, file_type="glb", include_normals=True)
