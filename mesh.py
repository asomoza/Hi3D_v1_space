import numpy as np
import trimesh


def to_trimesh(mesh):
    result = trimesh.Trimesh(mesh["vertices"], mesh["faces"], process=False)
    result.remove_unreferenced_vertices()
    return result


def export_glb(mesh, path):
    """Write the mesh as GLB with a plain gray material and no metadata."""
    mesh = mesh.copy()
    colors = np.tile(np.array([200, 200, 200, 255], np.uint8), (len(mesh.vertices), 1))
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=colors)
    mesh.export(path, file_type="glb", include_normals=True)
