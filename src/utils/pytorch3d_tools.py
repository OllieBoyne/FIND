from pytorch3d.structures import Meshes
import trimesh
from pytorch3d.renderer import TexturesVertex

def extend_template(meshes: Meshes, N=1):
	"""Extend single mesh to multiple meshes """
	verts = meshes.verts_padded().expand(N, -1, -1).to(meshes.device)
	faces = meshes.faces_padded().expand(N, -1, -1).to(meshes.device)

	tex = None
	if meshes.textures is not None:
		tex = meshes.textures.expand(N, -1, -1)

	return Meshes(verts=verts, faces=faces, textures=tex)


def to_trimesh(meshes: Meshes, idx=0, include_texture=True):
	"""Converts meshes to a single trimesh"""

	verts, faces = meshes.verts_padded().cpu().detach().numpy()[idx], meshes.faces_padded().cpu().detach().numpy()[idx]

	mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

	if include_texture:
		if isinstance(meshes.textures, TexturesVertex):
			mesh.visual = trimesh.visual.color.ColorVisuals(vertex_colors = meshes.textures.verts_features_padded().cpu().detach().numpy()[idx])
		else:
			raise NotImplementedError(f"Cannot export PyTorch3D Mesh to Trimesh with texture type {type(meshes.textures)}")

	return mesh

def export_mesh(mesh, export_loc, include_texture=True):
	"""Render displacement example AND template mesh"""

	mesh = to_trimesh(mesh, include_texture=True)
	obj_data = trimesh.exchange.obj.export_obj(mesh, include_color=True)
	with open(export_loc, 'w') as outfile:
		outfile.write(obj_data)