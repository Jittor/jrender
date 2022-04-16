import os
import jittor as jt
import numpy as np
from skimage.io import imread

from .load_textures import _load_textures_for_softras

def load_mtl(filename_mtl):
    '''
    load color (Kd) and filename of textures from *.mtl
    '''
    texture_filenames = {}
    colors = {}
    material_name = ''
    with open(filename_mtl) as f:
        for line in f.readlines():
            if len(line.split()) != 0:
                if line.split()[0] == 'newmtl':
                    material_name = line.split()[1]
                if line.split()[0] == 'map_Kd':
                    texture_filenames[material_name] = line.split()[1]
                if line.split()[0] == 'Kd':
                    colors[material_name] = np.array(list(map(float, line.split()[1:4])))
    return colors, texture_filenames


def load_textures(filename_obj, filename_mtl, texture_res):
    # load vertices
    vertices = []
    with open(filename_obj) as f:
        lines = f.readlines()
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'vt':
            vertices.append([float(v) for v in line.split()[1:3]])
    vertices = np.vstack(vertices).astype(np.float32)

    # load faces for textures
    faces = []
    material_names = []
    material_name = ''
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            if '/' in vs[0] and '//' not in vs[0]:
                v0 = int(vs[0].split('/')[1])
            else:
                v0 = 0
            for i in range(nv - 2):
                if '/' in vs[i + 1] and '//' not in vs[i + 1]:
                    v1 = int(vs[i + 1].split('/')[1])
                else:
                    v1 = 0
                if '/' in vs[i + 2] and '//' not in vs[i + 2]:
                    v2 = int(vs[i + 2].split('/')[1])
                else:
                    v2 = 0
                faces.append((v0, v1, v2))
                material_names.append(material_name)
        if line.split()[0] == 'usemtl':
            material_name = line.split()[1]
    faces = np.vstack(faces).astype(np.int32) - 1
    faces = vertices[faces]
    faces = jt.array(faces).float32()

    colors, texture_filenames = load_mtl(filename_mtl)

    textures = jt.ones((faces.shape[0], texture_res**2, 3), dtype="float32")

    for material_name, color in colors.items():
        color = jt.array(color).float32()
        for i, material_name_f in enumerate(material_names):
            if material_name == material_name_f:
                textures[i, :, :] = color.unsqueeze(0)

    for material_name, filename_texture in texture_filenames.items():
        filename_texture = os.path.join(os.path.dirname(filename_obj), filename_texture)
        image = imread(filename_texture).astype(np.float32) / 255.

        # texture image may have one channel (grey color)
        if len(image.shape) == 2:
            image = np.stack((image,)*3,-1)
        # or has extral alpha channel shoule ignore for now
        if image.shape[2] == 4:
            image = image[:,:,:3]

        image = image[::-1, :, :]
        image = jt.array(image.copy()).float32()
        is_update = (np.array(material_names) == material_name).astype(np.int32)
        is_update = jt.array(is_update).int()
        textures = _load_textures_for_softras(image, faces, textures, is_update)
    return textures

def load_obj(filename_obj, normalization=False, load_texture=False, texture_res=4, texture_type='surface'):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x) and faces (f x x x).
    """
    assert texture_type in ['surface', 'vertex']

    # load vertices
    vertices = []
    with open(filename_obj) as f:
        lines = f.readlines()

    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])

    vertices = jt.array(np.vstack(vertices)).float32()
    # load faces
    faces = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[0])
                v2 = int(vs[i + 2].split('/')[0])
                faces.append((v0, v1, v2))
    faces = jt.array(np.vstack(faces).astype(np.int32)).float32() - 1
    # load textures
    if load_texture and texture_type == 'surface':
        textures = None
        for line in lines:
            if line.startswith('mtllib'):
                filename_mtl = os.path.join(os.path.dirname(filename_obj), line.split()[1])
                textures = load_textures(filename_obj, filename_mtl, texture_res)
        if textures is None:
            raise Exception('Failed to load textures.')
    elif load_texture and texture_type == 'vertex':
        textures = []
        for line in lines:
            if len(line.split()) == 0:
                continue
            if line.split()[0] == 'v':
                textures.append([float(v) for v in line.split()[4:7]])
        textures = jt.array(np.vstack(textures)).float()
    # normalize into a unit cube centered zero
    if normalization:
        vertices -= vertices.min(0)
        vertices /= jt.abs(vertices).max()
        vertices *= 2
        vertices -= vertices.max(0) / 2
    if load_texture:
        return vertices, faces, textures
    else:
        return vertices, faces
