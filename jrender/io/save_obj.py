import os
import numpy as np
from skimage.io import imsave
import jittor as jt
from .utils import create_texture_image as create_texture_image_cuda

def create_texture_image(textures, texture_res=16):
    num_faces = textures.shape[0]
    tile_width = int((num_faces - 1.) ** 0.5) + 1
    tile_height = int((num_faces - 1.) / tile_width) + 1
    image = jt.ones((tile_height * texture_res, tile_width * texture_res, 3))
    vertices = jt.zeros((num_faces, 3, 2)) # [:, :, UV]
    face_nums = jt.array(np.arange(num_faces))
    column = face_nums % tile_width
    row = face_nums / tile_width
    vertices[:, 0, 0] = column * texture_res + texture_res / 2
    vertices[:, 0, 1] = row * texture_res + 1
    vertices[:, 1, 0] = column * texture_res + 1
    vertices[:, 1, 1] = (row + 1) * texture_res - 1 - 1
    vertices[:, 2, 0] = (column + 1) * texture_res - 1 - 1
    vertices[:, 2, 1] = (row + 1) * texture_res - 1 - 1
    image = create_texture_image_cuda.create_texture_image(vertices, textures, image, 1e-5)
    
    vertices[:, :, 0] /= (image.shape[1] - 1)
    vertices[:, :, 1] /= (image.shape[0] - 1)
    image = image.numpy()
    vertices = vertices.numpy()
    image = image[::-1, ::1]
    return image, vertices

def save_obj(filename, vertices, faces, textures=None, texture_res=16, texture_type='surface'):
    assert len(vertices.shape) == 2
    assert len(faces.shape) == 2
    assert texture_type in ['surface', 'vertex']
    assert texture_res >= 2

    filename_mtl = filename[:-4] + '.mtl'
    if textures is not None and texture_type == 'surface':
        filename_texture = filename[:-4] + '.png'
        material_name = 'material_1'
        texture_image, vertices_textures = create_texture_image(textures, texture_res)
        texture_image = texture_image.clip(0, 1)
        texture_image = (texture_image * 255).astype('uint8')
        imsave(filename_texture, texture_image)

    faces = faces.numpy()
    vertices = vertices.numpy()

    with open(filename, 'w') as f:
        f.write('# %s\n' % os.path.basename(filename))
        f.write('#\n')
        f.write('\n')

        if textures is not None:
            f.write('mtllib %s\n\n' % os.path.basename(filename_mtl))

        if textures is not None and texture_type == 'vertex':
            for vertex, color in zip(vertices, textures):
                f.write('v %.8f %.8f %.8f %.8f %.8f %.8f\n' % (vertex[0], vertex[1], vertex[2], 
                                                               color[0], color[1], color[2]))
            f.write('\n')
        else:
            for vertex in vertices:
                f.write('v %.8f %.8f %.8f\n' % (vertex[0], vertex[1], vertex[2]))
            f.write('\n')

        if textures is not None and texture_type == 'surface':
            for vertex in vertices_textures.reshape((-1, 2)):
                f.write('vt %.8f %.8f\n' % (vertex[0], vertex[1]))
            f.write('\n')

            f.write('usemtl %s\n' % material_name)
            for i, face in enumerate(faces):
                f.write('f %d/%d %d/%d %d/%d\n' % (
                    face[0] + 1, 3 * i + 1, face[1] + 1, 3 * i + 2, face[2] + 1, 3 * i + 3))
            f.write('\n')
        else:
            for face in faces:
                f.write('f %d %d %d\n' % (face[0] + 1, face[1] + 1, face[2] + 1))

    if textures is not None and texture_type == 'surface':
        with open(filename_mtl, 'w') as f:
            f.write('newmtl %s\n' % material_name)
            f.write('map_Kd %s\n' % os.path.basename(filename_texture))

def save_voxel(filename, voxel):
    vertices = []
    for i in range(voxel.shape[0]):
        for j in range(voxel.shape[1]):
            for k in range(voxel.shape[2]):
                if voxel[i, j, k] == 1:
                    vertices.append([i / voxel.shape[0], j / voxel.shape[1], k / voxel.shape[2]])
    vertices = jt.array(vertices)
    return save_obj(filename, vertices, jt.array([]))