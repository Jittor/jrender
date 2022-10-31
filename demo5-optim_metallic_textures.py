import os
import argparse
import glob

import jittor as jt
from jittor import nn
import numpy as np
from skimage.io import imread, imsave
import tqdm
import imageio

import jrender as jr
jt.flags.use_cuda = 1

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')
np.random.seed(1)

class Model(nn.Module):
    def __init__(self, filename_obj, filename_ref):
        super(Model, self).__init__()

        # set template mesh
        texture_size = 4
        self.template_mesh = jr.Mesh.from_obj(filename_obj, texture_res=texture_size,load_texture=True, dr_type='softras')
        self.vertices = (self.template_mesh.vertices).stop_grad()
        self.faces = self.template_mesh.faces.stop_grad()
        self.textures = self.template_mesh.textures.stop_grad()
        self.metallic_textures = jt.zeros((1, self.faces.shape[1], texture_size * texture_size, 1)).float32()
        self.roughness_textures = jt.zeros((1, self.faces.shape[1], texture_size * texture_size, 1)).float32() + 0.5
        self.roughness_textures = self.roughness_textures.stop_grad()
        # load reference image
        self.image_ref = jt.array(imread(filename_ref).astype('float32') / 255.).permute(2,0,1).unsqueeze(0).stop_grad()
        # setup renderer
        self.renderer = jr.Renderer(dr_type='softras', light_intensity_directionals=1.0, light_intensity_ambient=0.0)

    def execute(self):
        self.renderer.transform.set_eyes_from_angles(2.732, 30, 140)
        image = self.renderer(self.vertices, self.faces, self.textures, metallic_textures=self.metallic_textures, roughness_textures=self.roughness_textures)
        loss = jt.sum((image - self.image_ref).sqr())
        return loss


def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('./tmp/_tmp_*.png')):
            writer.append_data(imageio.imread(filename))
            os.remove(filename)
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join(data_dir, 'obj/spot/spot_triangulated.obj'))
    parser.add_argument('-ir', '--filename_ref', type=str, default=os.path.join(data_dir, 'ref/ref_metallic.png'))
    parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(data_dir, 'results/output_optim_metallic_textures'))
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.filename_output, exist_ok=True)

    model = Model(args.filename_obj, args.filename_ref)

    optimizer = nn.Adam([model.metallic_textures], lr=0.1, betas=(0.5,0.999))
    loop = tqdm.tqdm(range(20))
    for num in loop:
        loop.set_description('Optimizing')
        loss = model()
        optimizer.step(loss)
        
        model.renderer.transform.set_eyes_from_angles(2.732, 30, 140)
        images = model.renderer(model.vertices, model.faces, model.textures, metallic_textures=model.metallic_textures, roughness_textures=model.roughness_textures)
        image = images.numpy()[0].transpose((1, 2, 0))
        imsave('./tmp/_tmp_%04d.png' % num, image)

    make_gif(os.path.join(args.filename_output, 'result.gif'))


if __name__ == '__main__':
    main()
