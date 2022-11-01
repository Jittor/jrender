import jittor as jt
from jittor import nn
jt.flags.use_cuda = 1

import os
import tqdm
import numpy as np
import imageio
import argparse

import jrender as jr
from jrender import neg_iou_loss, LaplacianLoss, FlattenLoss

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

class Model(nn.Module):
    def __init__(self, template_path):
        super(Model, self).__init__()

        # set template mesh
        self.template_mesh = jr.Mesh.from_obj(template_path, dr_type='n3mr')
        self.vertices = (self.template_mesh.vertices * 0.5).stop_grad()
        self.faces = self.template_mesh.faces.stop_grad()
        self.textures = self.template_mesh.textures.stop_grad()

        # optimize for displacement map and center
        self.displace = jt.zeros(self.template_mesh.vertices.shape)
        self.center = jt.zeros((1, 1, 3))

        # define Laplacian and flatten geometry constraints
        self.laplacian_loss = LaplacianLoss(self.vertices[0], self.faces[0])
        self.flatten_loss = FlattenLoss(self.faces[0])

    def execute(self, batch_size):
        base = jt.log(self.vertices.abs() / (1 - self.vertices.abs()))
        centroid = jt.tanh(self.center)
        vertices = (base + self.displace).sigmoid() * nn.sign(self.vertices)
        vertices = nn.relu(vertices) * (1 - centroid) - nn.relu(-vertices) * (centroid + 1)
        vertices = vertices + centroid

        # apply Laplacian and flatten geometry constraints
        laplacian_loss = self.laplacian_loss(vertices).mean()
        flatten_loss = self.flatten_loss(vertices).mean()
        return jr.Mesh(vertices.repeat(batch_size, 1, 1), 
                       self.faces.repeat(batch_size, 1, 1), dr_type='n3mr'), laplacian_loss, flatten_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename-input', type=str, 
        default=os.path.join(data_dir, 'source.npy'))
    parser.add_argument('-c', '--camera-input', type=str, 
        default=os.path.join(data_dir, 'camera.npy'))
    parser.add_argument('-t', '--template-mesh', type=str, 
        default=os.path.join(data_dir, 'obj/sphere/sphere_1352.obj'))
    parser.add_argument('-o', '--output-dir', type=str, 
        default=os.path.join(data_dir, 'results/output_deform'))
    parser.add_argument('-b', '--batch-size', type=int,
        default=120)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model = Model(args.template_mesh)

    renderer = jr.Renderer(image_size=64, sigma_val=1e-4, aggr_func_rgb='hard', camera_mode='look_at', viewing_angle=15, dr_type='softras', bin_size=16, max_elems_per_bin=2700, max_faces_per_pixel_for_grad=16)

    # read training images and camera poses
    images = np.load(args.filename_input).astype('float32') / 255.
    cameras = np.load(args.camera_input).astype('float32')
    optimizer = nn.Adam(model.parameters(), 0.01, betas=(0.5, 0.99))
    
    camera_distances = jt.array(cameras[:, 0])
    elevations = jt.array(cameras[:, 1])
    viewpoints = jt.array(cameras[:, 2])
    renderer.transform.set_eyes_from_angles(camera_distances, elevations, viewpoints)

    import time
    sta = time.time()
    loop = tqdm.tqdm(list(range(0, 1000)))
    writer = imageio.get_writer(os.path.join(args.output_dir, 'deform.gif'), mode='I')
    for i in loop:
        images_gt = jt.array(images)

        mesh, laplacian_loss, flatten_loss = model(args.batch_size)
        images_pred = renderer.render_mesh(mesh, mode='silhouettes')

        # optimize mesh with silhouette reprojection error and 
        # geometry constraints
        loss = neg_iou_loss(images_pred, images_gt[:, 3]) + \
               0.03 * laplacian_loss + \
               0.0003 * flatten_loss
            
        loop.set_description('Loss: %.4f'%(loss.item()))
        optimizer.step(loss)
        
        if i % 100 == 0:
            image = images_pred.numpy()[0]#.transpose((1, 2, 0))
            imageio.imsave(os.path.join(args.output_dir, 'deform_%05d.png'%i), (255*image).astype(np.uint8))
            writer.append_data((255*image).astype(np.uint8))

    # save optimized mesh
    model(1)[0].save_obj(os.path.join(args.output_dir, 'plane.obj'), save_texture=False)
    print(f"Cost {time.time() - sta} secs.")


if __name__ == '__main__':
    main()