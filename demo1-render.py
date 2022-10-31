import jittor as jt
import jrender as jr
jt.flags.use_cuda = 1
import os
import tqdm
import numpy as np
import imageio
import argparse

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename-input', type=str, 
        default=os.path.join(data_dir, 'obj/spot/spot_triangulated.obj'))
    parser.add_argument('-o', '--output-dir', type=str, 
        default=os.path.join(data_dir, 'results/output_render'))
    args = parser.parse_args()
    # other settings
    camera_distance = 2.732
    elevation = 30
    azimuth = 0

    # load from Wavefront .obj file
    mesh = jr.Mesh.from_obj(args.filename_input, load_texture=True, texture_res=5, texture_type='surface', dr_type='softras')
    # create renderer with SoftRas
    renderer = jr.Renderer(dr_type='softras')

    os.makedirs(args.output_dir, exist_ok=True)

    # draw object from different view
    loop = tqdm.tqdm(list(range(0, 360, 4)))
    writer = imageio.get_writer(os.path.join(args.output_dir, 'rotation.gif'), mode='I')
    imgs = []
    from PIL import Image
    for num, azimuth in enumerate(loop):
        # rest mesh to initial state
        mesh.reset_()
        loop.set_description('Drawing rotation')
        renderer.transform.set_eyes_from_angles(camera_distance, elevation, azimuth)
        rgb = renderer.render_mesh(mesh, mode='rgb')
        image = rgb.numpy()[0].transpose((1, 2, 0))
        writer.append_data((255*image).astype(np.uint8))
    writer.close()

    # draw object from different sigma and gamma
    loop = tqdm.tqdm(list(np.arange(-4, -2, 0.2)))
    renderer.transform.set_eyes_from_angles(camera_distance, elevation, 45)
    writer = imageio.get_writer(os.path.join(args.output_dir, 'bluring.gif'), mode='I')
    for num, gamma_pow in enumerate(loop):
        # rest mesh to initial state
        mesh.reset_()
        renderer.set_gamma(10**gamma_pow)
        renderer.set_sigma(10**(gamma_pow - 1))
        loop.set_description('Drawing blurring')
        images = renderer.render_mesh(mesh, mode='rgb')
        image = images.numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
        writer.append_data((255*image).astype(np.uint8))
    writer.close()

    # save to textured obj
    mesh.reset_()
    mesh.save_obj(os.path.join(args.output_dir, 'saved_spot.obj'))

if __name__ == '__main__':
    main()
