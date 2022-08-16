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
        default=os.path.join(data_dir, 'obj/cornell-box/CornellBox-Original.obj'))
    parser.add_argument('-o', '--output-dir', type=str,  
        default=os.path.join(data_dir, 'results/output_render'))
    args = parser.parse_args()

    # load from Wavefront .obj file
    mesh = jr.Mesh.from_obj(args.filename_input, load_texture=True, texture_res=5, texture_type='surface', dr_type='softras') 
    # create renderer with SoftRas
    renderer = jr.Renderer(dr_type='softras',image_size=1024,camera_mode="look",eye=[0,1,3],camera_direction=[0,0,-1],light_directions=[0,1,1])
    
    os.makedirs(args.output_dir, exist_ok=True)

    # draw object from different view
    writer = imageio.get_writer(os.path.join(args.output_dir, 'SSR.jpg'))
    rgb = renderer.render_mesh(mesh, mode='rgb')
    image = rgb.numpy()[0].transpose((1, 2, 0))
    writer.append_data((255*image).astype(np.uint8))
    writer.close()

    # save to textured obj
    mesh.reset_()
    mesh.save_obj(os.path.join(args.output_dir, 'saved_spot.obj'))

if __name__ == '__main__':
    main()
