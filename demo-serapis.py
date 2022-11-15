
import jittor as jt
jt.flags.use_cuda = 1
import os
import numpy as np
import imageio
import argparse
from jrender.render2.render2 import Render
from jrender.Scene import *
import tqdm


current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename-input', type=str, 
        default=os.path.join(data_dir, 'scene/scene_serapis'))
    parser.add_argument('-o', '--output-dir', type=str,  
        default=os.path.join(data_dir, 'results/output_render'))
    args = parser.parse_args()
    
    render = Render(image_size=2048,camera_mode="look_at",eye=[-0.4,1,4],camera_direction=[-0.1,0,-1],near=0.5,viewing_angle=30,bin_size=76)
    files_name = [ os.path.join(args.filename_input, filename) for filename in os.listdir(args.filename_input)]
    scene = Scene.load_scene_from_obj(files_name)
    light1 = Light(position=[0,0.5,6],direction=[-2.4,-4, -2],intensity=0.8,type="point",shadow=False)
    light2 = Light(position=[0,0.5,-6],direction=[3,-4, 0],intensity=0.8,type="point",shadow=False)
    light3 = Light(position=[0,0,-4],direction=[3,-4, 0],intensity=0.8,type="point",shadow=False)
    light4 = Light(intensity=0.8,type="ambient")
    #scene.append_light([light1,light2,light3,light4])
    scene.append_light([light1,light2,light4])
    scene.set_render(render)
    scene.set_roughness(1,0.1)
    scene.set_roughness(0,0.1)
    scene.set_GenerateNormal(1,"from_obj")
    #rescaling to [-1,1]^3
    scene.set_rescaling(1,scale=1.)
    scene.set_render_target([1])

    camera_distance = 2.732
    elevation = 15
    # draw object from different view
    loop = tqdm.tqdm(list(range(0, 360, 4)))
    writer = imageio.get_writer(os.path.join(args.output_dir, 'serapis_rotation.gif'), mode='I')
    for num, azimuth in enumerate(loop):
        # rest mesh to initial state
        loop.set_description('Drawing rotation')
        render.set_eyes_from_angles(camera_distance, elevation, azimuth)
        rgb = scene.deferred_render()[:,::-1,:]
        writer.append_data((255*rgb.numpy()).astype(np.uint8))
    writer.close()


if __name__ == '__main__':
    main()