
import jittor as jt
jt.flags.use_cuda = 1
import os
import numpy as np
import imageio
import argparse
from jrender.render2.render2 import Render
from jrender.Scene import *
import time


current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename-input', type=str, 
        default=os.path.join(data_dir, 'scene_cone'))
    parser.add_argument('-o', '--output-dir', type=str,  
        default=os.path.join(data_dir, 'results/output_render'))
    args = parser.parse_args()
    #TODO:light area 
    render = Render(image_size=2048,camera_mode="look",eye=[-3.5,2.5,2.5],camera_direction=[1.2,-1.2,-1],near=0.5,bin_size=128,max_elems_per_bin=8192)
    files_name = [os.path.join(args.filename_input, filename) for filename in os.listdir(args.filename_input)]
    scene = Scene.load_scene_from_obj(files_name)
    light1 = Light(position=[0,3.5,-2],direction=[0,-1,0],intensity=1.5,type="point",shadow=True,up=[0,0,1],view_angle=45)
    light3 = Light(position=[0,3.5,-2],direction=[0,-1,0],intensity=1.5,type="area",area = 0.01,shadow=True,up=[0,0,1],view_angle=55)
    light2 = Light(intensity=0.4,type="ambient")
    scene.set_specular([0,1,2,3,4,5],False)
    scene.append_light([light1,light2])
    scene.set_rescaling([0,1,2,3,4],0.5)
    scene.set_GenerateNormal([0,1,2,3,4],"from_obj")
    scene.set_roughness([0,1,2,3,4],0.4)
    scene.set_render(render)

    for i in range(5):
        vertices = scene.objects[i].face_vertices
        x = vertices[:,:,0]
        z = vertices[:,:,2]
        y = vertices[:,:,1]
        min_y = jt.min(y)
        y -= min_y
        scene.objects[i]._face_vertices = jt.stack([x - (i - 2) * 1,y,z],dim = 2)
        

    start_time = time.time()
    rgb = scene.deferred_render()[:,::-1,:]
    end_time = time.time()
    print(end_time-start_time)
    os.makedirs(args.output_dir, exist_ok=True)

    # draw object from different view
    writer = imageio.get_writer(os.path.join(args.output_dir, 'cone_hard_shadow.jpg'))
    writer.append_data((255*rgb.numpy()).astype(np.uint8))
    writer.close()

if __name__ == '__main__':
    main()
