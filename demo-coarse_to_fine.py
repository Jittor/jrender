
import jittor as jt
jt.flags.use_cuda = 1
import os
import numpy as np
import imageio
import argparse
from jrender.render2.render2 import Render
from jrender.Scene import *
import time

def set_vertices(vertices,offset_x,offset_y,offset_z):
    x = vertices[:,:,0] + offset_x
    y = vertices[:,:,1] + offset_y
    z = vertices[:,:,2] + offset_z
    return jt.stack([x,y,z],dim = 2)

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename-input', type=str, 
        default=os.path.join(data_dir, 'scene_cow'))
    parser.add_argument('-o', '--output-dir', type=str,  
        default=os.path.join(data_dir, 'results/output_render'))
    args = parser.parse_args()
    #TODO:light area 
    #render = Render(image_size=2048,camera_mode="look",eye=[0.07,0.9,-1.75],camera_direction=[0,-0.3,1],near=0.5)
    render = Render(image_size=2048,camera_mode="look",eye=[0,1,3],camera_direction=[0,0,-1],near=0.5,bin_size=0,max_elems_per_bin=8192)
    files_name = [os.path.join(args.filename_input, filename) for filename in os.listdir(args.filename_input)]
    scene = Scene.load_scene_from_obj(files_name)
    light1 = Light(position=[2,2,3],direction=[0,-1,0],intensity=1.35,type="point",shadow=False,view_angle=50,up=[0,0,1])
    #light1 = Light(position=[0.68,2.5,-1.7],direction=[0,-1,0],intensity=1.5,type="area",area = 0.01,shadow=True,view_angle=50,up=[0,0,1])
    light2 = Light(intensity=0.6,type="ambient")
    scene.append_light([light1,light2])
    scene.set_GenerateNormal([1],"from_obj")
    scene.set_rescaling(1,1.)
    scene.set_kd_res(49)
    scene.set_render(render)
    scene.set_render_target([1])

    vertices = scene.objects[1].face_vertices
    y = vertices[:,:,1]
    min_y = jt.min(y)
    y -= min_y
    scene.objects[1]._face_vertices = jt.stack([vertices[:,:,0],y,vertices[:,:,2]],dim = 2)

    start_time = time.time()
    rgb = scene.deferred_render()[:,::-1,:]
    os.makedirs(args.output_dir, exist_ok=True)

    # draw object from different view
    writer = imageio.get_writer(os.path.join(args.output_dir, 'test1.jpg'))
    writer.append_data((255*rgb.numpy()).astype(np.uint8))
    writer.close()
    end_time = time.time()
    print(end_time-start_time)

if __name__ == '__main__':
    main()
