
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
        default=os.path.join(data_dir, 'scene_desk'))
    parser.add_argument('-o', '--output-dir', type=str,  
        default=os.path.join(data_dir, 'results/output_render'))
    args = parser.parse_args()
    #TODO:light area 
    #render = Render(image_size=2048,camera_mode="look",eye=[0.07,0.9,-1.75],camera_direction=[0,-0.3,1],near=0.5)
    render = Render(image_size=2048,camera_mode="look",eye=[-1.75 * 0.9,1.2 * 0.9,2 * 0.9],camera_direction=[0.98,-0.7,-1.2],near=0.5)
    files_name = [os.path.join(args.filename_input, filename) for filename in os.listdir(args.filename_input)]
    scene = Scene.load_scene_from_obj(files_name)
    light1 = Light(position=[0.68,2.5,-1.7],direction=[0,-1,0],intensity=1.35,type="point",shadow=False,view_angle=50,up=[0,0,1])
    #light1 = Light(position=[0.68,2.5,-1.7],direction=[0,-1,0],intensity=1.5,type="area",area = 0.01,shadow=True,view_angle=50,up=[0,0,1])
    light2 = Light(intensity=0.6,type="ambient")
    scene.set_specular(1,False)
    scene.append_light([light1,light2])
    scene.set_rescaling(0,1.)
    scene.set_rescaling(3,0.2)
    scene.set_GenerateNormal([0,3],"from_obj")
    scene.set_roughness([0,1,3],0.3)
    scene.set_render(render)
    scene.set_render_target([0,1,3])

    vertices = scene.objects[0].face_vertices
    y = vertices[:,:,1]
    min_y = jt.min(y)
    y -= min_y
    scene.objects[0]._face_vertices = jt.stack([vertices[:,:,0],y,vertices[:,:,2]],dim = 2)
    scene.objects[2]._face_vertices = set_vertices(scene.objects[2]._face_vertices,0,-0.3,1)
    scene.objects[3]._face_vertices = set_vertices(scene.objects[3]._face_vertices,0.12,0.64,0)

    start_time = time.time()
    rgb = scene.deferred_render()[:,::-1,:]
    end_time = time.time()
    print(end_time-start_time)
    os.makedirs(args.output_dir, exist_ok=True)

    # draw object from different view
    writer = imageio.get_writer(os.path.join(args.output_dir, 'desk_hard_shadow.jpg'))
    writer.append_data((255*rgb.numpy()).astype(np.uint8))
    writer.close()

if __name__ == '__main__':
    main()
