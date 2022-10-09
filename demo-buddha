
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
        default=os.path.join(data_dir, 'scene_buddha'))
    parser.add_argument('-o', '--output-dir', type=str,  
        default=os.path.join(data_dir, 'results/output_render'))
    args = parser.parse_args()
    
    render = Render(image_size=2048,camera_mode="look",eye=[0,0.47,-1],camera_direction=[0,0,1],near=0.5,viewing_angle=30)
    #render.set_view(eye = [1,5,-7], camera_direction=[0.5,-0.2,1])
    files_name = [ os.path.join(args.filename_input, filename) for filename in os.listdir(args.filename_input)]
    scene = Scene.load_scene_from_obj(files_name)
    #light1 = Light(position=[-1,1,-2],direction=[-2.4,-4, -2],intensity=1.0,type="point",shadow=False)
    light3 = Light(position=[1,1,-2],direction=[3,-4, 0],intensity=1.2,type="point",shadow=False)
    light2 = Light(intensity=0.5,type="ambient")
    scene.append_light([light2,light3])
    scene.set_render(render)
    scene.set_specular(1,False)
    scene.set_GenerateNormal(0,"from_obj")
    scene.set_roughness(0,0.15)
    scene.set_roughness(1,0.2)
    #scene.set_reflection(1,"mirror")
    #scene.set_reflection(1,"glossy")
    scene.set_render_target([0])

    vertices = scene.objects[0].face_vertices
    y = vertices[:,:,1]
    min_y = jt.min(y)
    y -= min_y
    scene.objects[0]._face_vertices = jt.stack([vertices[:,:,0],y,vertices[:,:,2]],dim = 2)
    

    start_time = time.time()
    rgb = scene.deferred_render()[:,::-1,:]
    end_time = time.time()
    print(end_time-start_time)
    os.makedirs(args.output_dir, exist_ok=True)

    # draw object from different view
    writer = imageio.get_writer(os.path.join(args.output_dir, 'buddha_with_ssao.jpg'))
    writer.append_data((255*rgb.numpy()).astype(np.uint8))
    writer.close()

if __name__ == '__main__':
    main()