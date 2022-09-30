
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
        default=os.path.join(data_dir, 'obj/cornell-box/CornellBox-Original.obj'))
    parser.add_argument('-o', '--output-dir', type=str,  
        default=os.path.join(data_dir, 'results/output_render'))
    args = parser.parse_args()

    render = Render(image_size=2048,camera_mode="look",eye=[0,1,3],camera_direction=[0,0,-1],near=0.1,viewing_angle=30)
    scene = Scene.load_scene_from_obj(args.filename_input)
    light1 = Light(position=[2.4,4,2],direction=[-2.4,-4, -2],intensity=1.2,type="point",shadow=False)
    #light1 = Light(position=[2.4,4,2],direction=[-2.4,-4, -2],intensity=1.2,type="area",area = 0.25)
    light2 = Light(intensity=0.8,type="ambient")
    scene.append_light([light1,light2])
    scene.set_render(render)
    scene.set_reflection(3,"mirror")
    #scene.set_render_target([0,5,6])
    start_time = time.time()
    rgb = scene.deferred_render()[:,::-1,:]
    end_time = time.time()
    print(end_time-start_time)
    os.makedirs(args.output_dir, exist_ok=True)

    # draw object from different view
    writer = imageio.get_writer(os.path.join(args.output_dir, 'SSR.jpg'))
    writer.append_data((255*rgb.numpy()).astype(np.uint8))
    writer.close()

if __name__ == '__main__':
    main()
