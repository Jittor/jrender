import jittor as jt
jt.flags.use_cuda = 1
import os
import numpy as np
import imageio
import argparse
from jrender.renderer.render2 import Render
from jrender.renderer.scene import Scene,Light


current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename-input', type=str, 
        default=os.path.join(data_dir, 'obj/cornell-box/CornellBox-Original.obj'))
    parser.add_argument('-o', '--output-dir', type=str,  
        default=os.path.join(data_dir, 'results/output_render'))
    args = parser.parse_args()

    render = Render(image_size=1024,camera_mode="look",eye=[0,1.8,3],camera_direction=[0,-0.6,-1],near=0.1,viewing_angle=30)
    scene = Scene.load_scene_from_obj(args.filename_input)
    light1 = Light(position=[-2,4,2],direction=[2,-4, -2],intensity=0.8,type="point")
    light2 = Light(intensity=0.8,type="ambient")
    scene.append_light([light1,light2])
    scene.set_render(render)
    scene.set_render_target([0,5,6])
    rgb = scene.deferred_render()[:,::-1,:]
    os.makedirs(args.output_dir, exist_ok=True)

    # draw object from different view
    writer = imageio.get_writer(os.path.join(args.output_dir, 'SSR.jpg'))
    writer.append_data((255*rgb.numpy()).astype(np.uint8))
    writer.close()

if __name__ == '__main__':
    main()
