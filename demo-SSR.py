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

    # load from Wavefront .obj file
    # create renderer with SoftRas
    render = Render(image_size=1024,camera_mode="look",eye=[0,1,3],camera_direction=[0,0,-1])
    scene = Scene.load_scene_from_obj(args.filename_input)
    light1 = Light(position=[0,1,3],direction=[0,0,-1],type="point")
    scene.append_light(light1)
    scene.set_render(render)
    rgb = scene.deferred_render()
    os.makedirs(args.output_dir, exist_ok=True)

    # draw object from different view
    writer = imageio.get_writer(os.path.join(args.output_dir, 'SSR.jpg'))
    writer.append_data((255*rgb.numpy()).astype(np.uint8))
    writer.close()

if __name__ == '__main__':
    main()
