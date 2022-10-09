import jittor as jt 
from skimage.io import imread,imsave
from jrender.render2.utils.cuda.FXAA_cuda import FXAA_cuda
jt.flags.use_cuda = 1

def main():
    image = imread("D:\Render\jrender\data\\results\output_render\\bunny_with_sssr.jpg")
    image = jt.array(image).float32()
    image_AA = FXAA_cuda(image)
    imsave("D:\Render\jrender\data\\results\\temp\\FXAA.jpg", image_AA)

if __name__ == '__main__':
    main()