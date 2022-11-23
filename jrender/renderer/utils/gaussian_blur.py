import jittor as jt
import math

def gaussian_blur(image,tap_num,v,modulate_map,dim=0):
    
    image_blur=jt.zeros_like(image)
    Gauss=jt.zeros((tap_num),image.dtype)
    GaussWidth=math.sqrt(v)
    for i in range(tap_num):
        Gauss[i]=GaussWidth*(Gaussian_1d(v,-GaussWidth*((tap_num-1)/2-i)))

    tap_num=jt.int32(tap_num)
    GaussWidth=jt.float32(GaussWidth)

    dim=jt.int32(dim)                      
    if len(image.shape)==2:
        image=image.unsqueeze(2)
    image_blur=_blur_for_SSS(image_blur,image,modulate_map,Gauss,tap_num,GaussWidth,dim)
    if len(image_blur.shape)==2:
        image=image.squeeze(2)
    return image_blur
    

#v=sigma^2
def Gaussian_1d(v,r):
    return math.exp(-r**2/(2*v))/math.sqrt(2*math.pi*v)


def _blur_for_SSS(image_blur,image,modulate_map,Gauss,tap_num,GaussWidth,dim):
    return jt.code(image_blur.shape, image_blur.dtype, [image, modulate_map , Gauss],
    cuda_header='''

template <typename scalar_t>
__global__ void gaussain_blur_x_Kernel(const scalar_t* __restrict__ ptr1,
    const scalar_t* __restrict__ ptr2,
    const scalar_t* __restrict__ tap,
    scalar_t* __restrict__ ptr_result,
    int tap_num,
    float GaussWidth,
    size_t h,
    size_t w,
    size_t d)
{   
    
    int xi = threadIdx.x + blockDim.x * blockIdx.x;
    int yi = threadIdx.y + blockDim.y * blockIdx.y;
    if (xi > w - 1) {
        return;
    }
    if (yi > h - 1) {
        return;
    }
    int id = yi * w + xi;
    float netFilterWidth = GaussWidth * ptr2[id];
    float x = xi - netFilterWidth * (tap_num - 1) / 2;
    int k, l;
    scalar_t v = 0;
    
    
    int flag = 0;
    for (k = 0; k < d; k++){
        scalar_t test = ptr1[yi * w * d + xi * d + l];
        if (test > 0) flag = 1;
    }
    if (!flag) {
        ptr_result[id] = 0;
        return;
    }
    
    for (k = 0; k < tap_num; k++) {
        int x1 = int(x);
        int x2 = int(x) + 1;
        if (x1 < 0) {
            x1 = 0;
        }
        else if (x1 > w - 1) {
            x1 = w - 1;
        }
        if (x2 < 0) {
            x2 = 0;
        }
        else if (x2 > w - 1) {
            x2 = w - 1;
        }
        for (l = 0; l < d; l++) {
            scalar_t v1 = ptr1[yi * w * d + x1 * d + l];
            v1 = v1 > 1e-6 ? v1 : ptr1[yi * w * d + xi * d + l];
            scalar_t v2 = ptr1[yi * w * d + x2 * d + l];
            v2 = v2 > 1e-6 ? v2 : ptr1[yi * w * d + xi * d + l];
            scalar_t interpolation = (x - x1) * v2 + (x2 - x) * v1;
            v += interpolation * tap[k];
        }
        x += netFilterWidth;
    }
    ptr_result[id] = v;
    
}

template <typename scalar_t>
__global__ void gaussain_blur_y_Kernel(const scalar_t* __restrict__ ptr1,
    const scalar_t* __restrict__ ptr2,
    const scalar_t* __restrict__ tap,
    scalar_t* __restrict__ ptr_result,
    int tap_num,
    float GaussWidth,
    size_t h,
    size_t w,
    size_t d)
{   
    
    int xi = threadIdx.x + blockDim.x * blockIdx.x;
    int yi = threadIdx.y + blockDim.y * blockIdx.y;
    if (xi > w - 1) {
        return;
    }
    if (yi > h - 1) {
        return;
    }
    int id = yi * w + xi;
    float netFilterWidth = GaussWidth * ptr2[id];
    float y = yi - netFilterWidth * (tap_num - 1) / 2;
    int k, l;
    scalar_t v = 0;
    
    
    int flag = 0;
    for (k = 0; k < d; k++){
        scalar_t test = ptr1[yi * w * d + xi * d + l];
        if (test > 0) flag = 1;
    }
    if (!flag) {
        ptr_result[id] = 0;
        return;
    }
    

    for (k = 0; k < tap_num; k++) {
        int y1 = int(y);
        int y2 = int(y) + 1;
        if (y1 < 0) {
            y1 = 0;
        }
        else if (y1 > w - 1) {
            y1 = w - 1;
        }
        if (y2 < 0) {
            y2 = 0;
        }
        else if (y2 > w - 1) {
            y2 = w - 1;
        }
        for (l = 0; l < d; l++) {
            scalar_t v1 = ptr1[y1 * w * d + xi * d + l];
            v1 = v1 > 1e-6 ? v1 : ptr1[yi * w * d + xi * d + l];
            scalar_t v2 = ptr1[y2 * w * d + xi * d + l];
            v2 = v2 > 1e-6 ? v2 : ptr1[yi * w * d + xi * d + l];
            scalar_t interpolation = v1 * (y2 - y) + v2 * (y - y1);
            v += interpolation * tap[k];
        }
        y += netFilterWidth;
    }
    ptr_result[id] = v;
    
}

    ''',
    cuda_src=f'''
    @alias(image, in0)
    @alias(modulate_map, in1)
    @alias(Gauss, in2)
    @alias(image_blur, out0)
    const auto image_height = image_shape0;
    const auto image_width = image_shape1;
    const auto image_depth = image_shape2;
    const dim3 block(32,32);
    const dim3 grid((image_width - 1) / block.x + 1,(image_height - 1) / block.y + 1);
    
    if ({dim}==0)
        gaussain_blur_y_Kernel<float32><<<grid, block>>>(
            image_p,
            modulate_map_p,
            Gauss_p,
            image_blur_p,
            {tap_num},
            {GaussWidth},
            image_height,
            image_width,
            image_depth);
    else if ({dim}==1)
        gaussain_blur_x_Kernel<float32><<<grid, block>>>(
            image_p,
            modulate_map_p,
            Gauss_p,
            image_blur_p,
            {tap_num},
            {GaussWidth},
            image_height,
            image_width,
            image_depth);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in SSS: %s\\n", cudaGetErrorString(err));
    '''
    )
