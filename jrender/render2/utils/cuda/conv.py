import jittor as jt

# x:[h,w,(d)] w:[h,w]
# x:[h,w,(d)]
def conv_for_image(x, w, overflow=0):
    flag_d=0
    if len(x.shape) == 2:
        x = x.unsqueeze(2)
        flag_d=1
    Kh, Kw= w.shape
    H, W, d = x.shape
    w = w.unsqueeze(2)
    xx = x.reindex([H, W, Kh, Kw, d], [
        'i0 + i2 - 1', 'i1 + i3 - 1', 'i4'
    ], overflow)
    ww = w.broadcast_var(xx)
    yy = xx * ww
    y = yy.sum([2, 3])
    if flag_d:
        y = y.squeeze(2)
    return y

"""
def conv_for_image(image, w=jt.ones([3, 3]) / 9):
    if len(image.shape)==3:
        image = image.squeeze(2)
    im = jt.zeros_like(image)
    h = im.shape[0]
    filter_size = w.shape[0] 
    padding = jt.zeros([1,h])
    image = jt.concat([padding,image,padding],dim = 0)
    padding = jt.zeros([h+2,1])
    image = jt.concat([padding,image,padding],dim = 1)
    return jt.code(im.shape, im.dtype, [image, w],
                   cuda_header='''
    #include<cuda.h>
    #include<cuda_runtime.h>
    template <typename scalar_t>
    __global__ void SSR_cuda(scalar_t* im,const scalar_t* image, const scalar_t* w,const int is, const int f_size){
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= is*is){
            return;
        }
        const int ii = i + is + 1;
        
        //im[i] = image[ii-is-1]*w[0]+image[ii-is]*w[1]+image[ii-is+1]*w[2]+image[ii-1]*w[3]+image[ii]*w[4]+image[ii+1]*w[5]+image[ii+is-1]*w[6]+image[ii+is]*w[7]+image[ii+is+1]*w[8];
        im[i] = 0;
    }
    ''',
                   cuda_src=f'''
    @alias(im,out0)
    @alias(image,in0)
    @alias(w,in1)
    const int is = im_shape0;
    
    const int threads = 1024;
    const dim3 blocks = (is * is - 1) / threads + 1;
    SSR_cuda<float32><<<blocks,threads>>>(
        im_p,
        image_p,
        w_p,
        is,
        {filter_size}
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error in load_textures: %s\\n", cudaGetErrorString(err));

    ''').unsqueeze(2)

"""