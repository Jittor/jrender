import jittor as jt
import numpy as np

def sample2D(texture, pos, default=999999):  
    if isinstance(texture,np.ndarray):
        texture = jt.array(texture,"float32")
    else:
        texture = texture.float32()
    pos = pos.float32()
    if len(texture.shape) == 2:
        dimension = 1
        value = jt.zeros((pos.shape[0],pos.shape[1]),"float32")  
    else:
        dimension = texture.shape[2]
        value = jt.zeros((pos.shape[0],pos.shape[1],texture.shape[2]),"float32")      
    return jt.code(value.shape, value.dtype, [texture, pos],
                   cuda_header='''
    #include <cuda.h>
    #include <cuda_runtime.h>
    
    namespace {
    template <typename scalar_t>
    __global__ void sample2D_cuda_kernel(
        const scalar_t*  image,
        const scalar_t*  pos,
        scalar_t*  value, 
        size_t value_size,
        size_t image_height,
        size_t image_width,
        size_t dimension,
        scalar_t Default) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i * dimension >= value_size) {
        return;
    }

    const scalar_t* posn = &pos[i * 2];
    scalar_t* valuen = &value[i * dimension];

    const scalar_t pos_x =  (posn[0] * (image_width - 1));
    const scalar_t pos_y =  (posn[1] * (image_height - 1));

    if (pos_x < 0 || pos_x > (image_width - 1) || pos_y < 0 || pos_y > (image_height - 1)) {
        for (int k = 0; k < dimension; k++) {
            valuen[k] = Default;
        }
        return;
    }

    if (1) {
        /* bilinear sampling */
        const scalar_t weight_x1 = pos_x - (int)pos_x;
        const scalar_t weight_x0 = 1 - weight_x1;
        const scalar_t weight_y1 = pos_y - (int)pos_y;
        const scalar_t weight_y0 = 1 - weight_y1;
        for (int k = 0; k < dimension; k++) {
            scalar_t c = 0;
            c += image[((int)pos_y * image_width + (int)pos_x) * 3 + k] * (weight_x0 * weight_y0);
            c += image[((int)(pos_y + 1) * image_width + (int)pos_x) * 3 + k] * (weight_x0 * weight_y1);
            c += image[((int)pos_y * image_width + ((int)pos_x) + 1) * 3 + k] * (weight_x1 * weight_y0);
            c += image[((int)(pos_y + 1)* image_width + ((int)pos_x) + 1) * 3 + k] * (weight_x1 * weight_y1);
            valuen[k] = c;
        }
    } else {
        /* nearest neighbor */
        const int pos_xi = round(pos_x);
        const int pos_yi = round(pos_y);
        for (int k = 0; k < dimension; k++) {
            valuen[k] = image[(pos_yi * image_width + pos_xi) * dimension + k];
        }
    }
    }
    }
    ''',
                   cuda_src=f'''
    @alias(texture, in0)
    @alias(pos, in1)
    @alias(value, out0)
    const auto value_size = value->num;
    const auto image_height = texture_shape0;
    const auto image_width = texture_shape1;
    
    const int threads = 1024;
    const dim3 blocks ((value_size / {dimension} - 1) / threads + 1);

    sample2D_cuda_kernel<float32><<<blocks, threads>>>(
        texture_p,
        pos_p,
        value_p,
        value_size,
        image_height,
        image_width,
        {dimension},
        {default});

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in load_textures: %s\\n", cudaGetErrorString(err));
    '''
                   )




def _sample2D(tex, pos, default=999999):  

    pos = pos.float32()
    if len(tex.shape) == 2:
        dimension = 1
        value = jt.zeros((pos.shape[0],pos.shape[1]),"float32")
        tex = tex.unsqueeze(2)  
    else:
        dimension = tex.shape[2]
        value = jt.zeros((pos.shape[0],pos.shape[1],tex.shape[2]),"float32")
    texture = jt.concat([tex,jt.zeros((tex.shape[0],tex.shape[1],4-tex.shape[2]))],dim=2)
    return jt.code(value.shape, value.dtype, [texture, pos],
                   cuda_header='''
    #include <cuda.h>
    #include <cuda_runtime.h>

    namespace {
    template <typename scalar_t>
    __global__ void sample2D_cuda_kernel(
        cudaTextureObject_t texture,
        float* pos,
        float*  value,
        size_t value_size, 
        size_t dimension,
        float Default) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i * dimension >= value_size) {
        return;
    }

    float* posn = &pos[i * 2];
    float* valuen = &value[i * dimension];

    
    if (posn[0] < 0 || posn[0] > 1 || posn[1] < 0 || posn[1] > 1) {
        for (int k = 0; k < dimension; k++) {
            valuen[k] = Default;
        }
        return;
    }
    float4 v_rgb = tex2D<float4>(texture,posn[1],posn[0]);
    if(dimension >= 1)
        valuen[0] = v_rgb.x;
    if(dimension >= 2)
        valuen[1] = v_rgb.y;
    if(dimension >= 3)
        valuen[2] = v_rgb.z;
    if(dimension == 4)
        valuen[3] = v_rgb.w;
    }
    }
    ''',
                   cuda_src=f'''
    @alias(texture, in0)
    @alias(pos, in1)
    @alias(value, out0)
    const auto value_size = value->num;
    const auto image_height = texture_shape0;
    const auto image_width = texture_shape1;
    
    const int threads = 1024;
    const dim3 blocks ((value_size / {dimension} - 1) / threads + 1);

    //Create cudaArray
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,32,32,32,cudaChannelFormatKindFloat);
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, image_width, image_height);
    cudaMemcpy2DToArray(cuArray, 0, 0, texture_p, 16 * image_width, 16 * image_width, image_height, cudaMemcpyHostToDevice);
    //Create resDescription
    cudaResourceDesc resDesc;
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    //Create texDescription
    cudaTextureDesc texDesc;
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;
    //Create textureObj
    cudaTextureObject_t texObj = 1;
    //cudaResourceViewDesc resvDesc;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    
    sample2D_cuda_kernel<float32><<<blocks, threads>>>(
        texObj,
        pos_p,
        value_p,
        value_size,
        {dimension},
        {default}
    );
    /*
    */

    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in sample2D: %s\\n", cudaGetErrorString(err));
    '''
                   ,)