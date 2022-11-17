import jittor as jt

def _load_textures_for_softras(image, faces, textures, is_update):
    return jt.code(textures.shape, textures.dtype, [image, faces, is_update],
    cuda_header='''
    #include <cuda.h>
    #include <cuda_runtime.h>

    namespace {
    template <typename scalar_t>
    __global__ void load_textures_cuda_kernel(
        const scalar_t* __restrict__ image,
        const scalar_t* __restrict__ faces,
        const int32_t* __restrict__ is_update,
        scalar_t* __restrict__ textures, 
        size_t texture_size,
        size_t texture_res,
        size_t image_height,
        size_t image_width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i * 3 >= texture_size) {
        return;
    }
    const int R = texture_res;
    const int fn = i / (R * R);
    const int w_y = (i % (R * R)) / R;
    const int w_x = i % R;
    // compute barycentric coordinate
    scalar_t w0, w1, w2;
    if (w_x + w_y < R) {
        w0 = (w_x + 1. / 3.) / R;
        w1 = (w_y + 1. / 3.) / R;
        w2 = 1. - w0 - w1;
    } else {
        w0 = ((R - 1. - w_x) + 2. / 3.) / R;
        w1 = ((R - 1. - w_y) + 2. / 3.) / R;
        w2 = 1. - w0 - w1;
    }
    const scalar_t* face = &faces[fn * 3 * 2];
    scalar_t* texture = &textures[i * 3];
    if (is_update[fn] == 0) return;
    
    const scalar_t pos_x = (
        (face[2 * 0 + 0] * w0 + face[2 * 1 + 0] * w1 + face[2 * 2 + 0] * w2) * (image_width - 1));
    const scalar_t pos_y = (
        (face[2 * 0 + 1] * w0 + face[2 * 1 + 1] * w1 + face[2 * 2 + 1] * w2) * (image_height - 1));
    if (1) {
        /* bilinear sampling */
        const scalar_t weight_x1 = pos_x - (int)pos_x;
        const scalar_t weight_x0 = 1 - weight_x1;
        const scalar_t weight_y1 = pos_y - (int)pos_y;
        const scalar_t weight_y0 = 1 - weight_y1;
        for (int k = 0; k < 3; k++) {
            scalar_t c = 0;
            c += image[((int)pos_y * image_width + (int)pos_x) * 3 + k] * (weight_x0 * weight_y0);
            c += image[((int)(pos_y + 1) * image_width + (int)pos_x) * 3 + k] * (weight_x0 * weight_y1);
            c += image[((int)pos_y * image_width + ((int)pos_x) + 1) * 3 + k] * (weight_x1 * weight_y0);
            c += image[((int)(pos_y + 1)* image_width + ((int)pos_x) + 1) * 3 + k] * (weight_x1 * weight_y1);
            texture[k] = c;
        }
    } else {
        /* nearest neighbor */
        const int pos_xi = round(pos_x);
        const int pos_yi = round(pos_y);
        for (int k = 0; k < 3; k++) {
            texture[k] = image[(pos_yi * image_width + pos_xi) * 3 + k];
        }
    }
    }
    }
    ''',
    cuda_src=f'''
    @alias(image, in0)
    @alias(faces, in1)
    @alias(is_update, in2)
    @alias(textures, out0)
    // texture_size = size of the textures tensor
    const auto texture_size = textures->num;
    // notice that texture_res != texture_res
    const auto texture_res = sqrt(textures_shape1);
    const auto image_height = image_shape0;
    const auto image_width = image_shape1;
    
    const int threads = 1024;
    const dim3 blocks ((texture_size / 3 - 1) / threads + 1);

    load_textures_cuda_kernel<float32><<<blocks, threads>>>(
        image_p,
        faces_p,
        is_update_p,
        textures_p,
        texture_size,
        texture_res,
        image_height,
        image_width);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in load_textures: %s\\n", cudaGetErrorString(err));
    '''
    )

def _load_textures_for_n3mr(image, faces, textures, is_update, texture_wrapping, use_bilinear):
    return jt.code(textures.shape, textures.dtype, [image, faces, is_update],
    cuda_header='''
    #include <cuda.h>
    #include <cuda_runtime.h>

    template <typename scalar_t>
    static __inline__ __device__ scalar_t mod(scalar_t x, scalar_t y) {
        if (x > 0) {
            return fmod(x,y);
        }
        else {
            return y + fmod(x,y);
        }
    }

    namespace {

    const int REPEAT = 0;
    const int MIRRORED_REPEAT = 1;
    const int CLAMP_TO_EDGE = 2;
    const int CLAMP_TO_BORDER = 3;

    template <typename scalar_t>
    __global__ void load_textures_cuda_kernel(
        const scalar_t* image,
        const int32_t* is_update,
        scalar_t* faces,
        scalar_t* __restrict__ textures, 
        int textures_size,
        int texture_size,
        int image_height,
        int image_width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= textures_size / 3) {
        return;
    }
    const int ts = texture_size;
    const int fn = i / (ts * ts * ts);
    scalar_t dim0 = ((i / (ts * ts)) % ts) / (ts - 1.) ;
    scalar_t dim1 = ((i / ts) % ts) / (ts - 1.);
    scalar_t dim2 = (i % ts) / (ts - 1.);
    if (0 < dim0 + dim1 + dim2) {
        float sum = dim0 + dim1 + dim2;
        dim0 /= sum;
        dim1 /= sum;
        dim2 /= sum;
    }
    scalar_t* face = &faces[fn * 3 * 2];
    scalar_t* texture_ = &textures[i * 3];

    if (is_update[fn] != 0) {
        if (texture_wrapping == REPEAT) {
            #pragma unroll
            for (int i = 0; i < 6; ++i) {
                face[i] = mod(face[i], (scalar_t)1.);
            }
        }
        else if (texture_wrapping == MIRRORED_REPEAT) {
            #pragma unroll
            for (int i = 0; i < 6; ++i) {
                if (mod(face[i], (scalar_t)2) < 1) {
                    face[i] = mod(face[i], (scalar_t)1.);
                }
                else {
                    face[i] = 1 - mod(face[i], (scalar_t)1.);
                }
            }
        }
        else if (texture_wrapping == CLAMP_TO_EDGE) {
            #pragma unroll
            for (int i = 0; i < 6; ++i) {
                face[i] = max(min(face[i], (scalar_t) 1), (scalar_t) 0);
            }
        }
        const scalar_t pos_x = (
            (face[2 * 0 + 0] * dim0 + face[2 * 1 + 0] * dim1 + face[2 * 2 + 0] * dim2) * (image_width - 1));
        const scalar_t pos_y = (
            (face[2 * 0 + 1] * dim0 + face[2 * 1 + 1] * dim1 + face[2 * 2 + 1] * dim2) * (image_height - 1));
        if (use_bilinear) {
            /* bilinear sampling */
            const scalar_t weight_x1 = pos_x - (int)pos_x;
            const scalar_t weight_x0 = 1 - weight_x1;
            const scalar_t weight_y1 = pos_y - (int)pos_y;
            const scalar_t weight_y0 = 1 - weight_y1;
            for (int k = 0; k < 3; k++) {
                if (texture_wrapping != CLAMP_TO_BORDER) {
                    scalar_t c = 0;
                    c += image[(int)pos_y * image_width * 3 + (int)pos_x * 3 + k] * (weight_x0 * weight_y0);
                    c += image[min((int)(pos_y + 1), image_height-1) * image_width * 3 + (int)pos_x * 3 + k] * (weight_x0 * weight_y1);
                    c += image[(int)pos_y * image_width * 3 + min((int)pos_x + 1, image_width-1) * 3 + k] * (weight_x1 * weight_y0);
                    c += image[min((int)(pos_y + 1), image_height-1) * image_width * 3 + min((int)pos_x + 1, image_width-1) * 3 + k] * (weight_x1 * weight_y1);
                    texture_[k] = c;
                }
                else {
                    texture_[k] = 0;
                }
            }
        } else {
            /* nearest neighbor */
            const int pos_xi = round(pos_x);
            const int pos_yi = round(pos_y);
            for (int k = 0; k < 3; k++) {
                if (texture_wrapping != CLAMP_TO_BORDER) {
                    texture_[k] = image[pos_yi * image_width * 3 + pos_xi * 3 + k];
                }
                else {
                    texture_[k] = 0;
                }
            }
        }
    }
    }
    }
    '''.replace("texture_wrapping", str(texture_wrapping))
    .replace("use_bilinear", str(use_bilinear)),
    cuda_src='''
    @alias(image, in0)
    @alias(faces, in1)
    @alias(is_update, in2)
    @alias(textures, out0)
    // textures_size = size of the textures tensor
    const auto textures_size = textures->num;
    // notice that texture_size != texture_size
    const auto texture_size = textures_shape1;
    const auto image_height = image_shape0;
    const auto image_width = image_shape1;
    
    const int threads = 1024;
    const dim3 blocks ((textures_size / 3 - 1) / threads + 1);

    load_textures_cuda_kernel<float32><<<blocks, threads>>>(
        image_p,
        is_update_p,
        faces_p,
        textures_p,
        textures_size,
        texture_size,
        image_height,
        image_width);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error in load_textures: %s\\n", cudaGetErrorString(err));
    ''')