import jittor as jt
from math import ceil


def forward_face_index_map(
    faces,
    face_index_map,
    weight_map,
    depth_map,
    face_inv_map,
    faces_inv,
    image_size,
    near,
    far,
    return_rgb,
    return_alpha,
    return_depth):
    lock = jt.empty(depth_map.shape, 'int')
    return jt.code([face_index_map.shape, weight_map.shape, depth_map.shape, face_inv_map.shape], [face_index_map.dtype, weight_map.dtype, depth_map.dtype, face_inv_map.dtype], [faces, faces_inv, lock], 
    cuda_header='''

#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

namespace{
template <typename scalar_t,
        int image_size,
        int return_rgb,
        int return_alpha,
        int return_depth>
__global__ void forward_face_index_map_cuda_kernel(
        const scalar_t* faces,
        scalar_t* faces_inv,
        int32_t*  face_index_map,
        scalar_t*  weight_map,
        scalar_t*  depth_map,
        scalar_t*  face_inv_map,
        int batch_size,
        int num_faces,
        scalar_t near,
        scalar_t far,
        int threads_n_bits,
        int32_t* dev_tilemutex) {
    /* batch number, face, number, image size, face[v012][RGB] */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * (1<<threads_n_bits)) {
        return;
    }
    const int fn = i & ((1<<threads_n_bits) - 1);
     if (fn >= num_faces)
        return;
    const int bn = i>>threads_n_bits;
    i = bn * num_faces + fn;
    const int is = image_size;
    const scalar_t* face = &faces[i * 9];
    scalar_t* face_inv_g = &faces_inv[i * 9];

    /* return if backside */
    if ((face[7] - face[1]) * (face[3] - face[0]) < (face[4] - face[1]) * (face[6] - face[0]))
        return;

    /* p[num][xy]: x, y is normalized from [-1, 1] to [0, is - 1]. */
    scalar_t p[3][2];
    for (int num = 0; num < 3; num++) {
        for (int dim = 0; dim < 2; dim++) {
            p[num][dim] = 0.5 * (face[3 * num + dim] * is + is - 1);
        }
    }

    /* compute face_inv */
    scalar_t face_inv[9] = {
        p[1][1] - p[2][1], p[2][0] - p[1][0], p[1][0] * p[2][1] - p[2][0] * p[1][1],
        p[2][1] - p[0][1], p[0][0] - p[2][0], p[2][0] * p[0][1] - p[0][0] * p[2][1],
        p[0][1] - p[1][1], p[1][0] - p[0][0], p[0][0] * p[1][1] - p[1][0] * p[0][1]};
    scalar_t face_inv_denominator = (
        p[2][0] * (p[0][1] - p[1][1]) +
        p[0][0] * (p[1][1] - p[2][1]) +
        p[1][0] * (p[2][1] - p[0][1]));
    /* set to global memory */
    for (int k = 0; k < 9; k++) {
        face_inv[k] /= face_inv_denominator;
        face_inv_g[k] = face_inv[k];
    }

    /* compute the bounding box of triangle facet */
    scalar_t x_min=is, y_min=is, x_max=0, y_max=0;
    for (int num = 0; num < 3; num++) {
        if (p[num][0] < x_min)
            x_min = p[num][0];
        if (p[num][0] > x_max)
            x_max = p[num][0];
        if (p[num][1] < y_min)
            y_min = p[num][1];
        if (p[num][1] > y_max)
            y_max = p[num][1];
    }

    int ix_min = max(0, (int)x_min);
    int ix_max = min(is-1, (int)x_max);
    int iy_min = max(0, (int)y_min);
    int iy_max = min(is-1, (int)y_max);

    /* traverse each pixel in the bounding box */
    for (int xi=ix_min;xi<=ix_max;xi++) {
        for (int yi=iy_min;yi<=iy_max;yi++) {
            const scalar_t yp = (2. * yi + 1 - is) / is;
            const scalar_t xp = (2. * xi + 1 - is) / is;
            /* check [py, px] is inside the face */
            if (((yp - face[1]) * (face[3] - face[0]) < (xp - face[0]) * (face[4] - face[1])) ||
                ((yp - face[4]) * (face[6] - face[3]) < (xp - face[3]) * (face[7] - face[4])) ||
                ((yp - face[7]) * (face[0] - face[6]) < (xp - face[6]) * (face[1] - face[7])))
                continue;

            int i1 = bn * is * is + yi * is + xi;
            /* compute w = face_inv * p */
            scalar_t w[3];
            w[0] = face_inv[3 * 0 + 0] * xi + face_inv[3 * 0 + 1] * yi + face_inv[3 * 0 + 2];
            w[1] = face_inv[3 * 1 + 0] * xi + face_inv[3 * 1 + 1] * yi + face_inv[3 * 1 + 2];
            w[2] = face_inv[3 * 2 + 0] * xi + face_inv[3 * 2 + 1] * yi + face_inv[3 * 2 + 2];

            /* sum(w) -> 1, 0 < w < 1 */
            scalar_t w_sum = 0;
            for (int k = 0; k < 3; k++) {
                w[k] = min(max(w[k], 0.), 1.);
                w_sum += w[k];
            }
            for (int k = 0; k < 3; k++) {
                w[k] /= w_sum;
            }
            /* compute 1 / zp = sum(w / z) */
            const scalar_t zp = 1. / (w[0] / face[2] + w[1] / face[5] + w[2] / face[8]);
            if (zp <= near || far <= zp) {
                continue;
            }
            /* check z-buffer */
            bool isSet;
            do
            {
                isSet = (atomicCAS(&dev_tilemutex[i1], 0, 1) == 0);
                if (isSet)
                {
                    if (zp < depth_map[i1]) {
                        depth_map[i1] = zp;
                        face_index_map[i1] = fn;
                        for (int k = 0; k < 3; k++) {
                            weight_map[3 * i1 + k] = w[k];
                        }
                        if (return_depth) {
                            for (int k = 0; k < 9; k++) {
                                face_inv_map[9 * i1 + k] = face_inv[k];
                            }
                        }
                    }
                    __threadfence();
                    dev_tilemutex[i1] = 0;
                }
            } while (!isSet);
        }
    }
}

}
    ''',
    cuda_src=f'''
    @alias(faces, in0)
    @alias(faces_inv, in1)
    @alias(face_index_map, out0)
    @alias(weight_map, out1)
    @alias(depth_map, out2)
    @alias(face_inv_map, out3)

    thrust::device_ptr<out0_type> dev_ptr0(out0_p);
    thrust::fill(dev_ptr0, dev_ptr0 + out0->num, -1);

    cudaMemsetAsync(out1_p, 0, out1->size);

    thrust::device_ptr<out2_type> dev_ptr2(out2_p);
    thrust::fill(dev_ptr2, dev_ptr2 + out2->num, {far});

    cudaMemsetAsync(out3_p, 0, out3->size);

    const auto batch_size = faces_shape0;
    const auto num_faces = faces_shape1;
    const int threads = 256;
    const int threads_n_bits = NanoVector::get_nbits(num_faces) - 1;
    const dim3 blocks_1 ((batch_size * (1<<threads_n_bits) - 1) / threads +1);

    cudaMemsetAsync(in2_p, 0, in2->size);

    forward_face_index_map_cuda_kernel<
        float32,
        (int) {image_size},
        {return_rgb},
        {return_alpha},
        {return_depth}  
    ><<<blocks_1, threads>>>(
        faces_p,
        faces_inv_p,
        face_index_map_p,
        weight_map_p,
        depth_map_p,
        face_inv_map_p,
        (int) batch_size,
        (int) num_faces,
        (float32) {near},
        (float32) {far},
        threads_n_bits,
        in2_p);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in forward_face_index_map: %s\\n", cudaGetErrorString(err));
    ''')

def forward_texture_sampling(faces, textures, face_index_map, weight_map, depth_map, rgb_map, sampling_index_map, sampling_weight_map, image_size, eps):
    return jt.code([rgb_map.shape, sampling_index_map.shape, sampling_weight_map.shape], [rgb_map.dtype, sampling_index_map.dtype, sampling_weight_map.dtype], [faces, textures, face_index_map, weight_map, depth_map], 
    cuda_header='''
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

namespace{
template <typename scalar_t>
__global__ void forward_texture_sampling_cuda_kernel(
		const scalar_t* faces,
		const scalar_t* textures,
		const int32_t* face_index_map,
		const scalar_t* weight_map,
		const scalar_t* depth_map,
		scalar_t* rgb_map,
		int32_t* sampling_index_map,
        scalar_t* sampling_weight_map,
        size_t batch_size,
        int num_faces,
        int image_size,
        int texture_size,
        scalar_t eps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * image_size * image_size) {
        return;
    }
    const int face_index = face_index_map[i];
    
    if (face_index >= 0) {
        /*
            from global variables:
            batch number, num of faces, image_size, face[v012][RGB], pixel[RGB], weight[v012],
            texture[ts][ts][ts][RGB], sampling indices[8], sampling_weights[8];
        */
        const int bn = i / (image_size * image_size);
        const int nf = num_faces;
        const int ts = texture_size;
        const scalar_t* face = &faces[(bn * nf + face_index) * 9];
        const scalar_t* texture = &textures[(bn * nf + face_index) * ts * ts * ts * 3];
        scalar_t* pixel = &rgb_map[i * 3];
        const scalar_t* weight = &weight_map[i * 3];
        const scalar_t depth = depth_map[i];
        int32_t* sampling_indices = &sampling_index_map[i * 8];
        scalar_t* sampling_weights = &sampling_weight_map[i * 8];
    
        /* get texture index (float) */
        scalar_t texture_index_float[3];
        for (int k = 0; k < 3; k++) { scalar_t tif = weight[k] * (ts - 1) * (depth / (face[3 * k + 2]));
            tif = max(tif, 0.);
            tif = min(tif, ts - 1 - eps);
            texture_index_float[k] = tif;
        }
    
        /* blend */
        scalar_t new_pixel[3] = {0, 0, 0};
        for (int pn = 0; pn < 8; pn++) {
            scalar_t w = 1;                         // weight
            int texture_index_int[3];            // index in source (int)
            for (int k = 0; k < 3; k++) {
                if ((pn >> k) % 2 == 0) {
                    w *= 1 - (texture_index_float[k] - (int)texture_index_float[k]);
                    texture_index_int[k] = (int)texture_index_float[k];
                }
                else {
                    w *= texture_index_float[k] - (int)texture_index_float[k];
                    texture_index_int[k] = (int)texture_index_float[k] + 1;
                }
            }
    
            int isc = texture_index_int[0] * ts * ts + texture_index_int[1] * ts + texture_index_int[2];
            for (int k = 0; k < 3; k++)
                new_pixel[k] += w * texture[isc * 3 + k];
            sampling_indices[pn] = isc;
            sampling_weights[pn] = w;
        }
        for (int k = 0; k < 3; k++)
            pixel[k] = new_pixel[k];
    }
}
}
    ''',
    cuda_src=f'''
    @alias(faces, in0)
    @alias(textures, in1)
    @alias(face_index_map, in2)
    @alias(weight_map, in3)
    @alias(depth_map, in4)
    @alias(rgb_map, out0)
    @alias(sampling_index_map, out1)
    @alias(sampling_weight_map, out2)

    cudaMemsetAsync(out0_p, 0, out0->size);
    cudaMemsetAsync(out1_p, 0, out1->size);
    cudaMemsetAsync(out2_p, 0, out2->size);

    const auto batch_size = faces_shape0;
    const auto num_faces = faces_shape1;
    const auto texture_size = textures_shape2;
    const int threads = 512;
    const dim3 blocks ((batch_size * {image_size} * {image_size} - 1) / threads + 1);

    forward_texture_sampling_cuda_kernel<float32><<<blocks, threads>>>(
        faces_p,
        textures_p,
        face_index_map_p,
        weight_map_p,
        depth_map_p,
        rgb_map_p,
        sampling_index_map_p,
        sampling_weight_map_p,
        batch_size,
        num_faces,
        {image_size},
        texture_size,
        (float32) {eps});

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in forward_texture_sampling: %s\\n", cudaGetErrorString(err));
    ''')


def backward_pixel_map(faces, face_index_map, rgb_map, alpha_map, grad_rgb_map, grad_alpha_map, grad_faces, image_size, eps, return_rgb, return_alpha):
    return jt.code(grad_faces.shape, grad_faces.dtype, [faces, face_index_map, rgb_map, alpha_map, grad_rgb_map, grad_alpha_map], 
    cuda_header='''
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

namespace{

template <typename scalar_t>
__global__ void backward_pixel_map_cuda_kernel(
		const scalar_t* faces,
        int32_t*  face_index_map,
        scalar_t*  rgb_map,
        scalar_t*  alpha_map,
        scalar_t*  grad_rgb_map,
        scalar_t*  grad_alpha_map,
        scalar_t*  grad_faces,
        size_t batch_size,
        size_t num_faces,
        int image_size,
        scalar_t eps,
        int return_rgb,
        int return_alpha) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * num_faces) {
        return;
    }
    const int bn = i / num_faces;
    const int fn = i % num_faces;
    const int is = image_size;
    const scalar_t* face = &faces[i * 9];
    scalar_t grad_face[9] = {};

    /* check backside */
    if ((face[7] - face[1]) * (face[3] - face[0]) < (face[4] - face[1]) * (face[6] - face[0]))
        return;

    /* for each edge */
    for (int edge_num = 0; edge_num < 3; edge_num++) {
        /* set points of target edge */
        int pi[3];
        scalar_t pp[3][2];
        for (int num = 0; num < 3; num++)
            pi[num] = (edge_num + num) % 3;
        for (int num = 0; num < 3; num++) {
            for (int dim = 0; dim < 2; dim++) {
                pp[num][dim] = 0.5 * (face[3 * pi[num] + dim] * is + is - 1);
            }
        }

        /* for dy, dx */
        for (int axis = 0; axis < 2; axis++) {
            /* */
            scalar_t p[3][2];
            for (int num = 0; num < 3; num++) {
                for (int dim = 0; dim < 2; dim++) {
                    p[num][dim] = pp[num][(dim + axis) % 2];
                }
            }

            /* set direction */
            int direction;
            if (axis == 0) {
                if (p[0][0] < p[1][0])
                    direction = -1;
                else
                    direction = 1;
            } else {
                if (p[0][0] < p[1][0])
                    direction = 1;
                else
                    direction = -1;
            }

            /* along edge */
            int d0_from, d0_to;
            d0_from = max(ceil(min(p[0][0], p[1][0])), 0.);
            d0_to = min(max(p[0][0], p[1][0]), is - 1.);
            for (int d0 = d0_from; d0 <= d0_to; d0++) {
                /* get cross point */
                int d1_in, d1_out;
                const scalar_t d1_cross = (p[1][1] - p[0][1]) / (p[1][0] - p[0][0]) * (d0 - p[0][0]) + p[0][1];
                if (0 < direction)
                    d1_in = floor(d1_cross);
                else
                    d1_in = ceil(d1_cross);
                d1_out = d1_in + direction;

                /* continue if cross point is not shown */
                if (d1_in < 0 || is <= d1_in)
                    continue;
                if (d1_out < 0 || is <= d1_out)
                    continue;

                /* get color of in-pixel and out-pixel */
                scalar_t alpha_in;
                scalar_t alpha_out;
                scalar_t *rgb_in;
                scalar_t *rgb_out;
                int map_index_in, map_index_out;
                if (axis == 0) {
                    map_index_in = bn * is * is + d1_in * is + d0;
                    map_index_out = bn * is * is + d1_out * is + d0;
                }
                else {
                    map_index_in = bn * is * is + d0 * is + d1_in;
                    map_index_out = bn * is * is + d0 * is + d1_out;
                }
                if (return_alpha) {
                    alpha_in = alpha_map[map_index_in];
                    alpha_out = alpha_map[map_index_out];
                }
                if (return_rgb) {
                    rgb_in = &rgb_map[map_index_in * 3];
                    rgb_out = &rgb_map[map_index_out * 3];
                }

                /* out */
                bool is_in_fn = (face_index_map[map_index_in] == fn);
                if (is_in_fn) {
                    int d1_limit;
                    if (0 < direction)
                        d1_limit = is - 1;
                    else
                        d1_limit = 0;
                    int d1_from = max(min(d1_out, d1_limit), 0);
                    int d1_to = min(max(d1_out, d1_limit), is - 1);
                    scalar_t* alpha_map_p;
                    scalar_t* grad_alpha_map_p;
                    scalar_t* rgb_map_p;
                    scalar_t* grad_rgb_map_p;
                    int map_offset, map_index_from;
                    if (axis == 0) {
                        map_offset = is;
                        map_index_from = bn * is * is + d1_from * is + d0;
                    }
                    else {
                        map_offset = 1;
                        map_index_from = bn * is * is + d0 * is + d1_from;
                    }
                    if (return_alpha) {
                        alpha_map_p = &alpha_map[map_index_from];
                        grad_alpha_map_p = &grad_alpha_map[map_index_from];
                    }
                    if (return_rgb) {
                        rgb_map_p = &rgb_map[map_index_from * 3];
                        grad_rgb_map_p = &grad_rgb_map[map_index_from * 3];
                    }
                    for (int d1 = d1_from; d1 <= d1_to; d1++) {
                        scalar_t diff_grad = 0;
                        if (return_alpha) {
                            diff_grad += (*alpha_map_p - alpha_in) * *grad_alpha_map_p;
                        }
                        if (return_rgb) {
                            for (int k = 0; k < 3; k++)
                                diff_grad += (rgb_map_p[k] - rgb_in[k]) * grad_rgb_map_p[k];
                        }
                        if (return_alpha) {
                            alpha_map_p += map_offset;
                            grad_alpha_map_p += map_offset;
                        }
                        if (return_rgb) {
                            rgb_map_p += 3 * map_offset;
                            grad_rgb_map_p += 3 * map_offset;
                        }
                        if (diff_grad <= 0)
                            continue;
                        if (p[1][0] != d0) {
                            scalar_t dist = (p[1][0] - p[0][0]) / (p[1][0] - d0) * (d1 - d1_cross) * 2. / is;
                            dist = (0 < dist) ? dist + eps : dist - eps;
                            grad_face[pi[0] * 3 + (1 - axis)] -= diff_grad / dist;
                        }
                        if (p[0][0] != d0) {
                            scalar_t dist = (p[1][0] - p[0][0]) / (d0 - p[0][0]) * (d1 - d1_cross) * 2. / is;
                            dist = (0 < dist) ? dist + eps : dist - eps;
                            grad_face[pi[1] * 3 + (1 - axis)] -= diff_grad / dist;
                        }
                    }
                }

                /* in */
                {
                    int d1_limit;
                    scalar_t d0_cross2;
                    if ((d0 - p[0][0]) * (d0 - p[2][0]) < 0) {
                        d0_cross2 = (p[2][1] - p[0][1]) / (p[2][0] - p[0][0]) * (d0 - p[0][0]) + p[0][1];
                    }
                    else {
                        d0_cross2 = (p[1][1] - p[2][1]) / (p[1][0] - p[2][0]) * (d0 - p[2][0]) + p[2][1];
                    }
                    if (0 < direction)
                        d1_limit = ceil(d0_cross2);
                    else
                        d1_limit = floor(d0_cross2);
                    int d1_from = max(min(d1_in, d1_limit), 0);
                    int d1_to = min(max(d1_in, d1_limit), is - 1);

                    int* face_index_map_p;
                    scalar_t* alpha_map_p;
                    scalar_t* grad_alpha_map_p;
                    scalar_t* rgb_map_p;
                    scalar_t* grad_rgb_map_p;
                    int map_index_from;
                    int map_offset;
                    if (axis == 0)
                        map_offset = is;
                    else
                        map_offset = 1;
                    if (axis == 0) {
                        map_index_from = bn * is * is + d1_from * is + d0;
                    }
                    else {
                        map_index_from = bn * is * is + d0 * is + d1_from;
                    }
                    face_index_map_p = &face_index_map[map_index_from] - map_offset;
                    if (return_alpha) {
                        alpha_map_p = &alpha_map[map_index_from] - map_offset;
                        grad_alpha_map_p = &grad_alpha_map[map_index_from] - map_offset;
                    }
                    if (return_rgb) {
                        rgb_map_p = &rgb_map[map_index_from * 3] - 3 * map_offset;
                        grad_rgb_map_p = &grad_rgb_map[map_index_from * 3] - 3 * map_offset;
                    }

                    for (int d1 = d1_from; d1 <= d1_to; d1++) {
                        face_index_map_p += map_offset;
                        if (return_alpha) {
                            alpha_map_p += map_offset;
                            grad_alpha_map_p += map_offset;
                        }
                        if (return_rgb) {
                            rgb_map_p += 3 * map_offset;
                            grad_rgb_map_p += 3 * map_offset;
                        }
                        if (*face_index_map_p != fn)
                            continue;

                        scalar_t diff_grad = 0;
                        if (return_alpha) {
                            diff_grad += (*alpha_map_p - alpha_out) * *grad_alpha_map_p;
                        }
                        if (return_rgb) {
                            for (int k = 0; k < 3; k++)
                                diff_grad += (rgb_map_p[k] - rgb_out[k]) * grad_rgb_map_p[k];
                        }
                        if (diff_grad <= 0)
                            continue;

                        if (p[1][0] != d0) {
                            scalar_t dist = (p[1][0] - p[0][0]) / (p[1][0] - d0) * (d1 - d1_cross) * 2. / is;
                            dist = (0 < dist) ? dist + eps : dist - eps;
                            grad_face[pi[0] * 3 + (1 - axis)] -= diff_grad / dist;
                        }
                        if (p[0][0] != d0) {
                            scalar_t dist = (p[1][0] - p[0][0]) / (d0 - p[0][0]) * (d1 - d1_cross) * 2. / is;
                            dist = (0 < dist) ? dist + eps : dist - eps;
                            grad_face[pi[1] * 3 + (1 - axis)] -= diff_grad / dist;
                        }
                    }
                }
            }
        }
    }

    /* set to global gradient variable */
    for (int k = 0; k < 9; k++)
        grad_faces[i * 9 + k] = grad_face[k];
}

}
    ''',
    cuda_src=f'''
    @alias(faces, in0)
    @alias(face_index_map, in1)
    @alias(rgb_map, in2)
    @alias(alpha_map, in3)
    @alias(grad_rgb_map, in4)
    @alias(grad_alpha_map, in5)
    @alias(grad_faces, out0)

    cudaMemsetAsync(out0_p, 0, out0->size);

    const auto batch_size = faces_shape0;
    const auto num_faces = faces_shape1;
    const int threads = 512;
    const dim3 blocks ((batch_size * num_faces - 1) / threads + 1);

    backward_pixel_map_cuda_kernel<float32><<<blocks, threads>>>(
        faces_p,
        face_index_map_p,
        rgb_map_p,
        alpha_map_p,
        grad_rgb_map_p,
        grad_alpha_map_p,
        grad_faces_p,
        batch_size,
        num_faces,
        {image_size},
        (float32) {eps},
        {return_rgb},
        {return_alpha});

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in backward_pixel_map: %s\\n", cudaGetErrorString(err));
    ''')

def backward_textures(face_index_map, sampling_weight_map, sampling_index_map, grad_rgb_map, grad_textures, num_faces):
    return jt.code(grad_textures.shape, grad_textures.dtype, [face_index_map, sampling_weight_map, sampling_index_map, grad_rgb_map], 
    cuda_header='''
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

namespace{

template <typename scalar_t>
__global__ void backward_textures_cuda_kernel(
        const int32_t* face_index_map,
        scalar_t* sampling_weight_map,
        int32_t* sampling_index_map,
        scalar_t* grad_rgb_map,
        scalar_t* grad_textures,
        size_t batch_size,
        size_t num_faces,
        int image_size,
        size_t texture_size) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * image_size * image_size) {
        return;
    }
    const int face_index = face_index_map[i];
    if (0 <= face_index) {
        int is = image_size;
        int nf = num_faces;
        int ts = texture_size;
        int bn = i / (is * is);    // batch number [0 -> bs]
    
        scalar_t* grad_texture = &grad_textures[(bn * nf + face_index) * ts * ts * ts * 3];
        scalar_t* sampling_weight_map_p = &sampling_weight_map[i * 8];
        int* sampling_index_map_p = &sampling_index_map[i * 8];
        for (int pn = 0; pn < 8; pn++) {
            scalar_t w = *sampling_weight_map_p++;
            int isc = *sampling_index_map_p++;
            scalar_t* grad_texture_p = &grad_texture[isc * 3];
            scalar_t* grad_rgb_map_p = &grad_rgb_map[i * 3];
            for (int k = 0; k < 3; k++)
                atomicAdd(grad_texture_p++, w * *grad_rgb_map_p++);
        }
    }
}

}
    ''',
    cuda_src=f'''
    @alias(face_index_map, in0)
    @alias(sampling_weight_map, in1)
    @alias(sampling_index_map, in2)
    @alias(grad_rgb_map, in3)
    @alias(grad_textures, out0)

    cudaMemsetAsync(out0_p, 0, out0->size);

    const auto batch_size = face_index_map_shape0;
    const auto image_size = face_index_map_shape1;
    const auto texture_size = grad_textures_shape2;
    const int threads = 512;
    const dim3 blocks ((batch_size * image_size * image_size - 1) / threads + 1);

    backward_textures_cuda_kernel<float32><<<blocks, threads>>>(
        face_index_map_p,
        sampling_weight_map_p,
        sampling_index_map_p,
        grad_rgb_map_p,
        grad_textures_p,
        batch_size,
        {num_faces},
        image_size,
        texture_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in backward_textures: %s\\n", cudaGetErrorString(err));
    ''')

def backward_depth_map(faces, depth_map, face_index_map, face_inv_map, weight_map, grad_depth_map, grad_faces, image_size):
    return jt.code([faces, depth_map, face_index_map, face_inv_map, weight_map, grad_depth_map], [grad_faces], 
    cuda_header='''
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

namespace{

template <typename scalar_t>
__global__ void backward_depth_map_cuda_kernel(
        const scalar_t*  faces,
        const scalar_t*  depth_map,
        const int32_t* face_index_map,
        const scalar_t* face_inv_map,
        const scalar_t* weight_map,
        scalar_t*  grad_depth_map,
        scalar_t*  grad_faces,
        size_t batch_size,
        size_t num_faces,
        int image_size) {
    
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * image_size * image_size) {
        return;
    }
    const int fn = face_index_map[i];
    if (0 <= fn) {
        const int nf = num_faces;
        const int is = image_size;
        const int bn = i / (is * is);
        const scalar_t* face = &faces[(bn * nf + fn) * 9];
        const scalar_t depth = depth_map[i];
        const scalar_t depth2 = depth * depth;
        const scalar_t* face_inv = &face_inv_map[i * 9];
        const scalar_t* weight = &weight_map[i * 3];
        const scalar_t grad_depth = grad_depth_map[i];
        scalar_t* grad_face = &grad_faces[(bn * nf + fn) * 9];
        
        /* derivative wrt z */
        for (int k = 0; k < 3; k++) {
            const scalar_t z_k = face[3 * k + 2];
            atomicAdd(&grad_face[3 * k + 2], grad_depth * weight[k] * depth2 / (z_k * z_k));
        }
    
        /* derivative wrt x, y */
        scalar_t tmp[3] = {};
        for (int k = 0; k < 3; k++) {
            for (int l = 0; l < 3; l++) {
                tmp[k] += -face_inv[3 * l + k] / face[3 * l + 2];
            }
        }
        for (int k = 0; k < 3; k++) {
            for (int l = 0; l < 2; l++) {
            // k: point number, l: dimension
            atomicAdd(&grad_face[3 * k + l], -grad_depth * tmp[l] * weight[k] * depth2 * is / 2);
            }
        }
    }
}

}
    ''',
    cuda_src=f'''
    @alias(faces, in0)
    @alias(depth_map, in1)
    @alias(face_index_map, in2)
    @alias(face_inv_map, in3)
    @alias(weight_map, in4)
    @alias(grad_depth_map, in5)
    @alias(grad_faces, out0)

    const auto batch_size = faces_shape0;
    const auto num_faces = faces_shape1;
    const int threads = 512;
    const dim3 blocks ((batch_size * {image_size} * {image_size} - 1) / threads + 1);

    backward_depth_map_cuda_kernel<float32><<<blocks, threads>>>(
        faces_p,
        depth_map_p,
        face_index_map_p,
        face_inv_map_p,
        weight_map_p,
        grad_depth_map_p,
        grad_faces_p,
        batch_size,
        num_faces,
        {image_size});

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in backward_depth_map: %s\\n", cudaGetErrorString(err));

    ''')