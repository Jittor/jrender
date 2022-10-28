import jittor as jt
from math import ceil

def forward_face_index_map_coarse_to_fine(
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
    return_depth,
    bin_size,
    max_elems_per_bin):
    blur_radius = 0.01
    num_bins_edge = ceil(image_size / bin_size)
    if num_bins_edge > 27:
        raise ValueError("forward_soft_rasterize_coarse_to_fine got num_bins_edge, that's too many, try to enlarge the bin_size")
    return jt.code([face_index_map.shape, weight_map.shape, depth_map.shape, face_inv_map.shape], [face_index_map.dtype, weight_map.dtype, depth_map.dtype, face_inv_map.dtype], [faces], 
    cuda_header='''

#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <math.h>
#include <device_functions.h>
#include <sm_32_atomic_functions.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>



// for the older gpus atomicAdd with double arguments does not exist
#if  __CUDA_ARCH__ < 600 and defined(__CUDA_ARCH__)
static __inline__ __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) } while (assumed != old);
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

class BitMask {
public:
    __device__ BitMask(unsigned int* data, int H, int W, int N)
        : data(data), H(H), W(W), B(8 * sizeof(unsigned int)), D(N / B) {
        // TODO: check if the data is null.
        N = ceilf(N % 32); // take ceil incase N % 32 != 0
        block_clear(); // clear the data
    }

    // Use all threads in the current block to clear all bits of this BitMask
    __device__ void block_clear() {
        for (int i = threadIdx.x; i < H * W * D; i += blockDim.x) {
            data[i] = 0;
        }
        __syncthreads();
    }

    __device__ int _get_elem_idx(int y, int x, int d) {
        return y * W * D + x * D + d / B;
    }

    __device__ int _get_bit_idx(int d) {
        return d % B;
    }

    // Turn on a single bit (y, x, d)
    __device__ void set(int y, int x, int d) {
        int elem_idx = _get_elem_idx(y, x, d);
        int bit_idx = _get_bit_idx(d);
        const unsigned int mask = 1U << bit_idx;
        atomicOr(data + elem_idx, mask);
    }

    // Turn off a single bit (y, x, d)
    __device__ void unset(int y, int x, int d) {
        int elem_idx = _get_elem_idx(y, x, d);
        int bit_idx = _get_bit_idx(d);
        const unsigned int mask = ~(1U << bit_idx);
        atomicAnd(data + elem_idx, mask);
    }

    // Check whether the bit (y, x, d) is on or off
    __device__ bool get(int y, int x, int d) {
        int elem_idx = _get_elem_idx(y, x, d);
        int bit_idx = _get_bit_idx(d);
        return (data[elem_idx] >> bit_idx) & 1U;
    }

    // Compute the number of bits set in the row (y, x, :)
    __device__ int count(int y, int x) {
        int total = 0;
        for (int i = 0; i < D; ++i) {
            int elem_idx = y * W * D + x * D + i;
            unsigned int elem = data[elem_idx];
            total += __popc(elem);
        }
        return total;
    }

private:
    unsigned int* data;
    int H, W, B, D;
};

__global__ void TriangleBoundingBoxKernel(
    const float* face_verts, // (F, 3, 3)
    const int F,
    const float blur_radius,
    float* bboxes, // (4, F)
    bool* skip_face) { // (F,)
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_threads = blockDim.x * gridDim.x;
    const float sqrt_radius = sqrt(blur_radius);
    for (int f = tid; f < F; f += num_threads) {
        const float v0x = face_verts[f * 9 + 0 * 3 + 0];
        const float v0y = face_verts[f * 9 + 0 * 3 + 1];
        const float v0z = face_verts[f * 9 + 0 * 3 + 2];
        const float v1x = face_verts[f * 9 + 1 * 3 + 0];
        const float v1y = face_verts[f * 9 + 1 * 3 + 1];
        const float v1z = face_verts[f * 9 + 1 * 3 + 2];
        const float v2x = face_verts[f * 9 + 2 * 3 + 0];
        const float v2y = face_verts[f * 9 + 2 * 3 + 1];
        const float v2z = face_verts[f * 9 + 2 * 3 + 2];
        const float xmin = min(min(v0x, v1x), v2x) - sqrt_radius;
        const float xmax = max(max(v0x, v1x), v2x) + sqrt_radius;
        const float ymin = min(min(v0y, v1y), v2y) - sqrt_radius;
        const float ymax = max(max(v0y, v1y), v2y) + sqrt_radius;
        const float zmin = min(min(v0z, v1z), v2z);
        const bool skip = zmin < kEpsilon;
        bboxes[0 * F + f] = xmin;
        bboxes[1 * F + f] = xmax;
        bboxes[2 * F + f] = ymin;
        bboxes[3 * F + f] = ymax;
        skip_face[f] = skip;
    }
}

__global__ void RasterizeCoarseCudaKernel(
    const float* bboxes, // (4, E) (xmin, xmax, ymin, ymax)
    const bool* should_skip, // (E,)
    const int N,                  //batch_size
    const int E,                  //num_faces
    const int is,
    const int bin_size,
    const int chunk_size,
    const int max_elem_per_bin,
    int* elems_per_bin,
    int* bin_elems) {
    
    extern __shared__ char sbuf[];
    const int M = max_elem_per_bin;
    // Integer divide round up
    const int num_bins_edge = 1 + (is - 1) / bin_size;

    // Size of half a pixel in NDC units is the NDC half range
    // divided by the corresponding image dimension
    const float half_pix = float(1 / is);

    // This is a boolean array of shape (num_bins_y, num_bins_x, chunk_size)
    // stored in shared memory that will track whether each elem in the chunk
    // falls into each bin of the image.
    BitMask binmask((unsigned int*)sbuf, num_bins_edge, num_bins_edge, chunk_size);

    // Have each block handle a chunk of elements
    const int chunks_per_batch = 1 + (E - 1) / chunk_size;
    const int num_chunks = N * chunks_per_batch;

    for (int chunk = blockIdx.x; chunk < num_chunks; chunk += gridDim.x) {
        const int batch_idx = chunk / chunks_per_batch; // batch index
        const int chunk_idx = chunk % chunks_per_batch;
        const int elem_chunk_start_idx = batch_idx * E + chunk_idx * chunk_size;

        binmask.block_clear();
        const int64_t elem_start_idx = batch_idx * E;
        const int64_t elem_stop_idx = (batch_idx + 1) * E;
        // Have each thread handle a different face within the chunk
        for (int e = threadIdx.x; e < chunk_size; e += blockDim.x) {
            const int e_idx = elem_chunk_start_idx + e;

            /*
            if (blockIdx.x == 16){
                printf("chunk: %d threadIdx.x: %d num_chunks: %d e: %d chunk_idx: %d elem_chunk_start_idx: %d e_idx: %d \\n", 
                                chunk,threadIdx.x,num_chunks,e,chunk_idx,elem_chunk_start_idx,e_idx);
            }
            */

            // Check that we are still within the same element of the batch
            if (e_idx >= elem_stop_idx || e_idx < elem_start_idx) {
                continue;
            }

            if (should_skip[e_idx]) {
                continue;
            }
            const float xmin = bboxes[0 * E * N + e_idx];
            const float xmax = bboxes[1 * E * N + e_idx];
            const float ymin = bboxes[2 * E * N + e_idx];
            const float ymax = bboxes[3 * E * N + e_idx];

            // Brute-force search over all bins; TODO(T54294966) something smarter.
            for (int by = 0; by < num_bins_edge; ++by) {
                // Y coordinate of the top and bottom of the bin.
                // PixToNdc gives the location of the center of each pixel, so we
                // need to add/subtract a half pixel to get the true extent of the bin.
                // Reverse ordering of Y axis so that +Y is upwards in the image.
                const float bin_y_min =
                    float(2 * by * bin_size + 1) / is - 1 - half_pix;
                const float bin_y_max =
                    float(2 * ((by + 1) * bin_size - 1) + 1) / is - 1 + half_pix;
                const bool y_overlap = (ymin <= bin_y_max) && (bin_y_min < ymax);

                for (int bx = 0; bx < num_bins_edge; ++bx) {
                    // X coordinate of the left and right of the bin.
                    // Reverse ordering of x axis so that +X is left.
                    const float bin_x_max =
                     float(2 * ((bx + 1) * bin_size - 1) + 1) / is - 1 + half_pix;
                    const float bin_x_min =
                     float(2 * (bx * bin_size) + 1) / is - 1 - half_pix;

                    const bool x_overlap = (xmin <= bin_x_max) && (bin_x_min < xmax);
                    if (y_overlap && x_overlap) {
                        binmask.set(by, bx, e);

                    /*
                    if (blockIdx.x == 0){
                        printf("bin_y_min: %f bin_y_max: %f bx: %d by: %d chunk: %d threadIdx.x: %d num_chunks: %d e: %d chunk_idx: %d elem_chunk_start_idx: %d e_idx: %d \\n", 
                                bin_y_min,bin_y_max,bx,by,chunk,threadIdx.x,num_chunks,e,chunk_idx,elem_chunk_start_idx,e_idx);
                    }
                    */

                    }

                }
            }
        }
        __syncthreads();
        // Now we have processed every elem in the current chunk. We need to
        // count the number of elems in each bin so we can write the indices
        // out to global memory. We have each thread handle a different bin.
        for (int byx = threadIdx.x; byx < num_bins_edge * num_bins_edge;
            byx += blockDim.x) {
            const int by = byx / num_bins_edge;
            const int bx = byx % num_bins_edge;
            const int count = binmask.count(by, bx);
            const int elems_per_bin_idx =
                batch_idx * num_bins_edge * num_bins_edge + by * num_bins_edge + bx;

            // This atomically increments the (global) number of elems found
            // in the current bin, and gets the previous value of the counter;
            // this effectively allocates space in the bin_faces array for the
            // elems in the current chunk that fall into this bin.
            const int start = atomicAdd(elems_per_bin + elems_per_bin_idx, count);
            if (start + count > M) {
                // The number of elems in this bin is so big that they won't fit.
                // We print a warning using CUDA's printf. This may be invisible
                // to notebook users, but apparent to others. It would be nice to
                // also have a Python-friendly warning, but it is not obvious
                // how to do this without slowing down the normal case.
                const char* warning =
                    "Bin size was too small in the coarse rasterization phase. "
                    "This caused an overflow, meaning output may be incomplete. "
                    "To solve, "
                    "try increasing max_faces_per_bin / max_points_per_bin, "
                    "decreasing bin_size, "
                    "or setting bin_size to 0 to use the naive rasterization.";
                printf(warning);
                continue;
            }

            // Now loop over the binmask and write the active bits for this bin
            // out to bin_faces.
            int next_idx = batch_idx * num_bins_edge * num_bins_edge * M +
                by * num_bins_edge * M + bx * M + start;
            for (int e = 0; e < chunk_size; ++e) {
                if (binmask.get(by, bx, e)) {
                    // TODO(T54296346) find the correct method for handling errors in
                    // CUDA. Throw an error if num_faces_per_bin > max_faces_per_bin.
                    // Either decrease bin size or increase max_faces_per_bin
                    bin_elems[next_idx] = elem_chunk_start_idx + e;
                    next_idx++;
                }
            }
        }
        __syncthreads();
    }
    
}

namespace{
template <typename scalar_t,
        int image_size,
        int return_rgb,
        int return_alpha,
        int return_depth>
__global__ void forward_face_index_map_cuda_kernel(
        const scalar_t* faces,
        int32_t*  face_index_map,
        scalar_t*  weight_map,
        scalar_t*  depth_map,
        scalar_t*  face_inv_map,
        int batch_size,
        int num_faces,
        scalar_t near,
        scalar_t far,
        int* bin_elems,
        int* elems_per_bin,
        int bin_size,
        int B,
        int M) {
    /* batch number, face, number, image size, face[v012][RGB] */

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int is = image_size;
    const int nf = num_faces;

    int ii = i;
    const int bn = ii / (B * B * bin_size * bin_size);
    if (bn >= batch_size){
        return;
    }
    ii %= B * B * bin_size * bin_size;
    // bin index y
    const int by = ii / (B * bin_size * bin_size);
    ii %= B * bin_size * bin_size;
    // bin index y
    const int bx = ii / (bin_size * bin_size);
    // pixel within the bin
    ii %= bin_size * bin_size;
    const int yi = (ii / bin_size + by * bin_size);
    const int xi = ii % bin_size + bx * bin_size;
    if (yi >= is || xi >= is){
        return;
    }
    const scalar_t yp = (2. * yi + 1. - is) / is;
    const scalar_t xp = (2. * xi + 1. - is) / is;
    const int pn = (is - 1 - yi) * is + xi;

    scalar_t* face;
    int32* face_index_m = face_index_m[bn * is * is + pn];
    scalar_t* weight_m = face_index_m[(bn * is * is + pn) * 3];
    scalar_t* depth_m = depth_map[bn * is * is + pn];
    scalar_t* face_inv_m = depth_inv_m[(bn * is * is + pn) * 9];

    scalar_t depth_min = 10000000;
    int face_index_min = -1;
    int faces_overlap_pixel = 0;
    int faces_to_search_max = num_bin_faces[bn * B * B  + by * B + bx];
    //////////////
    for (int m = 0; m < M; m++){

        int fn = bin_faces[bn * B * B * M + by * B * M + bx * M + m];
        
        if (fn < 0){
            continue;
        }
        
        if (faces_overlap_pixel >= faces_to_search_max){
            break;
        }
        faces_overlap_pixel++;

        face = &faces[i * 9];
        scalar_t w[3];
        scalar_t w_clip[3];
        scalar_t dis;

        /* return if backside */
        if ((face[7] - face[1]) * (face[3] - face[0]) < (face[4] - face[1]) * (face[6] - face[0])) continue;

        if (check_border(xp, yp, face, sqrt(threshold))) continue; // triangle too far away from pixel

        /////
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
        for (int k = 0; k < 9; k++) {
            face_inv[k] /= face_inv_denominator;
        }

        barycentric_coordinate(w, xp, yp, face_inv);

        forward_barycentric_p2f_distance(dis, w);
        if (-dis >= threshold) continue; // ignore triangle far away from the pixel
        
        for (int k = 0; k < 3; k++) w_clip[k] = w[k];
        barycentric_clip(w_clip);
        const scalar_t zp = 1. / (w_clip[0] / face[2] + w_clip[1] / face[5] + w_clip[2] / face[8]);
        if (zp < near || zp > far) continue; // triangle out of screen, pass
        if (zp < depth_min && check_pixel_inside(w) && (double_side || check_face_frontside(face))) {
            depth_min = zp;
            face_index_min = fn;
            }

        /////////////

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

}
    ''',
    cuda_src=f'''
    @alias(faces, in0)
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


    //////
    float* bboxes;
    bool* should_skip;
    cudaMalloc((void**)&bboxes, batch_size * num_faces * 4 * sizeof(float));
    cudaMalloc((void**)&should_skip, batch_size * num_faces * sizeof(bool));

    const size_t blocks_2 = 128;
    const size_t threads_2 = 256;
    TriangleBoundingBoxKernel<<<blocks_2, threads_2>>>(
        faces_p,
        batch_size * num_faces,
        {blur_radius},
        bboxes,
        should_skip);

    const int num_bins_edge = 1 + ({image_size} - 1) / {bin_size};
    int* elems_per_bin;
    int* bin_elems;
    size_t size2 = batch_size * num_bins_edge * num_bins_edge * sizeof(int);
    cudaMalloc((void**)&elems_per_bin, size2);
    cudaMemset(elems_per_bin,0,size2);
    size_t size1 = batch_size * num_bins_edge * num_bins_edge * {max_elems_per_bin} * sizeof(int);
    cudaMalloc((void**)&bin_elems, size1);
    cudaMemset(bin_elems,-1,size1);
    cudaDeviceSynchronize();

    const size_t blocks_3 = 64;
    const size_t threads_3 = 512;
    const int chunk_size = 512;
    const int shared_size = num_bins_edge * num_bins_edge * chunk_size / 8;
    RasterizeCoarseCudaKernel<<<blocks_3, threads_3, shared_size>>>(
        bboxes,
        should_skip,
        batch_size,
        num_faces,
        {image_size},
        {bin_size},
        chunk_size,
        {max_elems_per_bin},
        elems_per_bin,
        bin_elems);
    cudaDeviceSynchronize();

    cudaFree(bboxes);
    cudaFree(should_skip);
    //////
    const size_t threads_4 = 128;
    const size_t blocks_4 = ((batch_size * {bin_size} * {bin_size} * num_bins_edge * num_bins_edge - 1) / threads_4 + 1);
    forward_face_index_map_cuda_kernel<
        float32,
        (int) {image_size},
        {return_rgb},
        {return_alpha},
        {return_depth}  
    ><<<blocks_4, threads_4>>>(
        faces_p,
        face_index_map_p,
        weight_map_p,
        depth_map_p,
        face_inv_map_p,
        (int) batch_size,
        (int) num_faces,
        (float32) {near},
        (float32) {far},
        bin_elems,
        elems_per_bin,
        {bin_size},
        num_bins_edge,
        {max_elems_per_bin});

    cudaFree(bboxes);
    cudaFree(should_skip);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in forward_face_index_map: %s\\n", cudaGetErrorString(err));
    ''')

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


/*
// for the older gpus atomicAdd with double arguments does not exist
#if  __CUDA_ARCH__ < 600 and defined(__CUDA_ARCH__)
static __inline__ __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) } while (assumed != old);
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif
*/

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