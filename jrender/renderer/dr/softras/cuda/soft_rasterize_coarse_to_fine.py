import chunk
from math import ceil
import jittor as jt

# chunk_size : 512

def forward_soft_rasterize_coarse_to_fine(face_vertices, textures,
    faces_info, aggrs_info,
    soft_colors, faces_id_buffer,
    image_size, near, far, eps,
    sigma_val, func_id_dist, dist_eps,
    gamma_val, func_id_rgb, func_id_alpha,
    texture_sample_type, double_side, 
    bin_size, max_elems_per_bin):
    blur_radius = 0.01
    num_bins_edge = ceil(image_size / bin_size)
    if num_bins_edge > 25:
        raise ValueError("forward_soft_rasterize_coarse_to_fine got num_bins_edge, that's too many, try to enlarge the bin_size")
    return jt.code([faces_info.shape, aggrs_info.shape, soft_colors.shape, faces_id_buffer.shape], [faces_info.dtype, aggrs_info.dtype, soft_colors.dtype, faces_id_buffer.dtype],
    [face_vertices, textures],
    cuda_header='''
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <device_functions.h>
#include <sm_32_atomic_functions.h>

#define kEpsilon 1e-8
#define kMaxPointsPerPixel 128

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

namespace {

	template <typename scalar_t>
	__device__ __forceinline__ void barycentric_coordinate(scalar_t* w, const scalar_t x, const scalar_t y, const scalar_t* face_info) {
		w[0] = face_info[3 * 0 + 0] * x + face_info[3 * 0 + 1] * y + face_info[3 * 0 + 2];
		w[1] = face_info[3 * 1 + 0] * x + face_info[3 * 1 + 1] * y + face_info[3 * 1 + 2];
		w[2] = face_info[3 * 2 + 0] * x + face_info[3 * 2 + 1] * y + face_info[3 * 2 + 2];

	}


	template <typename scalar_t>
	__device__ __forceinline__ bool check_border(const scalar_t x, const scalar_t y, const scalar_t* face, const scalar_t threshold) {
		return (x > max(max(face[0], face[3]), face[6]) + threshold ||
			x < min(min(face[0], face[3]), face[6]) - threshold ||
			y > max(max(face[1], face[4]), face[7]) + threshold ||
			y < min(min(face[1], face[4]), face[7]) - threshold);
	}


	template <typename scalar_t>
	__device__ __forceinline__ bool check_face_frontside(const scalar_t* face) {
		return (face[7] - face[1]) * (face[3] - face[0]) < (face[4] - face[1]) * (face[6] - face[0]);
	}


	template <typename scalar_t>
	__device__ __forceinline__ bool check_pixel_inside(const scalar_t* w) {
		return w[0] <= 1 && w[0] >= 0 && w[1] <= 1 && w[1] >= 0 && w[2] <= 1 && w[2] >= 0;
	}


	template <typename scalar_t>
	__device__ __forceinline__ void barycentric_clip(scalar_t* w) {
		for (int k = 0; k < 3; k++) w[k] = max(min(w[k], 1.), 0.);
		const scalar_t w_sum = max(w[0] + w[1] + w[2], 1e-5);
		for (int k = 0; k < 3; k++) w[k] /= w_sum;
	}


	template <typename scalar_t>
	__device__ __forceinline__ void euclidean_p2f_distance(scalar_t& sign, scalar_t& dis_x, scalar_t& dis_y,
		scalar_t* w, scalar_t* t,
		const scalar_t* face, const scalar_t* face_info,
		const scalar_t xp, const scalar_t yp) {
		const scalar_t* face_sym = face_info + 9;
		const scalar_t* face_obt = face_info + 18;

		if (w[0] > 0 && w[1] > 0 && w[2] > 0 &&
			w[0] < 1 && w[1] < 1 && w[2] < 1) {
			// inside the triangle, w[0] + w[1] + w[2] = 0
			scalar_t dis_min = 100000000;
			scalar_t dis_x_min = 0;
			scalar_t dis_y_min = 0;
			scalar_t a0[3];
			scalar_t t0[3];
			for (int k = 0; k < 3; k++) {
				int v0 = k;
				int v1 = (k + 1) % 3;
				int v2 = (k + 2) % 3;
				a0[0] = face_sym[3 * v0 + 0] - face_sym[3 * v1 + 0];
				a0[1] = face_sym[3 * v0 + 1] - face_sym[3 * v1 + 1];
				a0[2] = face_sym[3 * v0 + 2] - face_sym[3 * v1 + 2];

				t0[v0] = (w[0] * a0[0] + w[1] * a0[1] + w[2] * a0[2] - a0[v1]) / (a0[v0] - a0[v1]);
				t0[v1] = 1 - t0[v0];
				t0[v2] = 0;

				t0[0] -= w[0];
				t0[1] -= w[1];
				t0[2] -= w[2];

				// calculate distance
				dis_x = t0[0] * face[0] + t0[1] * face[3] + t0[2] * face[6];
				dis_y = t0[0] * face[1] + t0[1] * face[4] + t0[2] * face[7];
				scalar_t dis = dis_x * dis_x + dis_y * dis_y;

				if (dis < dis_min) {
					dis_min = dis;
					dis_x_min = dis_x;
					dis_y_min = dis_y;
					t[0] = t0[0];
					t[1] = t0[1];
					t[2] = t0[2];
				}
			}
			dis_x = dis_x_min;
			dis_y = dis_y_min;
			sign = 1;
		}
		else {
			int v0 = -1;

			if (w[1] <= 0 && w[2] <= 0) {
				v0 = 0;
				if (face_obt[0] == 1 && (xp - face[0]) * (face[6] - face[0]) + (yp - face[1]) * (face[7] - face[1]) > 0) v0 = 2;
			}
			else if (w[2] <= 0 && w[0] <= 0) {
				v0 = 1;
				if (face_obt[1] == 1 && (xp - face[3]) * (face[0] - face[3]) + (yp - face[4]) * (face[1] - face[4]) > 0) v0 = 0;
			}
			else if (w[0] <= 0 && w[1] <= 0) {
				v0 = 2;
				if (face_obt[2] == 1 && (xp - face[6]) * (face[3] - face[6]) + (yp - face[7]) * (face[4] - face[7]) > 0) v0 = 1;
			}
			else
				if (w[0] <= 0) v0 = 1;
				else if (w[1] <= 0) v0 = 2;
				else if (w[2] <= 0) v0 = 0;

			const int v1 = (v0 + 1) % 3;
			const int v2 = (v0 + 2) % 3;

			scalar_t a0[3];

			a0[0] = face_sym[3 * v0 + 0] - face_sym[3 * v1 + 0];
			a0[1] = face_sym[3 * v0 + 1] - face_sym[3 * v1 + 1];
			a0[2] = face_sym[3 * v0 + 2] - face_sym[3 * v1 + 2];

			t[v0] = (w[0] * a0[0] + w[1] * a0[1] + w[2] * a0[2] - a0[v1]) / (a0[v0] - a0[v1]);
			t[v1] = 1 - t[v0];
			t[v2] = 0;

			// clamp to [0, 1]
			for (int k = 0; k < 3; k++) {
				t[k] = min(max(t[k], 0.), 1.);
				t[k] -= w[k];
			}

			// calculate distance
			dis_x = t[0] * face[0] + t[1] * face[3] + t[2] * face[6];
			dis_y = t[0] * face[1] + t[1] * face[4] + t[2] * face[7];
			sign = -1;
		}
	}


	template <typename scalar_t>
	__device__ __forceinline__ void forward_barycentric_p2f_distance(scalar_t& dis, const scalar_t* w) {
		dis = w[0] > w[1] ? (w[1] > w[2] ? w[2] : w[1]) : (w[0] > w[2] ? w[2] : w[0]);
		dis = dis > 0 ? dis * dis : -dis * dis;
	}

	template <typename scalar_t>
	__device__ __forceinline__ scalar_t forward_sample_texture(const scalar_t* texture, const scalar_t* w, const int R, const int k, const int texture_sample_type) {
		scalar_t texture_k;
		if (texture_sample_type == 0) { // sample surface color with resolution as R
			const int w_x = w[0] * R;
			const int w_y = w[1] * R;
			if ((w[0] + w[1]) * R - w_x - w_y <= 1) {
				texture_k = texture[(w_y * R + w_x) * 3 + k];
			}
			else {
				texture_k = texture[((R - 1 - w_y) * R + (R - 1 - w_x)) * 3 + k];
			}
		}
		else
			if (texture_sample_type == 1) { // sample vertex color
				texture_k = w[0] * texture[k] + w[1] * texture[3 + k] + w[2] * texture[6 + k];
			}
		return texture_k;
	}

	// triangle preprocessing
	template <typename scalar_t>
	__global__ void forward_soft_rasterize_inv_cuda_kernel(
		const scalar_t* __restrict__ faces,
		scalar_t* faces_info,
		int batch_size,
		int num_faces,
		int image_size) {
		/* batch number, face, number, image size, face[v012][RGB] */
		const int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= batch_size * num_faces) {
			return;
		}
		// const int is = image_size;
		const scalar_t* face = &faces[i * 9];
		scalar_t* face_inv = &faces_info[i * 27];
		scalar_t* face_sym = &faces_info[i * 27 + 9];
		scalar_t* face_obt = &faces_info[i * 27 + 18];

		/* return if backside */
		// if ((face[7] - face[1]) * (face[3] - face[0]) < (face[4] - face[1]) * (face[6] - face[0]))
			// return;
		/* p[num][xy]: x, y is (-1, 1). */
		scalar_t p[3][2];
		for (int num = 0; num < 3; num++) {
			for (int dim = 0; dim < 2; dim++) {
				p[num][dim] = face[3 * num + dim]; // no normalize
			}
		}
		/* compute face_inv */
		scalar_t face_inv_star[9] = {
			p[1][1] - p[2][1], p[2][0] - p[1][0], p[1][0] * p[2][1] - p[2][0] * p[1][1],
			p[2][1] - p[0][1], p[0][0] - p[2][0], p[2][0] * p[0][1] - p[0][0] * p[2][1],
			p[0][1] - p[1][1], p[1][0] - p[0][0], p[0][0] * p[1][1] - p[1][0] * p[0][1] };
		scalar_t face_inv_determinant = (
			p[2][0] * (p[0][1] - p[1][1]) +
			p[0][0] * (p[1][1] - p[2][1]) +
			p[1][0] * (p[2][1] - p[0][1]));
		face_inv_determinant = face_inv_determinant > 0 ? max(face_inv_determinant, 1e-10) : min(face_inv_determinant, -1e-10);
		/* set to global memory */
		for (int k = 0; k < 9; k++) {
			face_inv[k] = face_inv_star[k] / face_inv_determinant;
		}
		/* F * F.T */
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 3; k++) {
				face_sym[j * 3 + k] = face[j * 3 + 0] * face[k * 3 + 0] +
					face[j * 3 + 1] * face[k * 3 + 1] +
					1;
			}
		}
		/* check if one arc is obt arc */
		for (int k = 0; k < 3; k++) {
			const int k0 = k;
			const int k1 = (k + 1) % 3;
			const int k2 = (k + 2) % 3;
			if ((p[k1][0] - p[k0][0]) * (p[k2][0] - p[k0][0]) + (p[k1][1] - p[k0][1]) * (p[k2][1] - p[k0][1]) < 0) {
				face_obt[k0] = 1;
				break;
			}
		}
	}

    struct Pixel{
        int id;
        float z;
    };

	template <typename scalar_t>
	__global__ void forward_soft_rasterize_cuda_kernel(
		const scalar_t* __restrict__ faces,
		const scalar_t* __restrict__ textures,
		const scalar_t* __restrict__ faces_info,
		const int* bin_faces,
        const int* num_bin_faces,
		scalar_t* aggrs_info,
		scalar_t* soft_colors,
        int* faces_id_buffers,
		int B,
		int M,
		int bin_size,
		int batch_size,
		int num_faces,
		int image_size,
        int max_faces_id,
		int texture_size,
		int texture_res,
		float near,
		float far,
		float eps,
		float sigma_val,
		int func_id_dist,
		float dist_eps,
		float gamma_val,
		int func_id_rgb,
		int func_id_alpha,
		int texture_sample_type,
		int double_side) {

		////////////////////////
		////////////////////////

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

		const scalar_t* face;
		const scalar_t* texture;
		const scalar_t* face_info;

		const scalar_t threshold = dist_eps * sigma_val;

		// Initialize pixel color
		scalar_t soft_color[4] = { 1., 1., 1., 0. };
		if (func_id_alpha == 2) soft_color[3] = 1.;
		scalar_t softmax_sum = exp(eps / gamma_val);
		scalar_t softmax_max = eps;
		for (int k = 0; k < 3; k++) {
			if (func_id_rgb == 0) { // hard assign, set to background
				soft_color[k] = soft_colors[(bn * 4 + k) * (is * is) + pn];
			}
			else
				if (func_id_rgb == 1) {
					soft_color[k] = soft_colors[(bn * 4 + k) * (is * is) + pn] * softmax_sum; // initialize background color
				}
		}
		scalar_t depth_min = 10000000;
		int face_index_min = -1;
        int faces_overlap_pixel = 0;
        int faces_to_search_max = num_bin_faces[bn * B * B  + by * B + bx];
        Pixel q[kMaxPointsPerPixel];
        int q_size = 0;
        float q_max_z = -1;
        int q_max_id = -1;

		for (int m = 0; m < M; m++) {
            
			int fn = bin_faces[bn * B * B * M + by * B * M + bx * M + m];
            
            if (fn < 0){
                continue;
            }
            
            if (faces_overlap_pixel >= faces_to_search_max){
                break;
            }
            faces_overlap_pixel++;
            
            
			face = &faces[fn * 9];
			texture = &textures[fn * texture_size * 3];
			face_info = &faces_info[fn * 27];

			if (check_border(xp, yp, face, sqrt(threshold))) continue; // triangle too far away from pixel

			scalar_t dis;
			scalar_t dis_x;
			scalar_t dis_y;
			scalar_t t[3];
			scalar_t w[3];
			scalar_t w_clip[3];
			scalar_t sign;
			scalar_t soft_fragment;

			// compute barycentric coordinate w
			barycentric_coordinate(w, xp, yp, face_info);

			// compute probability map based on distance functions
			if (func_id_dist == 0) { // hard assign
				soft_fragment = check_pixel_inside(w) ? 1. : 0.;
				if (soft_fragment == 0.) continue; // ignore triangle outside of the pixel
			}
			else
				if (func_id_dist == 1) { // barycentric distance
					forward_barycentric_p2f_distance(dis, w);
					if (-dis >= threshold) continue; // ignore triangle far away from the pixel
					soft_fragment = 1. / (1. + exp(-dis / sigma_val));
				}
				else
					if (func_id_dist == 2) { // euclidean distance
						euclidean_p2f_distance(sign, dis_x, dis_y, w, t, face, face_info, xp, yp);
						dis = dis_x * dis_x + dis_y * dis_y;
						if (sign < 0 && dis >= threshold) continue; // ignore triangle far away from the pixel
						soft_fragment = 1. / (1. + exp(-sign * dis / sigma_val));
					}

			/////////////////////////////////////////////////////

			// aggragate for alpha channel
			if (func_id_alpha == 0) { // hard assign
				if (soft_fragment > 0.5) soft_color[3] = 1.;
			}
			else
				if (func_id_alpha == 1) { // Sum
					soft_color[3] += soft_fragment;
				}
				else
					if (func_id_alpha == 2) { // Logical-Or
						soft_color[3] *= 1. - soft_fragment;
					}

			/////////////////////////////////////////////////////

			for (int k = 0; k < 3; k++) w_clip[k] = w[k];
			barycentric_clip(w_clip);
			const scalar_t zp = 1. / (w_clip[0] / face[2] + w_clip[1] / face[5] + w_clip[2] / face[8]);
			if (zp < near || zp > far) continue; // triangle out of screen, pass

            ///////////////////////////////////////////////////
            // choose Topk  
            if (q_size < max_faces_id){
                q[q_size] = {fn % num_faces,zp};
                if (zp > q_max_z){
                    q_max_z = zp;
                    q_max_id = q_size;
                }
                q_size++;
            } else if (zp < q_max_z){
                q[q_max_id] = {fn % num_faces,zp};
                q_max_z = -1;
                for(int k = 0; k < q_size; k++){
                    if(q[k].z > q_max_z){
                        q_max_z = q[k].z;
                        q_max_id = k;
                    }
                }
            }


			/////////////////////////////////////////////////////
			// aggregate for rgb channels
			if (func_id_rgb == 0) { // Hard assign
				if (zp < depth_min && check_pixel_inside(w) && (double_side || check_face_frontside(face))) {
					depth_min = zp;
					face_index_min = fn;
					for (int k = 0; k < 3; k++) {
						soft_color[k] = forward_sample_texture(texture, w_clip, texture_res, k, texture_sample_type);
					}
				}
			}
			else
				if (func_id_rgb == 1) { // D * Softmax (Z)
					if (check_face_frontside(face) || double_side) {
						const scalar_t zp_norm = (far - zp) / (far - near);
						scalar_t exp_delta_zp = 1.;
						if (zp_norm > softmax_max) {
							exp_delta_zp = exp((softmax_max - zp_norm) / gamma_val);
							softmax_max = zp_norm;
						}
						const scalar_t exp_z = exp((zp_norm - softmax_max) / gamma_val);
						softmax_sum = exp_delta_zp * softmax_sum + exp_z * soft_fragment;
						for (int k = 0; k < 3; k++) {
							const scalar_t color_k = forward_sample_texture(texture, w_clip, texture_res, k, texture_sample_type);
							soft_color[k] = exp_delta_zp * soft_color[k] + exp_z * soft_fragment * color_k;// * soft_fragment;
						}
					}
				}
            
		}

		//////////////////////////////////////////////
        
		// finalize aggregation
		if (func_id_alpha == 0) {
			soft_colors[(bn * 4 + 3) * (is * is) + pn] = soft_color[3];
		}
		else
			if (func_id_alpha == 1) {
				soft_colors[(bn * 4 + 3) * (is * is) + pn] = soft_color[3] / nf;
			}
			else
				if (func_id_alpha == 2) {
					soft_colors[(bn * 4 + 3) * (is * is) + pn] = 1. - soft_color[3];
				}

		if (func_id_rgb == 0) {
			if (face_index_min != -1)
				for (int k = 0; k < 3; k++) {
					soft_colors[(bn * 4 + k) * (is * is) + pn] = soft_color[k];
				}
			aggrs_info[(bn * 2 + 0) * (is * is) + pn] = depth_min;
			aggrs_info[(bn * 2 + 1) * (is * is) + pn] = face_index_min;
		}
		else
			if (func_id_rgb == 1) {
				for (int k = 0; k < 3; k++) {
					soft_colors[(bn * 4 + k) * (is * is) + pn] = soft_color[k] / softmax_sum;
				}
				aggrs_info[(bn * 2 + 0) * (is * is) + pn] = softmax_sum;
				aggrs_info[(bn * 2 + 1) * (is * is) + pn] = softmax_max;
			}
        for (int k = 0; k < q_size; k++) {
            faces_id_buffers[(bn * max_faces_id + k) * (is * is) + pn] = q[k].id;
        }
	}
}

    ''',
    cuda_src=f'''
@alias(faces, in0)
@alias(textures, in1)
@alias(faces_info, out0)
@alias(aggrs_info, out1)
@alias(soft_colors, out2)
@alias(faces_id_buffer,out3)

cudaMemsetAsync(out0_p, 0, out0->size);
cudaMemsetAsync(out1_p, 0, out1->size);
cudaMemsetAsync(out2_p, 0, out2->size);
cudaMemsetAsync(out3_p, -1, out3->size);

const auto batch_size = faces_shape0;
const auto num_faces = faces_shape1;
const auto max_faces_id = faces_id_buffer_shape1;
const auto texture_size = textures_shape2;
const auto texture_res = int(sqrt(texture_size));
const size_t threads_1 = 512;
const size_t blocks_1 = ((batch_size * num_faces - 1) / threads_1 + 1);

float* bboxes;
bool* should_skip;
cudaMalloc((void**)&bboxes, batch_size * num_faces * 4 * sizeof(float));
cudaMalloc((void**)&should_skip, batch_size * num_faces * sizeof(bool));

forward_soft_rasterize_inv_cuda_kernel<float32><<<blocks_1, threads_1>>>(
    faces_p,
    faces_info_p,
    batch_size,
    num_faces,
    {image_size});

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


const size_t threads_4 = 128;
const size_t blocks_4 = ((batch_size * {bin_size} * {bin_size} * num_bins_edge * num_bins_edge - 1) / threads_4 + 1);
forward_soft_rasterize_cuda_kernel<float32><<<blocks_4, threads_4>>>(
    faces_p,
    textures_p,
    faces_info_p,
    bin_elems,
    elems_per_bin,
    aggrs_info_p,
    soft_colors_p,
    faces_id_buffer_p,
    num_bins_edge,
    {max_elems_per_bin},
    {bin_size},
    batch_size,
    num_faces,
    {image_size},
    max_faces_id,
    texture_size,
    texture_res,
    {near},
    {far},
    {eps},
    {sigma_val},
    {func_id_dist},
    {dist_eps},
    {gamma_val},
    {func_id_rgb},
    {func_id_alpha},
    {texture_sample_type},
    {double_side});


cudaFree(elems_per_bin);
cudaFree(bin_elems);


cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) 
    printf("Error in forward_soft_rasterize_coarse_to_fine: %s\\n", cudaGetErrorString(err));
''')