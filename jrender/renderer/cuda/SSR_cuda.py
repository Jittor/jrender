import math
import jittor as jt
from jittor import nn
from skimage.io import imsave

def SSR_cuda(color, world_buffer, normal_buffer, faces_ind_buffer, ssr_faces, width, step):        
    image = jt.zeros_like(color)
    depth = world_buffer[:,:,2]
    return jt.code(image.shape, image.dtype, [color, depth, normal_buffer, faces_ind_buffer, ssr_faces],
                   cuda_header='''
#include <cuda.h>
#include <cuda_runtime.h>

__device__ float3 operator*(float a, const float3& v) {
	return make_float3(v.x * a, v.y * a, v.z * a);
}

__device__ float2 operator*(float a, const float2& v) {
	return make_float2(v.x * a, v.y * a);
}

__device__ float3 operator+(const float3& v1, const float3& v2) {
	return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__device__ float2 operator-(const float2& v1, const float2& v2) {
	return make_float2(v1.x - v2.x, v1.y - v2.y);
}

__device__ __forceinline__ float dot(const float3& v1, const float3& v2) {
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__device__ __forceinline__ void normalize3(float3 & v) {
	float rmod = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
	v = rmod * v;
	return;
}

__device__ __forceinline__ void normalize2(float2& v) {
	float rmod = rsqrtf(v.x * v.x + v.y * v.y);
	v = rmod * v;
	return;
}

template <typename scalar_t>
__global__ void SSR_cuda_kernel(scalar_t* image, const scalar_t* colors, const scalar_t* depth, const scalar_t* normals, const int* faces_ind, const int* ssr_faces, int is, int ssr_obj, float width, int step) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	int np = is * is;
	if (i >= np) {
		return;
	}

	int ssr = 0;
	int face_ind = faces_ind[i];
	for (int j = 0; j < ssr_obj; j++) {
		if (face_ind >= ssr_faces[2 * j] && face_ind < ssr_faces[2 * j + 1]) {
			ssr = 1;
			break;
		}
	}

	if (!ssr) {
		image[i * 3 + 0] = colors[i * 3 + 0];
		image[i * 3 + 1] = colors[i * 3 + 1];
		image[i * 3 + 2] = colors[i * 3 + 2];
		return;
	}


	int xp = i % is;
	int yp = i / is;
	scalar_t xi = (2 * scalar_t(xp) + 1 - is) / is;
	scalar_t yi = (2 * (is - 1 - scalar_t(yp)) + 1 - is) / is;

	float2 wcoord_p = make_float2(xi, yi);
	float3 wcoord = make_float3(xi * depth[i] * width, yi * depth[i] * width, depth[i]);
	float3 indir(wcoord);
	float3 normal = make_float3(normals[i * 3 + 0], normals[i * 3 + 1], normals[i * 3 + 2]);

	normalize3(indir);
	float3 outdir;
	outdir = indir + 2 * (-dot(indir, normal)) * normal;        ///define dot
	normalize3(outdir);

	scalar_t world_thickness = 0.05;
	float3 wcoord0 = wcoord + outdir;
	float2 wcoord0_p = make_float2(wcoord0.x / wcoord0.z / width, wcoord0.y / wcoord0.z / width);

	float2 stepDir = wcoord0_p - wcoord_p;
	normalize2(stepDir);                    ///define normalize

	scalar_t ray_x = xp;
	scalar_t ray_y = is - 1 - yp;
	scalar_t ray_k;
	scalar_t ray_depth = wcoord.z;
	scalar_t step_x;
	scalar_t step_y;
	scalar_t k_axis;

	if (abs(stepDir.y) > abs(stepDir.x)) {
		if (stepDir.y > 0) {
			stepDir.y = max(stepDir.y, 1e-5);
			step_y = step;
			step_x = step * stepDir.x / stepDir.y;
		}
		else {
			stepDir.y = min(stepDir.y, -1e-5);
			step_y = -step;
			step_x = -step * stepDir.x / stepDir.y;
		}
	}
	else {
		if (stepDir.x > 0) {
			stepDir.x = max(stepDir.x, 1e-5);
			step_x = step;
			step_y = step * stepDir.y / stepDir.x;
		}
		else {
			stepDir.x = min(stepDir.x, -1e-5);
			step_x = -step;
			step_y = -step * stepDir.y / stepDir.x;
		}
	}

	if (abs(outdir.y) > abs(outdir.x)) {
		outdir.y = outdir.y > 0 ? max(outdir.y, 1e-5) : min(outdir.y, -1e-5);
		ray_k = outdir.x / outdir.y;
		k_axis = 1;
	}
	else {
		outdir.x = outdir.x > 0 ? max(outdir.x, 1e-5) : min(outdir.x, -1e-5);
		ray_k = outdir.y / outdir.x;
		k_axis = 0;
	}


	while (1) {
		ray_x += step_x;
		ray_y += step_y;
		int ind = (is - int(ray_y) - 1) * is + int(ray_x);
		if (ind >= np * 3 || ind < 0) {
			image[i * 3 + 0] = colors[i * 3 + 0];
			image[i * 3 + 1] = colors[i * 3 + 1];
			image[i * 3 + 2] = colors[i * 3 + 2];
			return;
		}
		scalar_t z = depth[ind];
		scalar_t ray_xi = (2 * ray_x + 1 - is) / is;
		scalar_t ray_yi = (2 * ray_y + 1 - is) / is;
		scalar_t numerator;
		scalar_t denominator;
		if (k_axis) {
			numerator = yi * ray_k - xi;
			denominator = ray_yi * ray_k - ray_xi;
		}
		else {
			numerator = xi * ray_k - yi;
			denominator = ray_xi * ray_k - ray_yi;
		}
		denominator = denominator > 0 ? max(denominator, 1e-5) : min(denominator, -1e-5);
		ray_depth = numerator / denominator * wcoord.z;

		/*
		if (i>940*1024+400 && i<940*1024+405){
			image[ind - 2] = (depth-2)/2;
			image[ind - 1] = (ray_depth-2)/2;
			image[ind - 0] = abs(ray_k);
		}
		*/

		if (ray_depth > z) {
            image[i * 3 + 0] = 0.6 * colors[i * 3 + 0] + 0.4 * colors[ind * 3 + 0];
            image[i * 3 + 1] = 0.6 * colors[i * 3 + 1] + 0.4 * colors[ind * 3 + 1];
            image[i * 3 + 2] = 0.6 * colors[i * 3 + 2] + 0.4 * colors[ind * 3 + 2];
			return;
		}
	}
}
        ''',
                   cuda_src=f'''
        @alias(image,out0)
        @alias(color,in0)
        @alias(depth,in1)
        @alias(normal_buffer,in2)
        @alias(faces_ind_buffer,in3)
        @alias(ssr_faces,in4)
        const auto is = color_shape0;
        const auto ssr_obj = ssr_faces_shape0/2;
        const int threads = 1024;
        const dim3 blocks = ( (is * is - 1) / threads + 1);
        
        SSR_cuda_kernel<float32><<<blocks,threads>>>(
            image_p,
            color_p,
            depth_p,
            normal_buffer_p,
            faces_ind_buffer_p,
            ssr_faces_p,
            is,
            ssr_obj,
            {width},
            {step}
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error in SSR_cuda: %s\\n", cudaGetErrorString(err));

        '''
                   )

def _SSR_cuda(color, world_buffer, normal_buffer, faces_ind_buffer, ssr_faces, width, step, level_intersect = 0):
    Hi_z = []
    depth = world_buffer[:,:,2]
    Hi_z.append(depth)
    nlevel = 1
    size = depth.shape[0]
    # need resize
    assert math.log(size,2) == round(math.log(size,2))
    while depth.shape[0] != 1:
        depth_pool = nn.pool(depth.unsqueeze(0).unsqueeze(0),2,"min",stride=2).squeeze(0).squeeze(0)
        Hi_z.append(depth_pool) 
        depth = depth_pool
        nlevel = nlevel + 1
        
    image = jt.zeros_like(color)
    Hi_depth = jt.array([]).float32()
    level_np = []
    for im in Hi_z:
        im_size = im.numel()
        level_np.append(im_size)
        im = im.reshape((1,im_size))
        if Hi_depth.numel() == 0:
            Hi_depth = jt.concat([Hi_depth,im],dim = 0)
        else:
            Hi_depth = jt.concat([Hi_depth,im],dim = 1)
    level_np = jt.array(level_np).int32()
    #Hi_depth = world_buffer[:,:,2]
    return jt.code(image.shape, image.dtype, [color, Hi_depth, normal_buffer, faces_ind_buffer, ssr_faces, level_np],
                   cuda_header='''
#include <cuda.h>
#include <cuda_runtime.h>

__device__ float3 operator*(float a, const float3& v) {
	return make_float3(v.x * a, v.y * a, v.z * a);
}

__device__ float2 operator*(float a, const float2& v) {
	return make_float2(v.x * a, v.y * a);
}

__device__ float3 operator+(const float3& v1, const float3& v2) {
	return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__device__ float2 operator-(const float2& v1, const float2& v2) {
	return make_float2(v1.x - v2.x, v1.y - v2.y);
}

__device__ __forceinline__ float dot(const float3& v1, const float3& v2) {
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__device__ __forceinline__ void normalize3(float3 & v) {
	float rmod = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
	v = rmod * v;
	return;
}

__device__ __forceinline__ void normalize2(float2& v) {
	float rmod = rsqrtf(v.x * v.x + v.y * v.y);
	v = rmod * v;
	return;
}

template <typename scalar_t>
__global__ void SSR_cuda_kernel(
	scalar_t* image,
	scalar_t* colors,
	const scalar_t* depth,
	const scalar_t* normals,
	const int* faces_ind,
	const int* ssr_faces,
	int is,
	int ssr_obj,
	float width,
	int step,
	int* level_ind,
	int level_intersect)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	int np = is * is;
	if (i >= np) {
		return;
	}

	int ssr = 0;
	int face_ind = faces_ind[i];
	for (int j = 0; j < ssr_obj; j++) {
		if (face_ind >= ssr_faces[2 * j] && face_ind < ssr_faces[2 * j + 1]) {
			ssr = 1;
			break;
		}
	}

	if (!ssr) {
		image[i * 3 + 0] = colors[i * 3 + 0];
		image[i * 3 + 1] = colors[i * 3 + 1];
		image[i * 3 + 2] = colors[i * 3 + 2];
		return;
	}

	int xp = i % is;
	int yp = i / is;
	scalar_t xi = (2 * scalar_t(xp) + 1 - is) / is;
	scalar_t yi = (2 * (is - 1 - scalar_t(yp)) + 1 - is) / is;

	float2 wcoord_p = make_float2(xi, yi);
	float3 wcoord = make_float3(xi * depth[i] * width, yi * depth[i] * width, depth[i]);
	float3 indir(wcoord);
	float3 normal = make_float3(normals[i * 3 + 0], normals[i * 3 + 1], normals[i * 3 + 2]);

	normalize3(indir);
	float3 outdir;
	outdir = indir + 2 * (-dot(indir, normal)) * normal;        ///define dot
	normalize3(outdir);

	scalar_t world_thickness = 0.05;
	float3 wcoord0 = wcoord + outdir;
	float2 wcoord0_p = make_float2(wcoord0.x / wcoord0.z / width, wcoord0.y / wcoord0.z / width);

	float2 stepDir = wcoord0_p - wcoord_p;
	normalize2(stepDir);                    ///define normalize

	scalar_t ray_x = xp;
	scalar_t ray_y = is - 1 - yp;
	scalar_t ray_k;
	scalar_t ray_depth = wcoord.z;
	scalar_t step_x;
	scalar_t step_y;
	scalar_t k_axis;

	if (abs(stepDir.y) > abs(stepDir.x)) {
		if (stepDir.y > 0) {
			stepDir.y = max(stepDir.y, 1e-5);
			step_y = step;
			step_x = step * stepDir.x / stepDir.y;
		}
		else {
			stepDir.y = min(stepDir.y, -1e-5);
			step_y = -step;
			step_x = -step * stepDir.x / stepDir.y;
		}
	}
	else {
		if (stepDir.x > 0) {
			stepDir.x = max(stepDir.x, 1e-5);
			step_x = step;
			step_y = step * stepDir.y / stepDir.x;
		}
		else {
			stepDir.x = min(stepDir.x, -1e-5);
			step_x = -step;
			step_y = -step * stepDir.y / stepDir.x;
		}
	}

	if (abs(outdir.y) > abs(outdir.x)) {
		outdir.y = outdir.y > 0 ? max(outdir.y, 1e-5) : min(outdir.y, -1e-5);
		ray_k = outdir.x / outdir.y;
		k_axis = 1;
	}
	else {
		outdir.x = outdir.x > 0 ? max(outdir.x, 1e-5) : min(outdir.x, -1e-5);
		ray_k = outdir.y / outdir.x;
		k_axis = 0;
	}

	const scalar_t* depth0 = &depth[0];
	int depth_size = is;
	int level = 0;

	while (1) {
		ray_x += step_x;
		ray_y += step_y;

		int ind = (depth_size - int(ray_y / is * depth_size) - 1) * depth_size + int(ray_x / is * depth_size);
		if (ind >= level_ind[level] || ind < 0) { 
			if (level == 0) {
                
				image[i * 3 + 0] = colors[i * 3 + 0];
				image[i * 3 + 1] = colors[i * 3 + 1];
				image[i * 3 + 2] = colors[i * 3 + 2];
				return;
			}
			ray_x -= step_x;
			ray_y -= step_y;
			level--;
			depth0 -= level_ind[level];
			depth_size *= 2;
			step_x /= 2;
			step_y /= 2;
            continue;
		}
		scalar_t z = depth0[ind];
		scalar_t ray_xi = (2 * ray_x + 1 - is) / is;
		scalar_t ray_yi = (2 * ray_y + 1 - is) / is;
		scalar_t numerator;
		scalar_t denominator;
		if (k_axis) {
			numerator = yi * ray_k - xi;
			denominator = ray_yi * ray_k - ray_xi;
		}
		else {
			numerator = xi * ray_k - yi;
			denominator = ray_xi * ray_k - ray_yi;
		}
		denominator = denominator > 0 ? max(denominator, 1e-5) : min(denominator, -1e-5);
		ray_depth = numerator / denominator * wcoord.z;		

		if (ray_depth > z) {
			if (level <= level_intersect) {
                int c_ind = ((is - int(ray_y) - 1) * is + int(ray_x)) * 3;
                
				image[i * 3 + 0] = 0.6 * colors[i * 3 + 0] + 0.4 * colors[c_ind + 0];
				image[i * 3 + 1] = 0.6 * colors[i * 3 + 0] + 0.4 * colors[c_ind + 1];
				image[i * 3 + 2] = 0.6 * colors[i * 3 + 0] + 0.4 * colors[c_ind + 2];
                
				return; 
			}

			ray_x -= step_x;
			ray_y -= step_y;
			level--;
			depth0 -= level_ind[level];
			depth_size *= 2;
			step_x /= 2;
			step_y /= 2;
            continue;
		}

		if (depth_size != 1) {
			depth0 += level_ind[level];
			depth_size /= 2;
			step_x *= 2;
			step_y *= 2;
			level++;
		}

	}
}
        ''',
                   cuda_src=f'''
        @alias(image,out0)
        @alias(color,in0)
        @alias(Hi_depth,in1)
        @alias(normal_buffer,in2)
        @alias(faces_ind_buffer,in3)
        @alias(ssr_faces,in4)
        @alias(level_np,in5)
        const auto is = color_shape0;
        const auto ssr_obj = ssr_faces_shape0/2;
        const int threads = 1024;
        const dim3 blocks = ( (is * is - 1) / threads + 1);
        
        SSR_cuda_kernel<float32><<<blocks,threads>>>(
            image_p,
            color_p,
            Hi_depth_p,
            normal_buffer_p,
            faces_ind_buffer_p,
            ssr_faces_p,
            is,
            ssr_obj,
            {width},
            {step},
            level_np_p,
            {level_intersect}
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error in SSR_cuda: %s\\n", cudaGetErrorString(err));

        '''
                   )