<<<<<<< HEAD
import math
import jittor as jt
from jittor import nn
from skimage.io import imsave


def SSR_cuda_naive2(color, world_buffer, normal_buffer, faces_ind_buffer, ssr_faces, width, far, step = 1):        
    image = jt.zeros_like(color)
    depth = world_buffer[:,:,2]
    return jt.code(image.shape, image.dtype, [color, depth, normal_buffer, faces_ind_buffer, ssr_faces],
                   cuda_header='''
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

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

__device__ __forceinline__ void draw_point(float* image,int x, int y, int l, int is, float3 color){
    int np = is * is;
    for(int xx = x-l;xx<x+l+1;xx++){
        for(int yy = y-l;yy<y+l+1;yy++){
            int ind = xx + yy * is;
            if(ind < np && ind >= 0){
                image[ind * 3 + 0] = color.x;
                image[ind * 3 + 1] = color.y;
                image[ind * 3 + 2] = color.z;
            }
        }
    }
    return;
}

template <typename scalar_t>
__global__ void SSR_cuda_kernel(scalar_t* image, 
                        const scalar_t* colors, 
                        const scalar_t* depth, 
                        const scalar_t* normals, 
                        const int* faces_ind, 
                        const int* ssr_faces, 
                        int is, int ssr_obj, 
                        float width, 
                        int step, 
                        scalar_t far) 
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
	outdir = indir + 2 * (-dot(indir, normal)) * normal;        
	normalize3(outdir);

	scalar_t world_thickness = 0.1;
    scalar_t pixel_bias = 0;
	float3 wcoord0 = wcoord + world_thickness * outdir;
	float2 wcoord0_p = make_float2(wcoord0.x / wcoord0.z / width, wcoord0.y / wcoord0.z / width);

	float2 stepDir = wcoord0_p - wcoord_p;
	normalize2(stepDir);                    ///define normalize

	double ray_x = xp;
	double ray_y = is - 1 - yp;
	scalar_t ray_k;
	scalar_t ray_depth = wcoord.z;
	scalar_t step_x;
	scalar_t step_y;
	scalar_t k_axis;
    scalar_t step_k;
    scalar_t step_axis;

	if (abs(stepDir.y) > abs(stepDir.x)) {
		if (stepDir.y > 0) {
			//stepDir.y = max(stepDir.y, 1e-6);
			step_y = step;
			step_x = step * stepDir.x / stepDir.y;
		}
		else {
			//stepDir.y = min(stepDir.y, -1e-6);
			step_y = -step;
			step_x = -step * stepDir.x / stepDir.y;
		}
        step_axis = 1;
        step_k = stepDir.x / stepDir.y;
	}
	else {
		if (stepDir.x > 0) {
			//stepDir.x = max(stepDir.x, 1e-6);
			step_x = step;
			step_y = step * stepDir.y / stepDir.x;
		}
		else {
			//stepDir.x = min(stepDir.x, -1e-6);
			step_x = -step;
			step_y = -step * stepDir.y / stepDir.x;
		}
        step_axis = 0;
        step_k = stepDir.y / stepDir.x;
	}

	if (abs(outdir.y) > abs(outdir.x)) {
		outdir.y = outdir.y > 0 ? max(outdir.y, 1e-6) : min(outdir.y, -1e-6);
		ray_k = outdir.x / outdir.y;
		k_axis = 1;
	}
	else {
		outdir.x = outdir.x > 0 ? max(outdir.x, 1e-6) : min(outdir.x, -1e-6);
		ray_k = outdir.y / outdir.x;
		k_axis = 0;
	}

    double ray_xi,ray_yi;
    double next_ray_y, next_ray_x;
    int ray_xp, ray_yp;
    scalar_t numerator = k_axis ? (yi * ray_k - xi) : (xi * ray_k - yi);
    scalar_t denominator;
    numerator *= wcoord.z;
    int ind;
    int iter = 0;
    scalar_t z = far + 1;
    
    //if (i == 1510 * 2048 + 1024){
    if (1){ 
	while (1) {
        iter++;
        if (ray_depth > far){
            return;
        }

        if (!step_axis) {
            ray_xp = step_x > 0 ? floor((ray_x + 1) / step) : floor((ray_x - 1) / step);
            next_ray_x = step_x > 0 ? (ray_xp + 1) * step - 0.5 : ray_xp * step + 0.5;
            next_ray_y = (next_ray_x - ray_x) * step_k + ray_y;
            ray_yp = floor(next_ray_y / step);
        }
        else {
            ray_yp = step_y > 0 ? floor((ray_y + 1) / step) : floor((ray_y - 1) / step);
            next_ray_y = step_y > 0 ? (ray_yp + 1) * step - 0.5 : ray_yp * step + 0.5;
            next_ray_x = (next_ray_y - ray_y) * step_k + ray_x;
            ray_xp = floor(next_ray_x / step);
        }
        ray_x = next_ray_x;
        ray_y = next_ray_y;

		if (next_ray_y > is - 1|| next_ray_y < 0 || next_ray_x > is - 1|| next_ray_x < 0) {
			return;
		}

        ind = (is - int(next_ray_y) - 1) * is + int(next_ray_x);
		z = depth[ind];
		ray_xi = (2 * next_ray_x + 1 - is) / is;
		ray_yi = (2 * next_ray_y + 1 - is) / is;
        denominator = k_axis ? ray_yi * ray_k - ray_xi : ray_xi * ray_k - ray_yi;
		denominator = denominator > 0 ? max(denominator, 1e-5) : min(denominator, -1e-5);
		ray_depth = numerator / denominator;
        //if (iter == 1)
        //printf("step_k:%f ray_k:%f ray_xi:%f ray_yi:%f numerator:%f denominator:%f ray_depth:%f z:%f\\n",step_k,ray_k,ray_xi,ray_yi,numerator,denominator,ray_depth,z);
        //draw_point((float*)image, round(ray_x), is - round(ray_y), 3, is, make_float3(ray_depth,z,0.));

		if (ray_depth > z + pixel_bias && ray_depth <= far && ray_depth < z + world_thickness) {
            image[i * 3 + 0] = colors[i * 3 + 0] + 0.6 * colors[ind * 3 + 0];
            image[i * 3 + 1] = colors[i * 3 + 0] + 0.6 * colors[ind * 3 + 1];
            image[i * 3 + 2] = colors[i * 3 + 0] + 0.6 * colors[ind * 3 + 2];
			return;
		}
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
        const int threads = 512;
        const dim3 blocks = ( (is * is - 1) / threads + 1);
        cudaMemcpy(image_p, color_p, 12 * is * is, cudaMemcpyDeviceToDevice);
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
            {step},
            {far}
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error in SSR_cuda: %s\\n", cudaGetErrorString(err));

        '''
                   )


def SSR_cuda(color, world_buffer, normal_buffer, faces_ind_buffer, ssr_faces, width, far, step = 1, level_intersect = 0):
    Hi_z = []
    depth = world_buffer[:,:,2]
    Hi_z.append(depth)
    size = depth.shape[0]
    # need resize 
    assert math.log(size,2) == round(math.log(size,2))
    while depth.shape[0] != 1:
        depth_pool = nn.pool(depth.unsqueeze(0).unsqueeze(0),2,"min",stride=2).squeeze(0).squeeze(0)
        #imsave("D:\Render\jrender\data\\results\\temp\\depth.jpg", depth_pool)
        Hi_z.append(depth_pool) 
        depth = depth_pool
        
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
    maxLevel = level_np.shape[0]
    interval = jt.pow(jt.ones((1,maxLevel)) * 2,jt.arange(0,maxLevel))
    #Hi_depth = world_buffer[:,:,2]
    return jt.code(image.shape, image.dtype, [color, Hi_depth, normal_buffer, faces_ind_buffer, ssr_faces, level_np, interval],
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

__device__ __forceinline__ void draw_point(float* image,int x, int y, int l, int is, float3 color){
    int np = is * is;
    for(int xx = x-l;xx<x+l+1;xx++){
        for(int yy = y-l;yy<y+l+1;yy++){
            int ind = xx + yy * is;
            if(ind < np && ind >= 0){
                image[ind * 3 + 0] = color.x;
                image[ind * 3 + 1] = color.y;
                image[ind * 3 + 2] = color.z;
            }
        }
    }
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
	const int* level_ind,
    scalar_t far,
	int level_intersect,
    const scalar_t* intervals)
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

	scalar_t world_thickness = 0.1;
    scalar_t pixel_bias = 0.04;
	float3 wcoord0 = wcoord + world_thickness * outdir;
	float2 wcoord0_p = make_float2(wcoord0.x / wcoord0.z / width, wcoord0.y / wcoord0.z / width);

	float2 stepDir = wcoord0_p - wcoord_p;
	normalize2(stepDir);                    ///define normalize

	double ray_x = xp;
	double ray_y = is - 1 - yp;
	scalar_t ray_k;
	scalar_t ray_depth = wcoord.z;
	scalar_t step_x;
	scalar_t step_y;
	scalar_t k_axis;
    scalar_t step_axis;
    scalar_t step_k;
    
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
        step_k = stepDir.x / stepDir.y;
        step_axis = 1;  // along to axis_y
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
        step_k = stepDir.y / stepDir.x;
        step_axis = 0;
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
    double next_ray_x;
    double next_ray_y; 
    scalar_t z = far + 1;
    int next_hz_xp, next_hz_yp;
    scalar_t numerator = k_axis ? (yi * ray_k - xi) : (xi * ray_k - yi);
    numerator *= wcoord.z;
    scalar_t denominator;
    scalar_t ray_xi,ray_yi;
    int ind;
    float interval;

    //if (i == 2048 * 1510 + 1024){
    //if (i >= 2048 * 1700 + 950 && i <= 2048 * 1700 + 1100){
    if (1){
    int iter = 0;
	while (1) {
        iter++;
        interval = intervals[level];
        if (ray_depth > far || ray_depth < 0) {
            break;
        }
        //marching
        if (!step_axis){
            next_hz_xp = step_x > 0 ? floor((ray_x + 1) / interval) : floor((ray_x - 1) / interval);
            next_ray_x = step_x > 0 ? (next_hz_xp + 1) * interval - 0.5 : next_hz_xp * interval + 0.5;
            next_ray_y = (next_ray_x - ray_x) * step_k + ray_y;
            next_hz_yp = floor(next_ray_y / interval);
        } else {
            next_hz_yp = step_y > 0 ? floor((ray_y + 1) / interval) : floor((ray_y - 1) / interval);
            next_ray_y = step_y > 0 ? (next_hz_yp + 1) * interval - 0.5 : next_hz_yp * interval + 0.5;
            next_ray_x = (next_ray_y - ray_y) * step_k + ray_x;
            next_hz_xp = floor(next_ray_x / interval);
        }
        ind = (depth_size - next_hz_yp - 1) * depth_size + next_hz_xp;
        //draw_point((float*)image,int(next_ray_x),is - int(next_ray_y) - 1,5,is,make_float3(4,4,4.));

		if (next_hz_xp < 0 || next_hz_xp >= depth_size || next_hz_yp < 0 || next_hz_yp >= depth_size) { 
			if (level <= level_intersect) {
                return;
			}
			level--;
			depth0 -= level_ind[level];
			depth_size *= 2;
            continue;
		} 
        else {
            z = depth0[ind];
            // normalize to [-1 , 1]
            ray_xi = (2 * next_ray_x + 1 - is) / is;
            ray_yi = (2 * next_ray_y + 1 - is) / is;
            denominator = k_axis? ray_yi * ray_k - ray_xi : ray_xi * ray_k - ray_yi;
            denominator = denominator > 0 ? max(denominator, 1e-5) : min(denominator, -1e-5);
            ray_depth = numerator / denominator;		
        }

        //if(iter == 1)
        //printf("step_k:%f ray_k:%f ray_xi:%f ray_yi:%f numerator:%f denominator:%f ray_depth:%f z:%f\\n",step_k,ray_k,ray_xi,ray_yi,numerator,denominator,ray_depth,z);

		if ( ray_depth > z + pixel_bias) {
			if (level <= level_intersect && ray_depth <= far) {
                if (ray_depth <= z + world_thickness){
                    //draw_point((float*)image,int(next_ray_x),is - int(next_ray_y) - 1,5,is,make_float3(4,0.,0.));
                    int c_ind = ((is - int(next_ray_y) - 1) * is + int(next_ray_x)) * 3;
                    image[i * 3 + 0] = 0.6 * colors[i * 3 + 0] + 0.4 * colors[c_ind + 0];
                    image[i * 3 + 1] = 0.6 * colors[i * 3 + 0] + 0.4 * colors[c_ind + 1];
                    image[i * 3 + 2] = 0.6 * colors[i * 3 + 0] + 0.4 * colors[c_ind + 2];
                
                }
                return;
			} 
            else {
                level--;
                depth0 -= level_ind[level];
                depth_size *= 2;
                continue;
            }
		}

		if (depth_size != 1) {
			depth0 += level_ind[level];
			depth_size /= 2;
			level++;
            ray_x = next_ray_x;
            ray_y = next_ray_y;
		}
	}
    }
    
}

template <typename scalar_t>
__global__ void draw_grid(scalar_t * image, int is, int interval){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    int np = is * is;
    if (i >= np){
        return;
    }
    int xp = i % is;
    int yp = i / is;
    if(xp % interval == 0 || yp % interval == 0){
        draw_point((float*)image,xp,yp,1,is,make_float3(0.,1.,0.));
    }
    return;
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
        @alias(interval,in6)
        const auto is = color_shape0;
        const auto ssr_obj = ssr_faces_shape0/2;
        const int threads = 512;
        const dim3 blocks = ( (is * is - 1) / threads + 1);
            
        cudaMemcpy(image_p, color_p, 12 * is * is, cudaMemcpyDeviceToDevice);
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
            {far},
            {level_intersect},
            interval_p
        );
        cudaDeviceSynchronize();
        //draw_grid<float32><<<blocks,threads>>>(image_p,is,256);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error in SSR_cuda: %s\\n", cudaGetErrorString(err));

        '''
                   )


def SSR_cuda_naive(color, world_buffer, normal_buffer, faces_ind_buffer, ssr_faces, width, far, step = 1):        
    image = jt.zeros_like(color)
    depth = world_buffer[:,:,2]
    return jt.code(image.shape, image.dtype, [color, depth, normal_buffer, faces_ind_buffer, ssr_faces],
                   cuda_header='''
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

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

__device__ __forceinline__ void draw_point(float* image,int x, int y, int l, int is, float3 color){
    int np = is * is;
    for(int xx = x-l;xx<x+l+1;xx++){
        for(int yy = y-l;yy<y+l+1;yy++){
            int ind = xx + yy * is;
            if(ind < np && ind >= 0){
                image[ind * 3 + 0] = color.x;
                image[ind * 3 + 1] = color.y;
                image[ind * 3 + 2] = color.z;
            }
        }
    }
    return;
}

__device__ __forceinline__ float tex_sample(const float* texture, double xp, double yp, const int is) {
    xp = min(max(xp,0.),double(is - 1));
    yp = min(max(yp,0.),double(is - 1));
    yp = is - 1 - yp;
	double weight_x1 = xp - int(xp);
	double weight_x2 = 1 - weight_x1;
	double weight_y1 = yp - int(yp);
	double weight_y2 = 1 - weight_y1;
	float c = 0;
    c += texture[int(xp) + int(yp) * is] * weight_x2 * weight_y2;
    c += texture[int(xp) + (int(yp) + 1) * is] * weight_x2 * weight_y1;
    c += texture[int(xp) + 1 + int(yp) * is] * weight_x1 * weight_y2;
    c += texture[int(xp) + 1 + (int(yp) + 1) * is] * weight_x1 * weight_y1;
    return c;
}

__device__ __forceinline__ float3& tex_sample3(const float* texture, double xp, double yp, const int is) {
    xp = min(max(xp,0.),float(is - 1));
    yp = min(max(yp,0.),float(is - 1));
    yp = is - 1 - yp;
	double weight_x1 = xp - int(xp);
	double weight_x2 = 1 - weight_x1;
	double weight_y1 = yp - int(yp);
	double weight_y2 = 1 - weight_y1;
	float c[3] = {0.,0.,0.};
	for (int k = 0; k < 3; k++) {
		c[k] += texture[(int(xp) + int(yp) * is) * 3 + k] * weight_x2 * weight_y2;
		c[k] += texture[(int(xp) + (int(yp) + 1) * is) * 3 + k] * weight_x2 * weight_y1;
		c[k] += texture[(int(xp) + 1 + int(yp) * is) * 3 + k] * weight_x1 * weight_y2;
		c[k] += texture[(int(xp) + 1 + (int(yp) + 1) * is) * 3 + k] * weight_x1 * weight_y1;
	}
	return make_float3(c[0],c[1],c[2]);
}

template <typename scalar_t>
__global__ void SSR_cuda_kernel(scalar_t* image, 
                        const scalar_t* colors, 
                        const scalar_t* depth, 
                        const scalar_t* normals, 
                        const int* faces_ind, 
                        const int* ssr_faces, 
                        int is, int ssr_obj, 
                        float width, 
                        int step, 
                        scalar_t far) 
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
	outdir = indir + 2 * (-dot(indir, normal)) * normal;        
	normalize3(outdir);

	scalar_t world_thickness = 0.03;
    scalar_t pixel_bias = 0;
	float3 wcoord0 = wcoord + world_thickness * outdir;
	float2 wcoord0_p = make_float2(wcoord0.x / wcoord0.z / width, wcoord0.y / wcoord0.z / width);

	float2 stepDir = wcoord0_p - wcoord_p;
	normalize2(stepDir);                    ///define normalize

	double ray_x = xp;
	double ray_y = is - 1 - yp;
	double ray_k;
	scalar_t ray_depth = wcoord.z;
	double step_x;
	double step_y;
	scalar_t k_axis;
    double step_k;
    scalar_t step_axis;

	if (abs(stepDir.y) > abs(stepDir.x)) {
		if (stepDir.y > 0) {
			//stepDir.y = max(stepDir.y, 1e-6);
			step_y = step;
			step_x = step * stepDir.x / stepDir.y;
		}
		else {
			//stepDir.y = min(stepDir.y, -1e-6);
			step_y = -step;
			step_x = -step * stepDir.x / stepDir.y;
		}
        step_axis = 1;
        step_k = stepDir.x / stepDir.y;
	}
	else {
		if (stepDir.x > 0) {
			//stepDir.x = max(stepDir.x, 1e-6);
			step_x = step;
			step_y = step * stepDir.y / stepDir.x;
		}
		else {
			//stepDir.x = min(stepDir.x, -1e-6);
			step_x = -step;
			step_y = -step * stepDir.y / stepDir.x;
		}
        step_axis = 0;
        step_k = stepDir.y / stepDir.x;
	}

	if (abs(outdir.y) > abs(outdir.x)) {
		outdir.y = outdir.y > 0 ? max(outdir.y, 1e-6) : min(outdir.y, -1e-6);
		ray_k = outdir.x / outdir.y;
		k_axis = 1;
	}
	else {
		outdir.x = outdir.x > 0 ? max(outdir.x, 1e-6) : min(outdir.x, -1e-6);
		ray_k = outdir.y / outdir.x;
		k_axis = 0;
	}

    double ray_xi,ray_yi;
    int ray_xp, ray_yp;
    double numerator = k_axis ? (yi * ray_k - xi) : (xi * ray_k - yi);
    double denominator;
    numerator *= wcoord.z;
    int iter = 0;
    scalar_t z = far + 1;

    ray_x += step_x;
    ray_y += step_y;
    
    //if (i == 1300 * 2048 + 1200){
    if (1){ 
	while (1) {
        iter++;
        if (ray_depth > far){
            return;
        }
        ray_x += step_x;
		ray_y += step_y;

		if (ray_y > is - 1|| ray_y < 0 || ray_x > is - 1|| ray_x < 0) {
			return;
		}

		z = tex_sample(depth, ray_x, ray_y, is);
		ray_xi = (2 * ray_x + 1 - is) / is;
		ray_yi = (2 * ray_y + 1 - is) / is;
        denominator = k_axis ? ray_yi * ray_k - ray_xi : ray_xi * ray_k - ray_yi;
		denominator = denominator > 0 ? max(denominator, 1e-5) : min(denominator, -1e-5);
		ray_depth = numerator / denominator;
        //if (iter == 1)
        //printf("step_k:%f ray_k:%f ray_xi:%f ray_yi:%f numerator:%f denominator:%f ray_depth:%f z:%f\\n",step_k,ray_k,ray_xi,ray_yi,numerator,denominator,ray_depth,z);
        //draw_point((float*)image, round(ray_x), is - round(ray_y), 3, is, make_float3(ray_depth,z,0.));

		if (ray_depth > z + pixel_bias && ray_depth <= far && ray_depth < z + world_thickness) {
            int ind = int(ray_x) + (is - int(ray_y) - 1) * is;
            /*
            float3 reflect_color = tex_sample3(colors, ray_x, ray_y, is);
            image[i * 3 + 0] = colors[i * 3 + 0] + 0.6 * reflect_color.x;
            image[i * 3 + 1] = colors[i * 3 + 1] + 0.6 * reflect_color.y;
            image[i * 3 + 2] = colors[i * 3 + 2] + 0.6 * reflect_color.z;
            */
            image[i * 3 + 0] = colors[i * 3 + 0] + 0.6 * colors[ind * 3 + 0];
            image[i * 3 + 1] = colors[i * 3 + 1] + 0.6 * colors[ind * 3 + 1];
            image[i * 3 + 2] = colors[i * 3 + 2] + 0.6 * colors[ind * 3 + 2];
            
			return;
		}
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
        const int threads = 512;
        const dim3 blocks = ( (is * is - 1) / threads + 1);
        cudaMemcpy(image_p, color_p, 12 * is * is, cudaMemcpyDeviceToDevice);
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
            {step},
            {far}
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error in SSR_cuda: %s\\n", cudaGetErrorString(err));

        '''
=======
import math
import jittor as jt
from jittor import nn
from skimage.io import imsave

def SSR_cuda_naive2(color, world_buffer, normal_buffer, faces_ind_buffer, ssr_faces, width, far, step = 1):        
    image = jt.zeros_like(color)
    depth = world_buffer[:,:,2]
    return jt.code(image.shape, image.dtype, [color, depth, normal_buffer, faces_ind_buffer, ssr_faces],
                   cuda_header='''
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

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

__device__ __forceinline__ void draw_point(float* image,int x, int y, int l, int is, float3 color){
    int np = is * is;
    for(int xx = x-l;xx<x+l+1;xx++){
        for(int yy = y-l;yy<y+l+1;yy++){
            int ind = xx + yy * is;
            if(ind < np && ind >= 0){
                image[ind * 3 + 0] = color.x;
                image[ind * 3 + 1] = color.y;
                image[ind * 3 + 2] = color.z;
            }
        }
    }
    return;
}

template <typename scalar_t>
__global__ void SSR_cuda_kernel(scalar_t* image, 
                        const scalar_t* colors, 
                        const scalar_t* depth, 
                        const scalar_t* normals, 
                        const int* faces_ind, 
                        const int* ssr_faces, 
                        int is, int ssr_obj, 
                        float width, 
                        int step, 
                        scalar_t far) 
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
	outdir = indir + 2 * (-dot(indir, normal)) * normal;        
	normalize3(outdir);

	scalar_t world_thickness = 0.1;
    scalar_t pixel_bias = 0;
	float3 wcoord0 = wcoord + world_thickness * outdir;
	float2 wcoord0_p = make_float2(wcoord0.x / wcoord0.z / width, wcoord0.y / wcoord0.z / width);

	float2 stepDir = wcoord0_p - wcoord_p;
	normalize2(stepDir);                    ///define normalize

	double ray_x = xp;
	double ray_y = is - 1 - yp;
	scalar_t ray_k;
	scalar_t ray_depth = wcoord.z;
	scalar_t step_x;
	scalar_t step_y;
	scalar_t k_axis;
    scalar_t step_k;
    scalar_t step_axis;

	if (abs(stepDir.y) > abs(stepDir.x)) {
		if (stepDir.y > 0) {
			//stepDir.y = max(stepDir.y, 1e-6);
			step_y = step;
			step_x = step * stepDir.x / stepDir.y;
		}
		else {
			//stepDir.y = min(stepDir.y, -1e-6);
			step_y = -step;
			step_x = -step * stepDir.x / stepDir.y;
		}
        step_axis = 1;
        step_k = stepDir.x / stepDir.y;
	}
	else {
		if (stepDir.x > 0) {
			//stepDir.x = max(stepDir.x, 1e-6);
			step_x = step;
			step_y = step * stepDir.y / stepDir.x;
		}
		else {
			//stepDir.x = min(stepDir.x, -1e-6);
			step_x = -step;
			step_y = -step * stepDir.y / stepDir.x;
		}
        step_axis = 0;
        step_k = stepDir.y / stepDir.x;
	}

	if (abs(outdir.y) > abs(outdir.x)) {
		outdir.y = outdir.y > 0 ? max(outdir.y, 1e-6) : min(outdir.y, -1e-6);
		ray_k = outdir.x / outdir.y;
		k_axis = 1;
	}
	else {
		outdir.x = outdir.x > 0 ? max(outdir.x, 1e-6) : min(outdir.x, -1e-6);
		ray_k = outdir.y / outdir.x;
		k_axis = 0;
	}

    double ray_xi,ray_yi;
    double next_ray_y, next_ray_x;
    int ray_xp, ray_yp;
    scalar_t numerator = k_axis ? (yi * ray_k - xi) : (xi * ray_k - yi);
    scalar_t denominator;
    numerator *= wcoord.z;
    int ind;
    int iter = 0;
    scalar_t z = far + 1;
    
    //if (i == 1510 * 2048 + 1024){
    if (1){ 
	while (1) {
        iter++;
        if (ray_depth > far){
            return;
        }

        if (!step_axis) {
            ray_xp = step_x > 0 ? floor((ray_x + 1) / step) : floor((ray_x - 1) / step);
            next_ray_x = step_x > 0 ? (ray_xp + 1) * step - 0.5 : ray_xp * step + 0.5;
            next_ray_y = (next_ray_x - ray_x) * step_k + ray_y;
            ray_yp = floor(next_ray_y / step);
        }
        else {
            ray_yp = step_y > 0 ? floor((ray_y + 1) / step) : floor((ray_y - 1) / step);
            next_ray_y = step_y > 0 ? (ray_yp + 1) * step - 0.5 : ray_yp * step + 0.5;
            next_ray_x = (next_ray_y - ray_y) * step_k + ray_x;
            ray_xp = floor(next_ray_x / step);
        }
        ray_x = next_ray_x;
        ray_y = next_ray_y;

		if (next_ray_y > is - 1|| next_ray_y < 0 || next_ray_x > is - 1|| next_ray_x < 0) {
			return;
		}

        ind = (is - int(next_ray_y) - 1) * is + int(next_ray_x);
		z = depth[ind];
		ray_xi = (2 * next_ray_x + 1 - is) / is;
		ray_yi = (2 * next_ray_y + 1 - is) / is;
        denominator = k_axis ? ray_yi * ray_k - ray_xi : ray_xi * ray_k - ray_yi;
		denominator = denominator > 0 ? max(denominator, 1e-5) : min(denominator, -1e-5);
		ray_depth = numerator / denominator;
        //if (iter == 1)
        //printf("step_k:%f ray_k:%f ray_xi:%f ray_yi:%f numerator:%f denominator:%f ray_depth:%f z:%f\\n",step_k,ray_k,ray_xi,ray_yi,numerator,denominator,ray_depth,z);
        //draw_point((float*)image, round(ray_x), is - round(ray_y), 3, is, make_float3(ray_depth,z,0.));

		if (ray_depth > z + pixel_bias && ray_depth <= far && ray_depth < z + world_thickness) {
            image[i * 3 + 0] = 0.6 * colors[ind * 3 + 0];
            image[i * 3 + 1] = 0.6 * colors[ind * 3 + 1];
            image[i * 3 + 2] = 0.6 * colors[ind * 3 + 2];
			return;
		}
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
        const int threads = 512;
        const dim3 blocks = ( (is * is - 1) / threads + 1);
        //cudaMemcpy(image_p, color_p, 12 * is * is, cudaMemcpyDeviceToDevice);
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
            {step},
            {far}
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error in SSR_cuda: %s\\n", cudaGetErrorString(err));

        '''
                   )

def SSR_cuda(color, world_buffer, normal_buffer, faces_ind_buffer, ssr_faces, width, far, step = 1, level_intersect = 0):
    Hi_z = []
    depth = world_buffer[:,:,2]
    Hi_z.append(depth)
    size = depth.shape[0]
    # need resize 
    assert math.log(size,2) == round(math.log(size,2))
    while depth.shape[0] != 1:
        depth_pool = nn.pool(depth.unsqueeze(0).unsqueeze(0),2,"min",stride=2).squeeze(0).squeeze(0)
        #imsave("D:\Render\jrender\data\\results\\temp\\depth.jpg", depth_pool)
        Hi_z.append(depth_pool) 
        depth = depth_pool
        
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
    maxLevel = level_np.shape[0]
    interval = jt.pow(jt.ones((1,maxLevel)) * 2,jt.arange(0,maxLevel))
    #Hi_depth = world_buffer[:,:,2]
    return jt.code(image.shape, image.dtype, [color, Hi_depth, normal_buffer, faces_ind_buffer, ssr_faces, level_np, interval],
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

__device__ __forceinline__ void draw_point(float* image,int x, int y, int l, int is, float3 color){
    int np = is * is;
    for(int xx = x-l;xx<x+l+1;xx++){
        for(int yy = y-l;yy<y+l+1;yy++){
            int ind = xx + yy * is;
            if(ind < np && ind >= 0){
                image[ind * 3 + 0] = color.x;
                image[ind * 3 + 1] = color.y;
                image[ind * 3 + 2] = color.z;
            }
        }
    }
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
	const int* level_ind,
    scalar_t far,
	int level_intersect,
    const scalar_t* intervals)
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

	scalar_t world_thickness = 0.1;
    scalar_t pixel_bias = 0.04;
	float3 wcoord0 = wcoord + world_thickness * outdir;
	float2 wcoord0_p = make_float2(wcoord0.x / wcoord0.z / width, wcoord0.y / wcoord0.z / width);

	float2 stepDir = wcoord0_p - wcoord_p;
	normalize2(stepDir);                    ///define normalize

	double ray_x = xp;
	double ray_y = is - 1 - yp;
	scalar_t ray_k;
	scalar_t ray_depth = wcoord.z;
	scalar_t step_x;
	scalar_t step_y;
	scalar_t k_axis;
    scalar_t step_axis;
    scalar_t step_k;
    
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
        step_k = stepDir.x / stepDir.y;
        step_axis = 1;  // along to axis_y
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
        step_k = stepDir.y / stepDir.x;
        step_axis = 0;
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
    double next_ray_x;
    double next_ray_y; 
    scalar_t z = far + 1;
    int next_hz_xp, next_hz_yp;
    scalar_t numerator = k_axis ? (yi * ray_k - xi) : (xi * ray_k - yi);
    numerator *= wcoord.z;
    scalar_t denominator;
    scalar_t ray_xi,ray_yi;
    int ind;
    float interval;

    //if (i == 2048 * 1510 + 1024){
    //if (i >= 2048 * 1700 + 950 && i <= 2048 * 1700 + 1100){
    if (1){
    int iter = 0;
	while (1) {
        iter++;
        interval = intervals[level];
        if (ray_depth > far || ray_depth < 0) {
            break;
        }
        //marching
        if (!step_axis){
            next_hz_xp = step_x > 0 ? floor((ray_x + 1) / interval) : floor((ray_x - 1) / interval);
            next_ray_x = step_x > 0 ? (next_hz_xp + 1) * interval - 0.5 : next_hz_xp * interval + 0.5;
            next_ray_y = (next_ray_x - ray_x) * step_k + ray_y;
            next_hz_yp = floor(next_ray_y / interval);
        } else {
            next_hz_yp = step_y > 0 ? floor((ray_y + 1) / interval) : floor((ray_y - 1) / interval);
            next_ray_y = step_y > 0 ? (next_hz_yp + 1) * interval - 0.5 : next_hz_yp * interval + 0.5;
            next_ray_x = (next_ray_y - ray_y) * step_k + ray_x;
            next_hz_xp = floor(next_ray_x / interval);
        }
        ind = (depth_size - next_hz_yp - 1) * depth_size + next_hz_xp;
        //draw_point((float*)image,int(next_ray_x),is - int(next_ray_y) - 1,5,is,make_float3(4,4,4.));

		if (next_hz_xp < 0 || next_hz_xp >= depth_size || next_hz_yp < 0 || next_hz_yp >= depth_size) { 
			if (level <= level_intersect) {
                return;
			}
			level--;
			depth0 -= level_ind[level];
			depth_size *= 2;
            continue;
		} 
        else {
            z = depth0[ind];
            // normalize to [-1 , 1]
            ray_xi = (2 * next_ray_x + 1 - is) / is;
            ray_yi = (2 * next_ray_y + 1 - is) / is;
            denominator = k_axis? ray_yi * ray_k - ray_xi : ray_xi * ray_k - ray_yi;
            denominator = denominator > 0 ? max(denominator, 1e-5) : min(denominator, -1e-5);
            ray_depth = numerator / denominator;		
        }

        //if(iter == 1)
        //printf("step_k:%f ray_k:%f ray_xi:%f ray_yi:%f numerator:%f denominator:%f ray_depth:%f z:%f\\n",step_k,ray_k,ray_xi,ray_yi,numerator,denominator,ray_depth,z);

		if ( ray_depth > z + pixel_bias) {
			if (level <= level_intersect && ray_depth <= far) {
                if (ray_depth <= z + world_thickness){
                    //draw_point((float*)image,int(next_ray_x),is - int(next_ray_y) - 1,5,is,make_float3(4,0.,0.));
                    int c_ind = ((is - int(next_ray_y) - 1) * is + int(next_ray_x)) * 3;
                    image[i * 3 + 0] = 0.6 * colors[i * 3 + 0] + 0.4 * colors[c_ind + 0];
                    image[i * 3 + 1] = 0.6 * colors[i * 3 + 0] + 0.4 * colors[c_ind + 1];
                    image[i * 3 + 2] = 0.6 * colors[i * 3 + 0] + 0.4 * colors[c_ind + 2];
                
                }
                return;
			} 
            else {
                level--;
                depth0 -= level_ind[level];
                depth_size *= 2;
                continue;
            }
		}

		if (depth_size != 1) {
			depth0 += level_ind[level];
			depth_size /= 2;
			level++;
            ray_x = next_ray_x;
            ray_y = next_ray_y;
		}
	}
    }
    
}

template <typename scalar_t>
__global__ void draw_grid(scalar_t * image, int is, int interval){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    int np = is * is;
    if (i >= np){
        return;
    }
    int xp = i % is;
    int yp = i / is;
    if(xp % interval == 0 || yp % interval == 0){
        draw_point((float*)image,xp,yp,1,is,make_float3(0.,1.,0.));
    }
    return;
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
        @alias(interval,in6)
        const auto is = color_shape0;
        const auto ssr_obj = ssr_faces_shape0/2;
        const int threads = 512;
        const dim3 blocks = ( (is * is - 1) / threads + 1);
            
        cudaMemcpy(image_p, color_p, 12 * is * is, cudaMemcpyDeviceToDevice);
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
            {far},
            {level_intersect},
            interval_p
        );
        cudaDeviceSynchronize();
        //draw_grid<float32><<<blocks,threads>>>(image_p,is,256);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error in SSR_cuda: %s\\n", cudaGetErrorString(err));

        '''
                   )


def SSR_cuda_naive(color, world_buffer, normal_buffer, faces_ind_buffer, ssr_faces, width, far, step = 1):        
    image = jt.zeros_like(color)
    depth = world_buffer[:,:,2]
    return jt.code(image.shape, image.dtype, [color, depth, normal_buffer, faces_ind_buffer, ssr_faces],
                   cuda_header='''
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

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

__device__ __forceinline__ void draw_point(float* image,int x, int y, int l, int is, float3 color){
    int np = is * is;
    for(int xx = x-l;xx<x+l+1;xx++){
        for(int yy = y-l;yy<y+l+1;yy++){
            int ind = xx + yy * is;
            if(ind < np && ind >= 0){
                image[ind * 3 + 0] = color.x;
                image[ind * 3 + 1] = color.y;
                image[ind * 3 + 2] = color.z;
            }
        }
    }
    return;
}

__device__ __forceinline__ float tex_sample(const float* texture, double xp, double yp, const int is) {
    xp = min(max(xp,0.),double(is - 1));
    yp = min(max(yp,0.),double(is - 1));
    yp = is - 1 - yp;
	double weight_x1 = xp - int(xp);
	double weight_x2 = 1 - weight_x1;
	double weight_y1 = yp - int(yp);
	double weight_y2 = 1 - weight_y1;
	float c = 0;
    c += texture[int(xp) + int(yp) * is] * weight_x2 * weight_y2;
    c += texture[int(xp) + (int(yp) + 1) * is] * weight_x2 * weight_y1;
    c += texture[int(xp) + 1 + int(yp) * is] * weight_x1 * weight_y2;
    c += texture[int(xp) + 1 + (int(yp) + 1) * is] * weight_x1 * weight_y1;
    return c;
}

__device__ __forceinline__ float3& tex_sample3(const float* texture, double xp, double yp, const int is) {
    xp = min(max(xp,0.),float(is - 1));
    yp = min(max(yp,0.),float(is - 1));
    yp = is - 1 - yp;
	double weight_x1 = xp - int(xp);
	double weight_x2 = 1 - weight_x1;
	double weight_y1 = yp - int(yp);
	double weight_y2 = 1 - weight_y1;
	float c[3] = {0.,0.,0.};
	for (int k = 0; k < 3; k++) {
		c[k] += texture[(int(xp) + int(yp) * is) * 3 + k] * weight_x2 * weight_y2;
		c[k] += texture[(int(xp) + (int(yp) + 1) * is) * 3 + k] * weight_x2 * weight_y1;
		c[k] += texture[(int(xp) + 1 + int(yp) * is) * 3 + k] * weight_x1 * weight_y2;
		c[k] += texture[(int(xp) + 1 + (int(yp) + 1) * is) * 3 + k] * weight_x1 * weight_y1;
	}
	return make_float3(c[0],c[1],c[2]);
}

template <typename scalar_t>
__global__ void SSR_cuda_kernel(scalar_t* image, 
                        const scalar_t* colors, 
                        const scalar_t* depth, 
                        const scalar_t* normals, 
                        const int* faces_ind, 
                        const int* ssr_faces, 
                        int is, int ssr_obj, 
                        float width, 
                        int step, 
                        scalar_t far) 
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
	outdir = indir + 2 * (-dot(indir, normal)) * normal;        
	normalize3(outdir);

	scalar_t world_thickness = 0.03;
    scalar_t pixel_bias = 0;
	float3 wcoord0 = wcoord + world_thickness * outdir;
	float2 wcoord0_p = make_float2(wcoord0.x / wcoord0.z / width, wcoord0.y / wcoord0.z / width);

	float2 stepDir = wcoord0_p - wcoord_p;
	normalize2(stepDir);                    ///define normalize

	double ray_x = xp;
	double ray_y = is - 1 - yp;
	double ray_k;
	scalar_t ray_depth = wcoord.z;
	double step_x;
	double step_y;
	scalar_t k_axis;
    double step_k;
    scalar_t step_axis;

	if (abs(stepDir.y) > abs(stepDir.x)) {
		if (stepDir.y > 0) {
			//stepDir.y = max(stepDir.y, 1e-6);
			step_y = step;
			step_x = step * stepDir.x / stepDir.y;
		}
		else {
			//stepDir.y = min(stepDir.y, -1e-6);
			step_y = -step;
			step_x = -step * stepDir.x / stepDir.y;
		}
        step_axis = 1;
        step_k = stepDir.x / stepDir.y;
	}
	else {
		if (stepDir.x > 0) {
			//stepDir.x = max(stepDir.x, 1e-6);
			step_x = step;
			step_y = step * stepDir.y / stepDir.x;
		}
		else {
			//stepDir.x = min(stepDir.x, -1e-6);
			step_x = -step;
			step_y = -step * stepDir.y / stepDir.x;
		}
        step_axis = 0;
        step_k = stepDir.y / stepDir.x;
	}

	if (abs(outdir.y) > abs(outdir.x)) {
		outdir.y = outdir.y > 0 ? max(outdir.y, 1e-6) : min(outdir.y, -1e-6);
		ray_k = outdir.x / outdir.y;
		k_axis = 1;
	}
	else {
		outdir.x = outdir.x > 0 ? max(outdir.x, 1e-6) : min(outdir.x, -1e-6);
		ray_k = outdir.y / outdir.x;
		k_axis = 0;
	}

    double ray_xi,ray_yi;
    int ray_xp, ray_yp;
    double numerator = k_axis ? (yi * ray_k - xi) : (xi * ray_k - yi);
    double denominator;
    numerator *= wcoord.z;
    int iter = 0;
    scalar_t z = far + 1;

    ray_x += step_x;
    ray_y += step_y;
    
    if (i == 1300 * 2048 + 1200){
    //if (1){ 
	while (1) {
        iter++;
        if (ray_depth > far){
            return;
        }
        ray_x += step_x;
		ray_y += step_y;

		if (ray_y > is - 1|| ray_y < 0 || ray_x > is - 1|| ray_x < 0) {
			return;
		}

		z = tex_sample(depth, ray_x, ray_y, is);
		ray_xi = (2 * ray_x + 1 - is) / is;
		ray_yi = (2 * ray_y + 1 - is) / is;
        denominator = k_axis ? ray_yi * ray_k - ray_xi : ray_xi * ray_k - ray_yi;
		denominator = denominator > 0 ? max(denominator, 1e-5) : min(denominator, -1e-5);
		ray_depth = numerator / denominator;
        //if (iter == 1)
        printf("step_k:%f ray_k:%f ray_xi:%f ray_yi:%f numerator:%f denominator:%f ray_depth:%f z:%f\\n",step_k,ray_k,ray_xi,ray_yi,numerator,denominator,ray_depth,z);
        draw_point((float*)image, round(ray_x), is - round(ray_y), 3, is, make_float3(ray_depth,z,0.));

		if (ray_depth > z + pixel_bias && ray_depth <= far && ray_depth < z + world_thickness) {
            int ind = int(ray_x) + (is - int(ray_y) - 1) * is;
            /*
            float3 reflect_color = tex_sample3(colors, ray_x, ray_y, is);
            image[i * 3 + 0] = colors[i * 3 + 0] + 0.6 * reflect_color.x;
            image[i * 3 + 1] = colors[i * 3 + 1] + 0.6 * reflect_color.y;
            image[i * 3 + 2] = colors[i * 3 + 2] + 0.6 * reflect_color.z;
            */
            image[i * 3 + 0] = colors[i * 3 + 0] + 0.6 * colors[ind * 3 + 0];
            image[i * 3 + 1] = colors[i * 3 + 1] + 0.6 * colors[ind * 3 + 1];
            image[i * 3 + 2] = colors[i * 3 + 2] + 0.6 * colors[ind * 3 + 2];
            
			return;
		}
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
        const int threads = 512;
        const dim3 blocks = ( (is * is - 1) / threads + 1);
        cudaMemcpy(image_p, color_p, 12 * is * is, cudaMemcpyDeviceToDevice);
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
            {step},
            {far}
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error in SSR_cuda: %s\\n", cudaGetErrorString(err));

        '''
>>>>>>> 5fbc0ef13b78afa7b9945513b1b5fd5e2980a95a
                   )