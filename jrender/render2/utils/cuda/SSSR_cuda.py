import jittor as jt
from jittor import nn
import math

def SSSR_cuda(
    color, world_buffer, normal_buffer, roughness_buffer, faces_ind_buffer, 
    ssr_faces, width, far, step = 1, level_intersect = 0, spp = 64):
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
    return jt.code(image.shape, image.dtype, [color, Hi_depth, normal_buffer, roughness_buffer, faces_ind_buffer, ssr_faces, level_np, interval],
                   cuda_header='''
#include<cuda.h>
#include<cuda_runtime.h>
#include<curand_kernel.h>
#include<math.h>

#define pi 3.1415

__device__ float3& operator*(float a, const float3& v) {
    return make_float3(v.x * a, v.y * a, v.z * a);
}

__device__ float2& operator*(float a, const float2& v) {
    return make_float2(v.x * a, v.y * a);
}

__device__ float3& operator*(const float3& v1, const float3& v2) {
    return make_float3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}

__device__ float3& operator/(const float3& v, float a) {
    return make_float3(v.x / a, v.y / a, v.z / a);
}

__device__ float2& operator/(const float2& v, float a) {
    return make_float2(v.x / a, v.y / a);
}

__device__ float3& operator+(const float3& v1, const float3& v2) {
    return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__device__ float3& operator-(const float3& v1, const float3& v2) {
    return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

__device__ float3& operator+=(const float3& v1, const float3& v2) {
    return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__device__ float2& operator-(const float2& v1, const float2& v2) {
    return make_float2(v1.x - v2.x, v1.y - v2.y);
}

__device__ __forceinline__ float dot(const float3& v1, const float3& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__device__ __forceinline__ float3& cross(const float3& v1, const float3& v2) {
    return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

__device__ __forceinline__ void normalize3(float3& v) {
    float rmod = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    v = rmod * v;
    return;
}

__device__ __forceinline__ void normalize2(float2& v) {
    float rmod = rsqrtf(v.x * v.x + v.y * v.y);
    v = rmod * v;
    return;
}

__device__ __forceinline__ void draw_point(float* image, int x, int y, int l, int is, float3 color) {
    int np = is * is;
    for (int xx = x - l; xx < x + l + 1; xx++) {
        for (int yy = y - l; yy < y + l + 1; yy++) {
            int ind = xx + yy * is;
            if (ind < np && ind >= 0) {
                image[ind * 3 + 0] = color.x;
                image[ind * 3 + 1] = color.y;
                image[ind * 3 + 2] = color.z;
            }
        }
    }
    return;
}

__device__ __forceinline__ float GGX(float3 N, float3 H, float roughness) {
    float a = powf(roughness, 4);
    float NdotH = max(dot(N, H), 0.);
    float NdotH2 = NdotH * NdotH;
    float num = a;
    float denom = NdotH2 * (a - 1) + 1;
    denom = 3.1415 * denom * denom;
    return num / denom;
}

__device__ __forceinline__ float SchlickGGX(float dot, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float num = dot;
    float denom = dot * (1.0 - k) + k;

    return num / denom;
}

__device__ __forceinline__ float GeometrySmith(float NdotV, float NdotL, float roughness) {
    float ggx2 = SchlickGGX(NdotV, roughness);
    float ggx1 = SchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

__device__ __forceinline__ float3& fresnel(float cosTheta, float3 F0) {
    return F0 + powf((1 - cosTheta), 5) * make_float3(1 - F0.x, 1 - F0.y, 1 - F0.z);
}

__device__ __forceinline__ void VisibleNormal_sampler(curandStateXORWOW_t state,float3& indir, float& pdf, float3 oudir, float3 N, float roughness) {
    
    return;
}

__device__ __forceinline__ void CosineWeighted_sampler(curandStateXORWOW_t* state, float3& indir, float& pdf) {
    float ksi1 = curand_uniform(state);
    float ksi2 = curand_uniform(state);
    float sinTheta = sqrtf(ksi1);
    indir.x = sinTheta * cos(2 * pi * ksi2);
    indir.z = sinTheta * sin(2 * pi * ksi2);
    indir.y = sqrtf(1 - ksi1);  
    pdf = sqrtf(1 - ksi1) / pi;
    return;
}

__device__ __forceinline__ void Uniform_sampler(curandStateXORWOW_t* state, float3& indir, float& pdf) {
    float ksi1 = curand_uniform(state);
    float ksi2 = curand_uniform(state);
    float sinTheta = sqrtf(1 - powf((1 - ksi1), 2));
    indir.x = sinTheta * cos(2 * pi * ksi2);
    indir.z = sinTheta * sin(2 * pi * ksi2);
    indir.y = 1 - ksi1;
    pdf = 1 / (2 * pi);
    return;
}

__device__ __forceinline__ void Cone_sampler(curandStateXORWOW_t* state, float3& H, float& pdf) {
    float ksi1 = curand_uniform(state);
    float ksi2 = curand_uniform(state);
    float k = 0.15;            // k = 1- cos2Theta_max
    float sinTheta = sqrtf(ksi1 * k / 2);
    H = make_float3(sinTheta * cos(2 * pi * ksi2),sqrtf(1 - sinTheta * sinTheta),sinTheta * sin(2 * pi * ksi2));
    pdf = 2 * H.y / (pi * k);
    return;
}

__device__ __forceinline__ float3& BRDF_withcos(float3 indir, float3 outdir, float3 N, float rough) {
    float NdotV = max(dot(N, outdir), 0.);
    float NdotL = max(dot(N, indir), 0.);
    float3 H = (outdir + indir) / 2;
    normalize3(H);
    float D = GGX(N, H, rough);
    float G = GeometrySmith(NdotV, NdotL, rough);
    float cosTheta = max(dot(H, outdir), 0.);
    float3 F0 = make_float3(0.89, 0.16, 0.11);
    float3 F = fresnel(cosTheta, F0);

    return (D * G * F) / (4 * NdotV);
    //return make_float3(D,D,D);
}

template <typename scalar_t>
__global__ void SSSR_cuda_kernel(
    scalar_t* image,
    scalar_t* colors,
    const scalar_t* depth,
    const scalar_t* normals,
    const scalar_t* roughness,
    const int* faces_ind,
    const int* ssr_faces,
    int is,
    int ssr_obj,
    float width,
    int step,
    const int* level_ind,
    scalar_t far,
    int level_intersect,
    const scalar_t* intervals,
    const int spp,
    curandStateXORWOW_t* states)
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
    float3 outdir = make_float3(-wcoord.x, -wcoord.y, -wcoord.z);
    float3 color = make_float3(colors[i * 3 + 0],colors[i * 3 + 1],colors[i * 3 + 2]);
    normalize3(outdir);
    float3 normal = make_float3(normals[i * 3 + 0], normals[i * 3 + 1], normals[i * 3 + 2]);
    float rough = roughness[i];
    scalar_t world_thickness = 0.02;

    scalar_t ray_k;
    scalar_t step_x;
    scalar_t step_y;
    scalar_t k_axis;
    scalar_t step_axis;
    scalar_t step_k;
    float3 indir;
    float pdf = 1;
    float3 radiance_o = make_float3(0.,0.,0.);
    float3 radiance_i;

    float3 T, B;
    // create TBN
    if (abs(normal.x) > 0.5 || abs(normal.y) > 0.5) {
        T.y = normal.x;
        T.x = -normal.y;
        T.z = 0;
        normalize3(T);
    }
    else {
        T.x = normal.z;
        T.z = -normal.x;
        T.y = 0;
        normalize3(T);
    }
    B = cross(T,normal);
 
    for (int k = 0; k < spp; k++) {
        float3 H;
        Cone_sampler(&states[i], H, pdf);
        H = H.x * B + H.y * normal + H.z * T;
        normalize3(H);
        indir = 2 * dot(outdir, H) * H - outdir;
        pdf /= 4 * max(dot(H,outdir),0.01);   

        float3 wcoord0 = wcoord + indir;
        float2 wcoord0_p = make_float2(wcoord0.x / wcoord0.z / width, wcoord0.y / wcoord0.z / width);

        float2 stepDir = wcoord0_p - wcoord_p;
        normalize2(stepDir);                    ///define normalize

        if (abs(stepDir.y) > abs(stepDir.x)) {
            if (stepDir.y > 0) {
                stepDir.y = max(stepDir.y, 1e-6);
                step_y = step;
                step_x = step * stepDir.x / stepDir.y;
            }
            else {
                stepDir.y = min(stepDir.y, -1e-6);
                step_y = -step;
                step_x = -step * stepDir.x / stepDir.y;
            }
            step_k = stepDir.x / stepDir.y;
            step_axis = 1;  // along to axis_y
        }
        else {
            if (stepDir.x > 0) {
                stepDir.x = max(stepDir.x, 1e-6);
                step_x = step;
                step_y = step * stepDir.y / stepDir.x;
            }
            else {
                stepDir.x = min(stepDir.x, -1e-6);
                step_x = -step;
                step_y = -step * stepDir.y / stepDir.x;
            }
            step_k = stepDir.y / stepDir.x;
            step_axis = 0;
        }

        if (abs(indir.y) > abs(indir.x)) {
            indir.y = indir.y > 0 ? max(indir.y, 1e-6) : min(indir.y, -1e-6);
            ray_k = indir.x / indir.y;
            k_axis = 1;
        }
        else {
            indir.x = indir.x > 0 ? max(indir.x, 1e-6) : min(indir.x, -1e-6);
            ray_k = indir.y / indir.x;
            k_axis = 0;
        }

        scalar_t ray_x = xp;
        scalar_t ray_y = is - 1 - yp;
        scalar_t ray_depth = wcoord.z;
        const scalar_t* depth0 = &depth[0];
        int depth_size = is;
        int level = 0;
        scalar_t next_ray_x;
        scalar_t next_ray_y;
        scalar_t z = far + 1;
        int next_hz_xp, next_hz_yp;
        scalar_t numerator = k_axis ? (yi * ray_k - xi) : (xi * ray_k - yi);
        numerator *= wcoord.z;
        scalar_t denominator;
        scalar_t ray_xi, ray_yi;
        int ind;
        float interval;
        bool hit = 0;

        if(1){
        //if (i == 1250 * 2048 + 480) {
            int iter = 0;
            while (1) {
                iter++;
                interval = intervals[level];
                if (ray_depth > far || ray_depth < 0) {
                    //radiance_i = make_float3(0.,0.,0.);
                    //draw_point((float*)image,int(next_ray_x),is - int(next_ray_y) - 1,5,is,make_float3(0.,4,0.));
                    break;
                }
                //marching
                if (!step_axis) {
                    next_hz_xp = step_x > 0 ? floor((ray_x + 1) / interval) : floor((ray_x - 1) / interval);
                    next_ray_x = step_x > 0 ? (next_hz_xp + 1) * interval - 0.5 : next_hz_xp * interval + 0.5;
                    next_ray_y = (next_ray_x - ray_x) * step_k + ray_y;
                    next_hz_yp = floor(next_ray_y / interval);
                }
                else {
                    next_hz_yp = step_y > 0 ? floor((ray_y + 1) / interval) : floor((ray_y - 1) / interval);
                    next_ray_y = step_y > 0 ? (next_hz_yp + 1) * interval - 0.5 : next_hz_yp * interval + 0.5;
                    next_ray_x = (next_ray_y - ray_y) * step_k + ray_x;
                    next_hz_xp = floor(next_ray_x / interval);
                }

                if (next_hz_xp < 0 || next_hz_xp >= depth_size || next_hz_yp < 0 || next_hz_yp >= depth_size) {
                    if (level <= level_intersect) {
                        //radiance_i = make_float3(0.,0.,0.);
                        break;
                    }
                    level--;
                    depth0 -= level_ind[level];
                    depth_size *= 2;
                    continue;
                }
                //draw_point((float*)image,int(next_ray_x),is - int(next_ray_y) - 1,5,is,make_float3(4,4,4.));
                ind = (depth_size - next_hz_yp) * depth_size + next_hz_xp;
                z = depth0[ind];
                // normalize to [-1 , 1]
                ray_xi = (2 * next_ray_x + 1 - is) / is;
                ray_yi = (2 * next_ray_y + 1 - is) / is;
                denominator = k_axis ? ray_yi * ray_k - ray_xi : ray_xi * ray_k - ray_yi;
                denominator = denominator > 0 ? max(denominator, 1e-6) : min(denominator, -1e-6);
                ray_depth = numerator / denominator;
                
                if (ray_depth > z + world_thickness) {
                    if (level <= level_intersect && ray_depth <= far) {
                        int c_ind = ((is - int(next_ray_y) - 1) * is + int(next_ray_x)) * 3;
                        hit = 1;
                        radiance_i = make_float3(colors[c_ind + 0], colors[c_ind + 1], colors[c_ind + 2]);
                        //image[i * 3 + 0] = 0.5 * colors[i * 3 + 0] + 0.5 * colors[c_ind + 0];
                        //image[i * 3 + 1] = 0.5 * colors[i * 3 + 1] + 0.5 * colors[c_ind + 1];
                        //image[i * 3 + 2] = 0.5 * colors[i * 3 + 2] + 0.5 * colors[c_ind + 2];
                        //return;
                        break;
                    }
                    level--;
                    depth0 -= level_ind[level];
                    depth_size *= 2;
                    continue;
                }

                if (depth_size != 1) {
                    depth0 += level_ind[level];
                    depth_size /= 2;
                    level++;
                    ray_x = next_ray_x;
                    ray_y = next_ray_y;
                }
            }
            //draw_point((float*)image,int(next_ray_x),is - int(next_ray_y) - 1,5,is,make_float3(0.,4,0.));
            
            //radiance_o = radiance_i;
            //float3 fr = BRDF_withcos(indir, outdir, normal, rough);
            if (hit) {
                radiance_o = radiance_o + BRDF_withcos(indir, outdir, normal, rough) * radiance_i / pdf;
            }
            //image[i * 3 + 0] = radiance_o.x;
            //image[i * 3 + 1] = radiance_o.y;
            //image[i * 3 + 2] = radiance_o.z;
            //draw_point((float*)image,1000,1000,20,is,make_float3(radiance_o.x,radiance_o.y,radiance_o.z));
            //return;
        }
    }
    radiance_o = radiance_o / float(spp);
    //image[i * 3 + 0] = 0.5 * colors[i * 3 + 0] + 0.5 * radiance_o.x;
    //image[i * 3 + 1] = 0.5 * colors[i * 3 + 1] + 0.5 * radiance_o.y;
    //image[i * 3 + 2] = 0.5 * colors[i * 3 + 2] + 0.5 * radiance_o.z;
    image[i * 3 + 0] = radiance_o.x;
    image[i * 3 + 1] = radiance_o.y;
    image[i * 3 + 2] = radiance_o.z;
}

__global__ void _curand_init(curandStateXORWOW_t* state, size_t is){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= is * is){
        return;
    }
    curand_init(1,i,0,&state[i]);
    return;
}

template <typename scalar_t>
__global__ void draw_grid(scalar_t* image, int is, int interval) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    int np = is * is;
    if (i >= np) {
        return;
    }
    int xp = i % is;
    int yp = i / is;
    if (xp % interval == 0 || yp % interval == 0) {
        draw_point((float*)image, xp, yp, 1, is, make_float3(0., 1., 0.));
    }
    return;
}
        ''',
    cuda_src=f'''
    @alias(image,out0)
    @alias(color,in0)
    @alias(Hi_depth,in1)
    @alias(normal_buffer,in2)
    @alias(roughness_buffer,in3)
    @alias(faces_ind_buffer,in4)
    @alias(ssr_faces,in5)
    @alias(level_np,in6)
    @alias(interval,in7)
    const auto is = color_shape0;
    const auto ssr_obj = ssr_faces_shape0/2;
    const int threads = 512;
    const dim3 blocks = ( (is * is - 1) / threads + 1);
    const auto np = is * is;

    curandStateXORWOW_t* states;
    const auto nbytes = np * sizeof(curandStateXORWOW_t);
    cudaMalloc(&states, nbytes);
    _curand_init<<<blocks,threads>>>(states, is);
    //cudaMemcpy(image_p, color_p, 12 * is * is, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    
    SSSR_cuda_kernel<float32><<<blocks,threads>>>(
        image_p,
        color_p,
        Hi_depth_p,
        normal_buffer_p,
        roughness_buffer_p,
        faces_ind_buffer_p,
        ssr_faces_p,
        is,
        ssr_obj,
        {width},
        {step},
        level_np_p,
        {far},
        {level_intersect},
        interval_p,
        {spp},
        states
    );
    
    cudaDeviceSynchronize();
    //draw_grid<float32><<<blocks,threads>>>(image_p,is,256);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error in SSSR_cuda: %s\\n", cudaGetErrorString(err));

    '''
                   )