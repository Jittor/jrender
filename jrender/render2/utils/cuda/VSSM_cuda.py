from cmath import tan
import jittor as jt
import math

# shading [sz,sz]
# eyeDepth [sz,sz]
# DepthMap [sz,sz]
# DepthMap2 [sz,sz]
# uv [sz,sz,2]             size:sz*sz*6

def VSSM_cuda(eyeDepth, SAT, SAT2, uv, light):
    shading = jt.ones_like(eyeDepth)
    #side_length = math.sqrt(light.area)
    near = light.near
    far = light.far
    DM_sl = math.tan(light.viewing_angle / 180. * math.pi)
    side_length = 0.08
    return jt.code(shading.shape, shading.dtype, [eyeDepth, SAT, SAT2, uv],
                   cuda_header='''
#include<cuda.h>
#include<cuda_runtime.h>

__device__ __forceinline__ float Chebyshev_test(float mean, float variance, float upper_bound) {
    float a = upper_bound - mean;
    float denominator = max((a * a), 1e-5);
    float tail = min(max(variance / (variance + denominator), 0.), 1.);
    return tail;
}

template <typename scalar_t>
__device__ __forceinline__ float region_mean(const scalar_t* SAT, int x, int y, int offset ,int is){
    int xp1 = max(x - offset - 1,0);
    int yp1 = max(y - offset - 1,0);
    int xp2 = min(x + offset,is - 1);
    int yp2 = min(y + offset,is - 1);
    float sum = SAT[xp2 + yp2 * is] - SAT[xp2 + yp1 * is] - SAT[xp1 + yp2 * is] + SAT[xp1 + yp1 * is];
    return sum / float((xp2 - xp1) * (yp2 - yp1));
}

template <typename scalar_t>
__global__ void VSSM_cuda_kernel(
	scalar_t* shading,
	const scalar_t* eyeDepth,
	const double* DepthMap,
	const double* DepthMap2,
	const scalar_t* uv,
	const int is,
	const float side_length,
	const float near,
    const float far,
	const float shadow_map_sl
)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int np = is * is;
	if (i >= np) {
		return;
	}
	float u = uv[i * 2 + 0];
	float v = uv[i * 2 + 1];

    float upper = 1 - 1/is;
	if (u < 0 || u > upper || v < 0 || v > upper) {
		shading[i] = 1;
		return;
	}

	//compute the size of block_search region on the DepthMap
	float unocclu_z = eyeDepth[i];

	if (unocclu_z < near || unocclu_z > far) {
        shading[i] = 1; 
		return;
	}
    
    int xp = u * is;
    int yp = v * is;
    int x = i % is;
    int y = i / is;
	float blocker_search_sl = min(side_length / unocclu_z * abs(unocclu_z - 1), shadow_map_sl);
    float search_offset = blocker_search_sl / shadow_map_sl / 2 * is;
    search_offset = min(max(search_offset, 0.), float(is / 2));
    
	float z_avg = region_mean<double>(DepthMap, xp, yp, search_offset, is);
	float z2_avg = region_mean<double>(DepthMap2, xp, yp, search_offset, is);
    float occlusion;

    if (unocclu_z < z_avg + 0.01) {
        shading[i] = 1;
        return;
    }
    else {
	    occlusion = 1 - Chebyshev_test(z_avg, z2_avg - z_avg * z_avg, unocclu_z);
    }

    float occlu_z_avg = (z_avg - (1 - occlusion) * unocclu_z) / occlusion;

    occlu_z_avg = min(max(occlu_z_avg, near), far);

	float filter_size = (unocclu_z - occlu_z_avg) / occlu_z_avg * side_length;
    float filter_offset = filter_size / shadow_map_sl / 2 * is;
    filter_offset = min(max(filter_offset, 0.), float(is / 2));

	//compute the visibility
	float filter_z_avg = region_mean<double>(DepthMap, xp, yp, filter_offset, is);
	float filter_z2_avg = region_mean<double>(DepthMap2, xp, yp, filter_offset, is);
    
    if (unocclu_z < filter_z_avg + 0.005) {
        shading[i] = 1;
        return;
    }
    
    float visibility = Chebyshev_test(filter_z_avg, filter_z2_avg - filter_z_avg * filter_z_avg, unocclu_z);
    visibility = min(max(visibility, 0.), 1.);
    shading[i] = visibility;

	return;
}
    ''',
                   cuda_src=f'''
    @alias(shading,out0)
    @alias(eyeDepth,in0)
    @alias(SAT,in1)
    @alias(SAT2,in2)
    @alias(uv,in3)
    const auto is = shading_shape0;
    const int threads = 1024;
    const dim3 blocks = ( ( ( is * is ) - 1 ) / threads ) + 1;

    VSSM_cuda_kernel<float32><<<blocks,threads>>>(
        shading_p,
        eyeDepth_p,
        SAT_p,
        SAT2_p,
        uv_p,
        is,
        {side_length},
        {near},
        {far},
        {DM_sl}
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error in VSSM_cuda: %s\\n", cudaGetErrorString(err));
    ''')