from cmath import tan
import jittor as jt
import math

# shading [sz,sz]
# eyeDepth [sz,sz]
# DepthMap [sz,sz]
# DepthMap2 [sz,sz]
# uv [sz,sz,2]             size:sz*sz*6


def VSSM_cuda_mip(eyeDepth, DepthMap, DepthMap2, index, uv, light): 
    shading = jt.ones_like(eyeDepth)
    maxMipmapLevel = index.shape[0] - 1 
    side_length = math.sqrt(light.area)
    side_length = 0.05
    near = light.near
    far = light.far
    DM_sl = math.tan(light.viewing_angle / 180. * math.pi)
    viewing_angle = light.viewing_angle
    return jt.code(shading.shape, shading.dtype, [eyeDepth, DepthMap, DepthMap2, index, uv],
                   cuda_header='''
#include<cuda.h>
#include<cuda_runtime.h>

__device__ __forceinline__ float Chebyshev_test(float mean, float variance, float lower_bound) {
float a = lower_bound - mean;
float denominator = min(variance + a * a, 1e-5);
return min(max(variance / (variance + a * a),0.),1.);
}

__device__ __forceinline__ float tex_fetch(float* texture, int x , int y , int is){
    int index = x + y * is;
    if (x >= is) {
        index -= 1;
    }
    if (y >= is){
        index -= is;
    } 
    return texture[index];
}

__device__ __forceinline__ float bilinear_sample(float* texture, float u , float v , int index_max){
    int size = sqrt(float(index_max));
    float xp = u * size;
    float yp = v * size;
    int x = int(u * size);
    int y = int(v * size);
    float weight_x1 = xp - int(xp);
    float weight_x2 = 1 - weight_x1;
    float weight_y1 = yp - int(yp);
    float weight_y2 = 1 - weight_y1;
    float depth = 0;
    depth += tex_fetch(texture, int(xp), int(yp), size) * weight_x2 * weight_y2;
    depth += tex_fetch(texture, int(xp), int(yp) + 1, size) * weight_x2 * weight_y1;
    depth += tex_fetch(texture, int(xp) + 1, int(yp), size) * weight_x1 * weight_y2;
    depth += tex_fetch(texture, int(xp) + 1, int(yp) + 1, size) * weight_x1 * weight_y1;
    return depth;
}

template <typename scalar_t>
__global__ void VSSM_cuda_kernel(
	scalar_t* shading,
	const scalar_t* eyeDepth,
	const scalar_t* DepthMap,
	const scalar_t* DepthMap2,
	const int* mipMap_index,
	const scalar_t* uv,
	const int is,
	const scalar_t side_length,
	const scalar_t near,
    const scalar_t far,
	const float shadow_map_sl,
    const int maxMipmapLevel)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int np = is * is;
	if (i >= np) {
		return;
	}
	float u = uv[i * 2 + 0];
	float v = uv[i * 2 + 1];

	if (u < 0 || u > 1 || v < 0 || v > 1) {
		shading[i] = 1;
		return;
	}

	//compute the size of block_search region on the DepthMap
	float unocclu_z = eyeDepth[i];
	if (unocclu_z < near || unocclu_z > far) {
        shading[i] = 1;
		return;
	}
	float blocker_search_sl = min(side_length / unocclu_z * abs(unocclu_z - 1), shadow_map_sl);
	float ddx = blocker_search_sl / shadow_map_sl * is;
	int blocker_search_lod = min(max(log2(ddx + 0.5), 0.),float(maxMipmapLevel)-1);
    
    //blocker_search_lod = 4;
    int mipMap_np = mipMap_index[blocker_search_lod + 1] - mipMap_index[blocker_search_lod];
	float z_avg = bilinear_sample((float*)&DepthMap[mipMap_index[blocker_search_lod]], u, v, mipMap_np);
	float z2_avg = bilinear_sample((float*)&DepthMap2[mipMap_index[blocker_search_lod]], u, v, mipMap_np);

	float occlusion = 1 - Chebyshev_test(z_avg, z2_avg - z_avg * z_avg, unocclu_z - 0.04);
	float occlu_z_avg = (z_avg - (1 - occlusion) * unocclu_z) / occlusion;
	float filter_size = (unocclu_z - occlu_z_avg) / occlu_z_avg * side_length;
    ddx = filter_size / shadow_map_sl * is;
	int filter_lod = min(max(log2(ddx + 0.5), 0.),float(maxMipmapLevel)-1);
	
    //compute the visibility
    int mipMap2_np = mipMap_index[filter_lod + 1] - mipMap_index[filter_lod];
	float filter_z_avg = bilinear_sample((float*)&DepthMap[mipMap_index[filter_lod]], u, v, mipMap2_np);
	float filter_z2_avg = bilinear_sample((float*)&DepthMap2[mipMap_index[filter_lod]], u, v, mipMap2_np);
    shading[i] = occlusion;
	//shading[i] = Chebyshev_test(filter_z_avg, filter_z2_avg - filter_z_avg * filter_z_avg, unocclu_z);
    
	return;
}
    ''',
                   cuda_src=f'''
    @alias(shading,out0)
    @alias(eyeDepth,in0)
    @alias(DepthMap,in1)
    @alias(DepthMap2,in2)
    @alias(index,in3)
    @alias(uv,in4)
    const auto is = shading_shape0;
    const int threads = 1024;
    const dim3 blocks = ( ( ( is * is ) - 1 ) / threads ) + 1;

    VSSM_cuda_kernel<float32><<<blocks,threads>>>(
        shading_p,
        eyeDepth_p,
        DepthMap_p,
        DepthMap2_p,
        index_p,
        uv_p,
        is,
        {side_length},
        {near},
        {far},
        {DM_sl},
        {maxMipmapLevel}
    );

    //cudaDestroyTextureObject(texObj);
    //cudaFreeArray(cuArray);
    //cudaDestroyTextureObject(texObj2);
    //cudaFreeArray(cuArray2);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error in VSSM_cuda: %s\\n", cudaGetErrorString(err));
    ''')


def _VSSM_cuda(eyeDepth, DepthMap, uv, light):
    DepthMap2 = DepthMap * DepthMap
    shading = jt.ones_like(eyeDepth)
    side_length = math.sqrt(light.area)
    near = light.near
    viewing_angle = light.viewing_angle
    return jt.code(shading.shape, shading.dtype, [eyeDepth, DepthMap, DepthMap2, uv],
                   cuda_header='''
#include<cuda.h>
#include<cuda_runtime.h>

__device__ __forceinline__ float Chebyshev_test(float mean, float variance, float lower_bound) {
float a = lower_bound - mean;
float denominator = min(variance + a * a, 1e-5);
return variance / (variance + a * a);
}

template <typename scalar_t>
__global__ void VSSM_cuda_kernel(
	scalar_t* shading,
	const scalar_t* eyeDepth,
	cudaTextureObject_t DepthMap,
	cudaTextureObject_t DepthMap2,
	const scalar_t* uv,
	const int is,
	const scalar_t side_length,
	const scalar_t near,
	const scalar_t viewing_angle
)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int np = is * is;
	if (i >= np) {
		return;
	}
	float u = uv[i * 2 + 0];
	float v = uv[i * 2 + 1];

	if (u < 0 || u > 1 || v < 0 || v > 1) {
		shading[i] = 1;
		return;
	}

	//compute the size of block_search region on the DepthMap
	float unocclu_z = eyeDepth[i];
	if (unocclu_z < near) {
		return;
	}
	float shadow_map_sl = tan(viewing_angle);
	float blocker_search_sl = min(side_length / unocclu_z * abs(unocclu_z - 1), shadow_map_sl);
	float ddx = blocker_search_sl / shadow_map_sl * is;
	float blocker_search_lod = max(log2(ddx + 0.5), 0.);
    
    blocker_search_lod = 8.2;
	float z_avg = tex2DLod<float>(DepthMap, u, v, blocker_search_lod);
	float z2_avg = tex2DLod<float>(DepthMap2, u, v, blocker_search_lod);
    
    /*
	float occlusion = Chebyshev_test(z_avg, z2_avg - z_avg * z_avg, unocclu_z);
	float occlu_z_avg = (z_avg - (1 - occlusion) * unocclu_z) / occlusion;
	float filter_size = (unocclu_z - occlu_z_avg) / occlu_z_avg * side_length;
    ddx = blocker_search_sl / shadow_map_sl * is;
	float filter_lod = max(log2(ddx + 0.5), 0.);
    */
	shading[i] = z_avg;

	//compute the visibility
	//float filter_z_avg = tex2DLod<float>(DepthMap, u, v, filter_lod);
	//float filter_z2_avg = tex2DLod<float>(DepthMap2, u, v, filter_lod);
	//shading[i] = 1 - Chebyshev_test(filter_z_avg, filter_z2_avg - filter_z_avg * filter_z_avg, unocclu_z);
    
	return;
}
    ''',
                   cuda_src=f'''
    @alias(shading,out0)
    @alias(eyeDepth,in0)
    @alias(DepthMap,in1)
    @alias(DepthMap2,in2)
    @alias(uv,in3)
    const auto is = shading_shape0;
    const int threads = 1024;
    const dim3 blocks = ( ( ( is * is ) - 1 ) / threads ) + 1;

    
    //Create cudaArray
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
    cudaArray_t cuArray,cuArray2;
    cudaMallocArray(&cuArray, &channelDesc, is, is);
    cudaMallocArray(&cuArray2, &channelDesc, is, is);
    cudaMemcpy2DToArray(cuArray, 0, 0, DepthMap_p, 4 * is, 4 * is, is, cudaMemcpyHostToDevice);
    cudaMemcpy2DToArray(cuArray2, 0, 0, DepthMap2_p, 4 * is, 4 * is, is, cudaMemcpyHostToDevice);

    //Create resDescription
    cudaResourceDesc resDesc,resDesc2;
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    resDesc2.resType = cudaResourceTypeArray;
    resDesc2.res.array.array = cuArray2;
    
    //Create texDescription
    cudaTextureDesc texDesc;
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.mipmapFilterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.maxMipmapLevelClamp = 10;
	texDesc.minMipmapLevelClamp = 0;
    texDesc.normalizedCoords = 1;
    
    //Create textureObj
    cudaTextureObject_t texObj = 1;
    cudaTextureObject_t texObj2 = 2;
    
    //cudaResourceViewDesc resvDesc;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    cudaCreateTextureObject(&texObj2, &resDesc2, &texDesc, NULL);
    

    VSSM_cuda_kernel<float32><<<blocks,threads>>>(
        shading_p,
        eyeDepth_p,
        texObj,
        texObj2,
        uv_p,
        is,
        {side_length},
        {near},
        {viewing_angle}
    );

    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
    cudaDestroyTextureObject(texObj2);
    cudaFreeArray(cuArray2);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error in VSSM_cuda: %s\\n", cudaGetErrorString(err));
    ''')

def VSSM_cuda(eyeDepth, SAT, SAT2, uv, light, SM):
    shading = jt.ones_like(eyeDepth)
    side_length = math.sqrt(light.area)
    near = light.near
    far = light.far
    DM_sl = math.tan(light.viewing_angle / 180. * math.pi)
    return jt.code(shading.shape, shading.dtype, [eyeDepth, SAT, SAT2, uv, SM],
                   cuda_header='''
#include<cuda.h>
#include<cuda_runtime.h>

__device__ __forceinline__ float Chebyshev_test(float mean, float variance, float upper_bound) {
    float a = upper_bound - mean;
    float denominator = max((a * a), 1e-5);
    float tail = min(max(variance / (variance + denominator), 0.), 1.);
    //float tail = min(max(variance / (denominator), 0.), 1.);
    if (a > 0){
        return 1 - tail / 2;
    } else {
        return tail / 2;
    }
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
    const int* SM,
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
	float occlusion = Chebyshev_test(z_avg, z2_avg - z_avg * z_avg, unocclu_z);
    //float occlusion = 1 - region_mean<int>(SM, x, y, search_offset, is);
    occlusion = min(max(occlusion, 0.), 1.);
    float occlu_z_avg;
    if (occlusion > 1e-5){
        occlu_z_avg = (z_avg - (1 - occlusion) * unocclu_z) / occlusion;
        //occlu_z_avg = 1;
    }
    else{
        occlu_z_avg = z_avg;
        //occlu_z_avg = 1;
    }
    occlu_z_avg = min(max(occlu_z_avg, near), far);

	float filter_size = (unocclu_z - occlu_z_avg) / occlu_z_avg * side_length;
    float filter_offset = filter_size / shadow_map_sl / 2 * is;
    filter_offset = min(max(filter_offset, 0.), float(is / 2));

	//compute the visibility
	//float filter_z_avg = region_mean<double>(DepthMap, xp, yp, filter_offset, is);
	//float filter_z2_avg = region_mean<double>(DepthMap2, xp, yp, filter_offset, is);
    //float visibility = (1 - Chebyshev_test(filter_z_avg, filter_z2_avg - filter_z_avg * filter_z_avg, unocclu_z));
    float visibility = region_mean<int>(SM, x, y, filter_offset, is);
    occlusion = min(max(visibility, 0.), 1.);
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
    @alias(SM,in4)
    const auto is = shading_shape0;
    const int threads = 1024;
    const dim3 blocks = ( ( ( is * is ) - 1 ) / threads ) + 1;

    VSSM_cuda_kernel<float32><<<blocks,threads>>>(
        shading_p,
        eyeDepth_p,
        SAT_p,
        SAT2_p,
        uv_p,
        SM_p,
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