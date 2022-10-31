import jittor as jt

def FXAA_cuda(texture):
    texture = texture.float32()
    image = jt.ones_like(texture)
    return jt.code(image.shape,image.dtype,[texture],
    cuda_header='''
#include<cuda.h>
#include<cuda_runtime.h>
#include<math.h>

#define EDGE_THRESHOLD_MIN 0.0312
#define EDGE_THRESHOLD_MAX 0.125
#define ITERATIONS 12 
#define SUBPIXEL_QUALITY 0.75

__device__ float2& operator-(const float2& v1, const float2& v2) {
    return make_float2(v1.x - v2.x, v1.y - v2.y);
}

__device__ float2& operator+(const float2& v1, const float2& v2) {
    return make_float2(v1.x + v2.x, v1.y + v2.y);
}

__device__ float2& operator+=(const float2& v1, const float2& v2) {
    return make_float2(v1.x + v2.x, v1.y + v2.y);
}

__device__ float2& operator-=(const float2& v1, const float2& v2) {
    return make_float2(v1.x + v2.x, v1.y + v2.y);
}

__device__ float2& operator*(const float2& v1, float v) {
    return make_float2(v1.x * v, v1.y * v);
}

__device__ __forceinline__ float3& tex_sample(const float* texture, float xp, float yp, const int is) {
    xp = min(max(xp,0.),float(is - 1));
    yp = min(max(yp,0.),float(is - 1));
    yp = is - 1 - yp;
	float weight_x1 = xp - int(xp);
	float weight_x2 = 1 - weight_x1;
	float weight_y1 = yp - int(yp);
	float weight_y2 = 1 - weight_y1;
	float c[3] = {0.,0.,0.};
	for (int k = 0; k < 3; k++) {
		c[k] += texture[(int(xp) + int(yp) * is) * 3 + k] * weight_x2 * weight_y2;
		c[k] += texture[(int(xp) + (int(yp) + 1) * is) * 3 + k] * weight_x2 * weight_y1;
		c[k] += texture[(int(xp) + 1 + int(yp) * is) * 3 + k] * weight_x1 * weight_y2;
		c[k] += texture[(int(xp) + 1 + (int(yp) + 1) * is) * 3 + k] * weight_x1 * weight_y1;
	}
	return make_float3(c[0],c[1],c[2]);
}

__device__ __forceinline__ float toluma(const float3& rgb) {
	return sqrtf(rgb.x * 0.299 + rgb.y * 0.587 + rgb.z * 0.114);
}

template <typename scalar_t>
__global__ void FXAA_cuda(scalar_t* image, const scalar_t* texture, const int is) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= is * is) {
		return;
	}

    /*
    if(i != 1024 * 2048 + 1024){
        return;
    }
    */

	int xp = i % is;
	int yp = i / is;
	float2 uv = make_float2(float(xp), is - 1 - float(yp));
	float3 colorCenter = tex_sample(texture, uv.x, uv.y, is);
	float lumaCenter = toluma(colorCenter);
	float lumaLeft = toluma(tex_sample(texture, uv.x - 1, uv.y, is));
	float lumaRight = toluma(tex_sample(texture, uv.x + 1, uv.y, is));
	float lumaUp = toluma(tex_sample(texture, uv.x, uv.y + 1, is));
	float lumaDown = toluma(tex_sample(texture, uv.x, uv.y - 1, is));

	float lumaMin = min(lumaCenter, min(min(lumaDown, lumaUp), min(lumaLeft, lumaRight)));
	float lumaMax = max(lumaCenter, max(max(lumaDown, lumaUp), max(lumaLeft, lumaRight)));

	float lumaRange = lumaMax - lumaMin;

	if (lumaRange < max(EDGE_THRESHOLD_MIN, lumaMax * EDGE_THRESHOLD_MAX)) {
		image[i * 3 + 0] = colorCenter.x;
		image[i * 3 + 1] = colorCenter.y;
		image[i * 3 + 2] = colorCenter.z;
		return;
	}
    
	float lumaUpLeft = toluma(tex_sample(texture, uv.x - 1, uv.y + 1, is));
	float lumaUpRight = toluma(tex_sample(texture, uv.x + 1, uv.y + 1, is));
	float lumaDownLeft = toluma(tex_sample(texture, uv.x - 1, uv.y - 1, is));
	float lumaDownRight = toluma(tex_sample(texture, uv.x + 1, uv.y - 1, is));

	float lumaDownUp = lumaDown + lumaUp;
	float lumaLeftRight = lumaLeft + lumaRight;

	float lumaLeftCorners = lumaDownLeft + lumaUpLeft;
	float lumaDownCorners = lumaDownLeft + lumaDownRight;
	float lumaRightCorners = lumaDownRight + lumaUpRight;
	float lumaUpCorners = lumaUpRight + lumaUpLeft;

	float edgeHorizontal = abs(-2.0 * lumaLeft + lumaLeftCorners) + abs(-2.0 * lumaCenter + lumaDownUp) * 2.0 + abs(-2.0 * lumaRight + lumaRightCorners);
	float edgeVertical = abs(-2.0 * lumaUp + lumaUpCorners) + abs(-2.0 * lumaCenter + lumaLeftRight) * 2.0 + abs(-2.0 * lumaDown + lumaDownCorners);

	bool isHorizontal = (edgeHorizontal >= edgeVertical);

	float luma1 = isHorizontal ? lumaDown : lumaLeft;
	float luma2 = isHorizontal ? lumaUp : lumaRight;

	float gradient1 = luma1 - lumaCenter;
	float gradient2 = luma2 - lumaCenter;

	bool is1Steepest = abs(gradient1) >= abs(gradient2);

	// Gradient in the corresponding direction, normalized.
	float gradientScaled = 0.25 * max(abs(gradient1), abs(gradient2));
	// Choose the step size (one pixel) according to the edge direction.

	float lumaLocalAverage = 0.0;

	float stepLength = 1;

	if (is1Steepest) {
		// Switch the direction
		stepLength = -stepLength;
		lumaLocalAverage = 0.5 * (luma1 + lumaCenter);
	}
	else {
		lumaLocalAverage = 0.5 * (luma2 + lumaCenter);
	}

	float2 currentUv = uv;

	if (isHorizontal) {
		currentUv.y += stepLength * 0.5;
	}
	else {
		currentUv.x += stepLength * 0.5;
	}

	float2 offset = isHorizontal ? make_float2(1., 0.) : make_float2(0., 1.);

	float2 uv1 = currentUv - offset;
	float2 uv2 = currentUv + offset;

	// Read the lumas at both current extremities of the exploration segment, and compute the delta wrt to the local average luma.
	float lumaEnd1 = toluma(tex_sample(texture, uv1.x, uv1.y, is));
	float lumaEnd2 = toluma(tex_sample(texture, uv2.x, uv2.y, is));
	lumaEnd1 -= lumaLocalAverage;
	lumaEnd2 -= lumaLocalAverage;

	// If the luma deltas at the current extremities are larger than the local gradient, we have reached the side of the edge.
	bool reached1 = abs(lumaEnd1) >= gradientScaled;
	bool reached2 = abs(lumaEnd2) >= gradientScaled;
	bool reachedBoth = reached1 && reached2;

	// If the side is not reached, we continue to explore in this direction.
	if (!reached1) {
		uv1 -= offset;
	}
	if (!reached2) {
		uv2 += offset;
	}
	
    float QUALITY[11] = {1.,1.,1.,1.5, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0};

	if (!reachedBoth) {
        
		for (int j = 2; j < ITERATIONS; j++) {
			// If needed, read luma in 1st direction, compute delta.
			if (!reached1) {
				lumaEnd1 = toluma(tex_sample(texture, uv1.x, uv1.y, is));
				lumaEnd1 = lumaEnd1 - lumaLocalAverage;
			}
			// If needed, read luma in opposite direction, compute delta.
			if (!reached2) {
				lumaEnd2 = toluma(tex_sample(texture, uv2.x, uv2.y, is));
				lumaEnd2 = lumaEnd2 - lumaLocalAverage;
			}
			// If the luma deltas at the current extremities is larger than the local gradient, we have reached the side of the edge.
			reached1 = abs(lumaEnd1) >= gradientScaled;
			reached2 = abs(lumaEnd2) >= gradientScaled;
			reachedBoth = reached1 && reached2;

			// If the side is not reached, we continue to explore in this direction, with a variable quality.
			if (!reached1) {
				uv1 -= offset * QUALITY[i];
			}
			if (!reached2) {
				uv2 += offset * QUALITY[i];
			}

			// If both sides have been reached, stop the exploration.
			if (reachedBoth) { break; }
		}
	}

	float distance1 = isHorizontal ? (uv.x - uv1.x) : (uv.y - uv1.y);
	float distance2 = isHorizontal ? (uv2.x - uv.x) : (uv2.y - uv.y);

	bool isDirection1 = distance1 < distance2;
	float distanceFinal = min(distance1, distance2);

	float edgeThickness = (distance1 + distance2);

	float pixelOffset = -distanceFinal / edgeThickness + 0.5;

	// Is the luma at center smaller than the local average ?
	bool isLumaCenterSmaller = lumaCenter < lumaLocalAverage;

	// If the luma at center is smaller than at its neighbour, the delta luma at each end should be positive (same variation).
	// (in the direction of the closer side of the edge.)
	bool correctVariation = ((isDirection1 ? lumaEnd1 : lumaEnd2) < 0.0) != isLumaCenterSmaller;

	// If the luma variation is incorrect, do not offset.
	float finalOffset = correctVariation ? pixelOffset : 0.0;

	// Sub-pixel shifting
	float lumaAverage = (1.0 / 12.0) * (2.0 * (lumaDownUp + lumaLeftRight) + lumaLeftCorners + lumaRightCorners);
	// Ratio of the delta between the global average and the center luma, over the luma range in the 3x3 neighborhood.
	float subPixelOffset1 = (min(max(abs(lumaAverage - lumaCenter) / lumaRange, 0.0), 1.0));
	float subPixelOffset2 = (-2.0 * subPixelOffset1 + 3.0) * subPixelOffset1 * subPixelOffset1;
	// Compute a sub-pixel offset based on this delta.
	float subPixelOffsetFinal = subPixelOffset2 * subPixelOffset2 * SUBPIXEL_QUALITY;

	// Pick the biggest of the two offsets.
	finalOffset = max(finalOffset, subPixelOffsetFinal);

	float2 finalUV = uv;

	if (isHorizontal) {
		finalUV.y += finalOffset * stepLength;
	}
	else {
		finalUV.x += finalOffset * stepLength;
	}
    
	float3 finalColor = tex_sample(texture, finalUV.x, finalUV.y, is);
    //if (i < 2048)
	//printf("finalUV:%f, %f, %f\\n",finalUV.x, finalUV.y, finalColor.z);
	image[i * 3 + 0] = finalColor.x;
	image[i * 3 + 1] = finalColor.y;
	image[i * 3 + 2] = finalColor.z;
}
    ''',
    cuda_src='''
    @alias(texture,in0)
    @alias(image,out0)
    const int is = texture_shape0;
    const int threads = 1024;
    const dim3 blocks = (is * is  - 1) / threads + 1;
    FXAA_cuda<float32><<<blocks,threads>>>(image_p, texture_p, is);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error in FXAA_cuda: %s\\n", cudaGetErrorString(err));
    ''')