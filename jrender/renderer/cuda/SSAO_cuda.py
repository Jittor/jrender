import jittor as jt
from cv2 import imread

def SSAO_cuda(depth,face_ind_buffer,normal_buffer,width,sample_num,sample_range_r):
    depth = depth.float32()
    occlusion = jt.zeros_like(depth)
    normal_buffer = normal_buffer.float32()
    face_ind_buffer = face_ind_buffer.int32()
    return jt.code(occlusion.shape,occlusion.dtype,[depth,normal_buffer,face_ind_buffer],
    cuda_header='''
#include<cuda.h>
#include<cuda_runtime.h>
#include<curand_kernel.h>
#include<math.h>
#include<math_functions.h>

__global__ void _curand_init(curandStateXORWOW_t* state, size_t is){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= is * is){
        return;
    }
    curand_init(1,i,0,&state[i]);
    return;
}

template <typename scalar_t>
class Vector3 {
public:
	scalar_t x;
	scalar_t y;
	scalar_t z;
public:
	__device__ Vector3(scalar_t xx = 0, scalar_t yy = 0, scalar_t zz = 0) {
		x = xx;
		y = yy;
		z = zz;
	}
	__device__ Vector3(const Vector3& v) {
		x = v.x;
		y = v.y;
		z = v.z;
	}
	__device__ Vector3& operator+(const Vector3& v) {
		return Vector3(x + v.x, y + v.y, z + v.z);
	}
	__device__ Vector3& operator-(const Vector3& v) {
		return Vector3(x - v.x, y - v.y, z - v.z);
	}
	__device__ Vector3& operator*(const Vector3& v) {
		return Vector3(x * v.x, y * v.y, z * v.z);
	}
	__device__ Vector3& operator*(const scalar_t& v) {
		return Vector3(x * v, y * v, z * v);
	}
	__device__ void normalize() {
		scalar_t rmod = rsqrt(x * x + y * y + z * z);
		x *= rmod;
		y *= rmod;
		z *= rmod;
	}
	__device__ scalar_t dot(const Vector3& v) {
		return x * v.x + y * v.y + z * v.z;
	}
	__device__ Vector3& cross(const Vector3& v) {
		return Vector3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
	}
	__device__ friend Vector3& operator*(const scalar_t& v, const Vector3& v0) {
		return Vector3(v0.x * v, v0.y * v, v0.z * v);
	}
};

template <typename scalar_t>
class Vector2 {
public:
	scalar_t x;
	scalar_t y;
public:
	__device__ Vector2(scalar_t xx = 0, scalar_t yy = 0) {
		x = xx;
		y = yy;
	}
	__device__ Vector2(const Vector2& v) {
		x = v.x;
		y = v.y;
	}
	__device__ Vector2& operator+(const Vector2& v) {
		return Vector2(x + v.x, y + v.y);
	}
	__device__ Vector2& operator-(const Vector2& v) {
		return Vector2(x - v.x, y - v.y);
	}
	__device__ Vector2& operator*(const Vector2& v) {
		return Vector2(x * v.x, y * v.y);
	}
	__device__ Vector2& operator*(const scalar_t& v) {
		return Vector2(x * v, y * v);
	}
	__device__ void normalize() {
		scalar_t rmod = rsqrt(x * x + y * y);
		x *= rmod;
		y *= rmod;
	}
	__device__ friend Vector2& operator*(const scalar_t& v, const Vector2& v0) {
		return Vector2(v0.x * v, v0.y * v);
	}
};

#define pi 3.1415926

template <typename scalar_t>
__global__ void SSAO_cuda(scalar_t* occlusions, const scalar_t* depths, const scalar_t* normals, const int* face_ind_buffer, curandStateXORWOW_t* states, size_t is, const int sample_num, const scalar_t sample_range_r ,const scalar_t width) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	int np = is * is;
	if (i >= np || face_ind_buffer[i] == -1) {
        occlusions[i] = 0;
		return;
	}
	
	int xp = i % is;
	int yp = i / is;
	scalar_t xi = (2 * scalar_t(xp) + 1 - is) / is;
	scalar_t yi = (2 * (is - 1 - scalar_t(yp)) + 1 - is) / is;

	Vector3<scalar_t> N(normals[i * 3 + 0], normals[i * 3 + 1], normals[i * 3 + 2]);
	Vector3<scalar_t> wcoord(xi * width * depths[i], yi * width * depths[i], depths[i]);
	Vector3<scalar_t> T, B;
	// create TBN
	if (abs(N.x) > 0.5 || abs(N.y) > 0.5) {
		T.y = N.x;
		T.x = -N.y;
		T.z = 0;
        T.normalize();
	}
	else {
		T.x = N.z;
		T.z = -N.x;
		T.y = 0;
        T.normalize();
	}
	B = T.cross(N);

	scalar_t occlusion = 0;
	for (int k = 0; k < sample_num; k++) {
		// create a random point in the sphere
		Vector3<scalar_t> rand(scalar_t(curand_uniform(&states[i])), scalar_t(curand_uniform(&states[i])), scalar_t(curand_uniform(&states[i])));
		Vector3<scalar_t> randp;
		scalar_t randr = sqrt(rand.z / 3);
		randp.x = randr * rand.y * cos(2 * pi * rand.x);
		randp.y = randr * sqrt(1 - rand.y * rand.y);
		randp.z = randr * rand.y * sin(2 * pi * rand.x);
		//transform to the world coordinate
		randp = randp.x * T + randp.y * N + randp.z * B;
		//randp.normalize()
		randp = randp * sample_range_r + wcoord;
		int randp_xp = round(is * (randp.x / randp.z / width + 1) / 2);
		int randp_yp = is - round(is * (randp.y / randp.z / width + 1) / 2);
		int rand_ind = randp_yp * is + randp_xp;
		
        if (rand_ind >= np || rand_ind < 0) {
			continue;
		}
        
		scalar_t eye_depth = depths[rand_ind];
		if (eye_depth < randp.z) {
			occlusion++;
		}

	}
	occlusion /= sample_num;
	occlusions[i] = occlusion;
	return;
}

    ''',
    cuda_src=f'''
    @alias(occlusion,out0)
    @alias(depth,in0)
    @alias(normal_buffer,in1)
    @alias(face_ind_buffer,in2)
    const auto is = depth_shape0;
    const auto np = is * is;

    const int threads = 1024;
    const dim3 blocks = (np - 1) / threads + 1;

    //Create rand_generators
    curandStateXORWOW_t* states;
    const auto nbytes = np * sizeof(curandStateXORWOW_t);
    cudaMalloc(&states, nbytes);
    //cudaMemcpy(image_p,color_p,12 * is * is,cudaMemcpyDeviceToDevice);

    _curand_init<<<blocks,threads>>>(states, is);
    cudaDeviceSynchronize();
    
    SSAO_cuda<float32><<<blocks,threads>>>(
        occlusion_p,
        depth_p,
        normal_buffer_p,
        face_ind_buffer_p,
        states,
        is,
        {sample_num},
        {sample_range_r},
        {width}
    );

    cudaFree(states);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error in SSR_cuda: %s\\n", cudaGetErrorString(err));

    ''')