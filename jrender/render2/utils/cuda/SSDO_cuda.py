import jittor as jt

def SSDO_cuda(color,depth,face_ind_buffer,normal_buffer,width,sample_num,sample_range_r):
    depth = depth.float32()
    image = jt.zeros_like(color)
    normal_buffer = normal_buffer.float32()
    face_ind_buffer = face_ind_buffer.int32()
    return jt.code(image.shape,image.dtype,[color,depth,normal_buffer,face_ind_buffer],
    cuda_header='''
#include<cuda.h>
#include<cuda_runtime.h>
#include<curand_kernel.h>
#include<math.h>

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
    __device__ Vector3& operator+(const scalar_t v) {
 		return Vector3(x + v, y + v, z + v);
	}
	__device__ Vector3& operator-(const Vector3& v) {
		return Vector3(x - v.x, y - v.y, z - v.z);
	}
	__device__ Vector3& operator*(const Vector3& v) {
		return Vector3(x * v.x, y * v.y, z * v.z);
	}
	__device__ Vector3& operator*(const scalar_t v) {
		return Vector3(x * v, y * v, z * v);
	}
    __device__ Vector3& operator=(const scalar_t v) {
        this->x = v;
        this->y = v;
        this->z = v;
		return *this;
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
	__device__ scalar_t mod() {
		return sqrt(x * x + y * y + z * z);
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
__global__ void SSDO_cuda(scalar_t* image, const scalar_t* colors, const scalar_t* depths, const scalar_t* normals, const int* face_ind_buffer, curandStateXORWOW_t* states, size_t is, const int sample_num, const scalar_t sample_range_r, const scalar_t width) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	int np = is * is;
	if (i >= np || face_ind_buffer[i] == -1) {
		image[3 * i + 0] = colors[3 * i + 0];
		image[3 * i + 1] = colors[3 * i + 1];
		image[3 * i + 2] = colors[3 * i + 2];
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

	scalar_t ambient = 0;
	Vector3<scalar_t> indirect_radiance(0,0,0);
    scalar_t rho = 0.2;
	scalar_t light_area_per_sample = 1 / (scalar_t)sample_num * sample_range_r * sample_range_r;
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
		if (eye_depth > randp.z - 0.01) {
			ambient++;
		}
		else {
			// add indirect lighting
			Vector3<scalar_t> rand_light(colors[3 * rand_ind + 0], colors[3 * rand_ind + 1], colors[3 * rand_ind + 2]);
			Vector3<scalar_t> rand_normal(normals[3 * rand_ind + 0], normals[3 * rand_ind + 1], normals[3 * rand_ind + 2]);
			Vector3<scalar_t> randp_to_shading = randp - wcoord;
            scalar_t rand_d = randp_to_shading.mod();
            randp_to_shading.normalize();
			scalar_t cos_randn = - randp_to_shading.dot(rand_normal);
			cos_randn = max(cos_randn, 0.);
			scalar_t cos_n = randp_to_shading.dot(N);
			cos_n = max(cos_n, 0.);
			indirect_radiance = indirect_radiance + (rho * cos_randn * cos_n * light_area_per_sample) / (rand_d * rand_d) * rand_light;
            //Vector3<scalar_t> temp(1/(scalar_t)sample_num,0,0);
            //indirect_radiance = cos_randn * cos_n * rand_light;
		}

	}
	ambient /= sample_num;
    //ambient = 0;
	image[3 * i + 0] = ambient * colors[3 * i + 0] + indirect_radiance.x;
	image[3 * i + 1] = ambient * colors[3 * i + 1] + indirect_radiance.y;
	image[3 * i + 2] = ambient * colors[3 * i + 2] + indirect_radiance.z;
	return;
}
    ''',
    cuda_src=f'''
    @alias(image,out0)
    @alias(color,in0)
    @alias(depth,in1)
    @alias(normal_buffer,in2)
    @alias(face_ind_buffer,in3)
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
    
    SSDO_cuda<float32><<<blocks,threads>>>(
        image_p,
        color_p,
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
        printf("Error in SSDO_cuda: %s\\n", cudaGetErrorString(err));

    ''')