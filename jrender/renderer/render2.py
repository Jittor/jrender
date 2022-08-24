from hmac import trans_36
import jittor as jt
from skimage.io import imsave

from .transform.transform import Transform
from .lighting import *
from .transform import *
from .dr import *
from ..structures import *


class Render():
    def __init__(self, image_size=256, background_color=[0, 0, 0], near=1, far=100,
                 camera_mode='look',
                 K=None, R=None, t=None, dist_coeffs=None, orig_size=512,
                 perspective=True, viewing_angle=30, viewing_scale=1.0,
                 eye=None, camera_direction=[0, 0, 1], threshold=1e-5
                 ):

        self.transform = Transform(camera_mode,
                                   K, R, t, dist_coeffs, orig_size,
                                   perspective, viewing_angle, viewing_scale,
                                   eye, camera_direction)

        self.rasterize = SoftRasterizeFunction(image_size,
                                               background_color=background_color, near=near, far=far, texture_type="vertex")

        self.eye = eye
        self.camera_direction = camera_direction
        self.viewing_angle = viewing_angle
        self.viewing_scale = viewing_scale
        self.camera_mode = camera_mode
        self.perspective = perspective

        self.threshold = threshold
        self._world_buffer = None
        self._normal_buffer = None
        self._KD_buffer = None
        self._obj_mark_buffer = None
        self._proj_vertices = None
        self.world_buffer_update = True
        self.normal_buffer_update = True
        self.KD_buffer_update = True
        self.obj_mark_buffer_update = True
        self.proj_vertices_update = True
        self.MRT = None

    def vp_transform(self, vertices, eye=None, camera_direction=None, viewing_angle=None, viewing_scale=None, camera_mode=None, perspective=None):
        if len(vertices) == 0:
            return jt.array([])
        if viewing_angle == None:
            viewing_angle = self.viewing_angle
        if eye == None:
            eye = self.eye
        if camera_direction == None:
            camera_direction = self.camera_direction
        if camera_mode == None:
            camera_mode = self.camera_mode
        if perspective == None:
            perspective = self.perspective
        if viewing_scale == None:
            viewing_scale = self.viewing_scale
        return Transform(eye=eye, camera_direction=camera_direction, viewing_angle=viewing_angle, viewing_scale=viewing_scale, camera_mode=camera_mode, perspective=perspective).tranpos(vertices)

    def view_transform(self, vertices, eye=None, camera_direction=None, camera_mode=None):
        if len(vertices) == 0:
            return jt.array([])
        if eye == None:
            eye = self.eye
        if camera_direction == None:
            camera_direction = self.camera_direction
        if camera_mode == None:
            camera_mode = self.camera_mode
        return Transform(eye=eye, camera_direction=camera_direction, camera_mode=camera_mode).view_transform(vertices)

    def Rasterize(self, face_proj, face_info):
        if len(face_info) == 0:
            return jt.array([])
        image = self.rasterize(face_proj.unsqueeze(0), face_info.unsqueeze(0))
        image = jt.transpose(image.squeeze(0)[:3, :, :], (1, 2, 0))
        return image

    def vertex_shader():
        return

    def fragment_shader(self, MRT, objects, lights):
        self.MRT = MRT
        shadow_maps = self.SM(MRT, lights)

        worldcoords = self.world_buffer
        N = self.normal_buffer
        KD = self.KD_buffer
        obj_mark = self.obj_mark_buffer
        name_dic = self.MRT["name_dic"]
        color = jt.zeros_like(worldcoords)
        for i, light in enumerate(lights):
            V = jt.normalize(self.eye-worldcoords)
            if light.type == "directional":
                light_color = jt.array(light.color,"float32")
                light_direction = jt.array(light.direction, "float32")
                L = -light_direction.unsqueeze(0).unsqueeze(0)
                proj_to_light_v = self.vp_transform(vertices=worldcoords, eye=light.position,
                                                    camera_direction=light.direction, perspective="False")
                eyeDepth = proj_to_light_v[:, :, 2]
                DepthMapUV = jt.stack([(proj_to_light_v[:, :, 0]+1.)/2, 1-(proj_to_light_v[:, :, 1]+1.)/2], dim=1)
                LightDepth = sample2D(shadow_maps[i], DepthMapUV)
                shadow = (eyeDepth - LightDepth) < 0.1
                shadow = shadow.unsqueeze(2)

            elif light.type == "point":
                light_color = jt.array(light.color,"float32")
                L = (jt.array(light.position, "float32")-worldcoords)
                proj_to_light_v = self.vp_transform(
                    vertices=worldcoords, eye=light.position, camera_direction=light.direction)
                eyeDepth = proj_to_light_v[:, :, 2]
                DepthMapUV = jt.concat([(proj_to_light_v[:, :, 0]+1.)/2, 1-(proj_to_light_v[:, :, 1]+1.)/2], dim=2)
                LightDepth = sample2D(shadow_maps[i], DepthMapUV)
                shadow = (eyeDepth - LightDepth) < 0.1
                shadow = shadow.unsqueeze(2)

            elif light.type == "ambient":
                color += light.intensity * light.color.unsqueeze(0).unsqueeze(0)
                continue

            H = jt.normalize(V + L, dim=2)
            cosine = nn.relu(jt.sum(L * N, dim=2)).unsqueeze(2)
            diffuse = light.intensity * light_color.unsqueeze(0).unsqueeze(0) * cosine * (1-shadow)
            specular = jt.pow(nn.relu(jt.sum(H*N, dim=2)), 10).unsqueeze(2) * light_color.unsqueeze(0).unsqueeze(0) * (1-shadow)

            color += diffuse + specular

        color *= KD

        return color

    def SM(self, MRT, lights):
        depth_map = []
        face_vertices = MRT.get("worldcoords")
        for light in lights:
            if light.type == "point":
                eye = light.position
                direction = light.direction
                proj_vertices = self.vp_transform(vertices=face_vertices, eye=eye,
                                                  camera_direction=direction, camera_mode="look", perspective=True)
                depth_map.append(self.Rasterize(proj_vertices, proj_vertices)[:, :, 2])
                
                a=(self.Rasterize(proj_vertices, proj_vertices)[:, :, 2]-2)/2.1
                a=(255*a.numpy()).astype(np.uint8)
                imsave("D:\Render\jrender\data\\results\output_render\\test.jpg",a)
                exit()
            elif light.type == "directional":
                direction = light.direction
                eye = light.position
                proj_vertices = self.vp_transform(vertices=face_vertices, eye=eye,
                                                  camera_direction=direction, viewing_scale=0.8, camera_mode="look", perspective=False)
                depth_map.append(self.Rasterize(proj_vertices, proj_vertices)[:, :, 2])
        return depth_map

    def set_view(self, eye, camera_direction):
        self.eye = jt.array(eye, "float32")
        self.camera_direction = jt.array(camera_direction, "float32")
        self.world_buffer_update = True
        self.normal_buffer_update = True
        self.KD_buffer_update = True
        self.obj_mark_buffer_update = True
        self.proj_vertices_update = True
        return

    @property
    def proj_vertices(self):
        if self.proj_vertices_update == True or self.MRT["render_update"][0] == True:
            self._proj_vertices = self.MRT.get("worldcoords")
        self.proj_vertices_update = False
        self.MRT["render_update"][0] == False
        return self._proj_vertices

    @property
    def world_buffer(self):
        if self.world_buffer_update == True or self.MRT["render_update"][1] == True:
            face_vertices = self.MRT.get("worldcoords")
            self._world_buffer = self.Rasterize(
                self.proj_vertices, face_vertices)
        self.world_buffer_update = False
        self.MRT["render_update"][1] = False
        return self._world_buffer

    @property
    def normal_buffer(self):
        if self.normal_buffer_update == True or self.MRT["render_update"][2] == True:
            face_normals = self.MRT.get("normals")
            self._normal_buffer = self.Rasterize(
                self.proj_vertices, face_normals)
        self.normal_buffer_update = False
        self.MRT["render_update"][2] = False
        return self._normal_buffer

    @property
    def KD_buffer(self):
        if self.KD_buffer_update == True or self.MRT["render_update"][3] == True:
            KD = self.MRT.get("KD")
            self._KD_buffer = self.Rasterize(
                self.proj_vertices, KD)
        self.KD_buffer_update = False
        self.MRT["render_update"][3] = False
        return self._KD_buffer

    @property
    def obj_mark_buffer(self):
        if self.obj_mark_buffer_update == True or self.MRT["render_update"][4] == True:
            obj_mark = self.MRT.get("obj_mark")
            self._obj_mark_buffer = self.Rasterize(
                self.proj_vertices, obj_mark)[0]
            self._obj_mark_buffer = jt.round(self._obj_mark_buffer)
        self.obj_mark_buffer_update = False
        self.MRT["render_update"][4] = False
        return self._obj_mark_buffer


def sample2D(texture, pos, default=999999):  # 超出查找范围的值为999999  pos [0,1]*[0,1]

    value = jt.ones_like(texture)
    if len(texture.shape) == 2:
        dimension = 1
    else:
        dimension = texture.shape[2]
    return jt.code(value.shape, value.dtype, [texture, pos],
                   cuda_header='''
    #include <cuda.h>
    #include <cuda_runtime.h>

    namespace {
    template <typename scalar_t>
    __global__ void sample2D_cuda_kernel(
        const scalar_t* __restrict__ image,
        const scalar_t* __restrict__ pos,
        scalar_t* __restrict__ value, 
        size_t value_size,
        size_t image_height,
        size_t image_width,
        size_t dimension,
        scalar_t Default) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i * dimension >= value_size) {
        return;
    }

    const scalar_t* posn = &pos[i * 2];
    scalar_t* valuen = &value[i * dimension];

    const scalar_t pos_x =  (posn[0] * (image_width - 1));
    const scalar_t pos_y =  (posn[1] * (image_height - 1));

    if (pos_x < 0 || pos_x > (image_width - 1) || pos_y < 0 || pos_y > (image_height - 1)) {
        for (int k = 0; k < dimension; k++) {
            valuen[k] = Default;
        }
        return;
    }

    if (1) {
        /* bilinear sampling */
        const scalar_t weight_x1 = pos_x - (int)pos_x;
        const scalar_t weight_x0 = 1 - weight_x1;
        const scalar_t weight_y1 = pos_y - (int)pos_y;
        const scalar_t weight_y0 = 1 - weight_y1;
        for (int k = 0; k < dimension; k++) {
            scalar_t c = 0;
            c += image[((int)pos_y * image_width + (int)pos_x) * dimension + k] * (weight_x0 * weight_y0);
            c += image[((int)(pos_y + 1) * image_width + (int)pos_x) * dimension + k] * (weight_x0 * weight_y1);
            c += image[((int)pos_y * image_width + ((int)pos_x) + 1) * dimension + k] * (weight_x1 * weight_y0);
            c += image[((int)(pos_y + 1)* image_width + ((int)pos_x) + 1) * dimension + k] * (weight_x1 * weight_y1);
            valuen[k] = c;
        }
    } else {
        /* nearest neighbor */
        const int pos_xi = round(pos_x);
        const int pos_yi = round(pos_y);
        for (int k = 0; k < dimension; k++) {
            valuen[k] = image[(pos_yi * image_width + pos_xi) * 3 + k];
        }
    }
    }
    }
    ''',
    cuda_src=f'''
    @alias(texture, in0)
    @alias(pos, in1)
    @alias(value, out0)
    const auto value_size = value->num;
    const auto image_height = texture_shape0;
    const auto image_width = texture_shape1;
    
    const int threads = 1024;
    const dim3 blocks ((value_size / 3 - 1) / threads + 1);

    sample2D_cuda_kernel<float32><<<blocks, threads>>>(
        texture_p,
        pos_p,
        value_p,
        value_size,
        image_height,
        image_width,
        {dimension},
        {default});

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in load_textures: %s\\n", cudaGetErrorString(err));
    '''
                   )
