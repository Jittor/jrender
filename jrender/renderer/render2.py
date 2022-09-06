from codecs import backslashreplace_errors
from enum import Flag
from hmac import trans_36
from statistics import mean
import jittor as jt
from jittor import nn
from skimage.io import imsave
import math
import copy

from .transform.transform import Transform
from .lighting import *
from .transform import *
from .dr import *
from ..structures import *
from .cuda import *


class Render():
    def __init__(self, image_size=256, background_color=[0, 0, 0], near=0.1, far=100,
                 camera_mode='look',
                 K=None, R=None, t=None, dist_coeffs=None, orig_size=512,
                 perspective=True, viewing_angle=30, viewing_scale=1.0,
                 eye=None, camera_direction=[0, 0, 1], threshold=1e-5, up=[0, 1, 0], MSAA=False
                 ):

        self.transform = Transform(camera_mode,
                                   K, R, t, dist_coeffs, orig_size,
                                   perspective, viewing_angle, viewing_scale,
                                   eye, camera_direction)

        self.rasterize = SoftRasterizeFunction(image_size,
                                               background_color=background_color, near=near, far=far, texture_type="vertex", dist_func="barycentric", aggr_func_rgb="hard")

        self.eye = eye
        self.camera_direction = camera_direction
        self.viewing_angle = viewing_angle
        self.viewing_scale = viewing_scale
        self.camera_mode = camera_mode
        self.perspective = perspective
        self.up =up

        self.threshold = threshold
        self._world_buffer = None
        self._normal_buffer = None
        self._KD_buffer = None
        self._faces_ind_buffer = None
        self._proj_vertices = None
        self.world_buffer_update = True
        self.normal_buffer_update = True
        self.KD_buffer_update = True
        self.proj_vertices_update = True
        self.MRT = None
        self._lights = None
        self.lights_transform = True

        self.image_size = image_size
        self.background_color = background_color
        self.near = near
        self.far = far
        self.MSAA = MSAA

    def view_rotate_m(self):
        z = jt.normalize(jt.array(self.camera_direction, "float32").unsqueeze(0), eps=1e-5)
        x = jt.normalize(jt.cross(jt.array(self.up).unsqueeze(0), z), eps=1e-5)
        y = jt.normalize(jt.cross(z, x), eps=1e-5)
        rotate = jt.concat([x, y, z], dim=0).transpose()
        return rotate

    def vp_transform(self, vertices, eye=None, camera_direction=None, viewing_angle=None, viewing_scale=None, camera_mode=None, perspective=None, up=None):
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
        if up == None:
            up = self.up
        return Transform(eye=eye, camera_direction=camera_direction, viewing_angle=viewing_angle, viewing_scale=viewing_scale, camera_mode=camera_mode, perspective=perspective, up=up).tranpos(vertices)

    def view_transform(self, vertices, eye=None, camera_direction=None, camera_mode=None, up=None):
        if len(vertices) == 0:
            return jt.array([])
        if eye == None:
            eye = self.eye
        if camera_direction == None:
            camera_direction = self.camera_direction
        if camera_mode == None:
            camera_mode = self.camera_mode
        if up == None:
            up = self.up
        return Transform(eye=eye, camera_direction=camera_direction, camera_mode=camera_mode, up=up).view_transform(vertices)

    def projection_transform(self, vertices, viewing_angle=None):
        if len(vertices) == 0:
            return jt.array([])
        if viewing_angle == None:
            viewing_angle = self.viewing_angle
        return Transform(viewing_angle=viewing_angle).projection_transform(vertices)

    def Rasterize(self, face_proj, face_info, MSAA = False):
        if len(face_info) == 0:
            return jt.array([])
        if MSAA:
            self.image_size = self.image_size * 2
        image = self.rasterize(face_proj.unsqueeze(0), face_info.unsqueeze(0))   
        image = jt.transpose(image.squeeze(0)[:3, :, :], (1, 2, 0))
        if MSAA:
            self.image_size = self.image_size / 2 
            image = nn.pool(image,2,"mean",stride=2)         
        return image

    def Rasterize_depth(self, face_proj):
        textures = jt.ones_like(face_proj)
        rasterize = RasterizeFunction(image=self.image_size,
                                      background_color=self.background_color, near=self.near, far=self.far, texture_type="vertex", dist_func="hard", aggr_func_rgb="none")
        rasterize(face_proj.unsqueeze(0), textures.unsqueeze(0))
        return rasterize.save_vars[4][:, 0, :, :].squeeze(0)

    def vertex_shader():
        return

    def fragment_shader(self, MRT):
        self.MRT = MRT
        lights = self.MRT["lights"]
        shadow_maps = self.SM(lights)

        worldcoords = self.world_buffer
        N = self.normal_buffer
        KD = self.KD_buffer
        color = jt.zeros_like(worldcoords)
        for i, light in enumerate(self.lights):

            if light.type == "ambient":
                color += light.intensity * light.color.unsqueeze(0).unsqueeze(0)
                continue

            V = jt.normalize(self.eye-worldcoords)
            if light.type == "directional":
                light_color = jt.array(light.color, "float32")
                L = -jt.normalize(jt.array(light.direction, "float32")).unsqueeze(0)
                proj_to_light_v = self.vp_transform(vertices=worldcoords, eye=light.position,
                                                    camera_direction=light.direction, viewing_scale=shadow_maps[i][1], perspective=False, camera_mode="look", up=light.up)
                eyeDepth = proj_to_light_v[:, :, 2]
                DepthMapUV = jt.stack([(proj_to_light_v[:, :, 0]+1.)/2, 1-(proj_to_light_v[:, :, 1]+1.)/2], dim=2)
                LightDepth = sample2D(shadow_maps[i][0], DepthMapUV)

                shading = (eyeDepth - LightDepth) < 0.05
                shading = shading.unsqueeze(2)

            elif light.type == "point":
                light_color = jt.array(light.color, "float32")
                L = jt.normalize((jt.array(light.position, "float32")-worldcoords), dim=2)
                proj_to_light_v = self.vp_transform(
                    vertices=worldcoords, eye=light.position, camera_direction=light.direction, viewing_angle=shadow_maps[i][1], perspective=True, camera_mode="look", up=light.up)
                eyeDepth = proj_to_light_v[:, :, 2]
                DepthMapUV = jt.stack([(proj_to_light_v[:, :, 0]+1.)/2, 1-(proj_to_light_v[:, :, 1]+1.)/2], dim=2)
                LightDepth = sample2D(shadow_maps[i][0], DepthMapUV, default=self.far+1)
     
                ''' c = eyeDepth.copy()
                a = LightDepth.copy()
                a[a > self.far] = 0
                #c[c > self.far] = 0
                #c[c < self.near] = 0
                c = jt.clamp(c, 0, 10.36)
                uv = jt.concat([DepthMapUV, jt.ones_like(eyeDepth.unsqueeze(2))], dim=2)
                uv = jt.clamp(uv, 0, 1)
                imsave("D:\Render\jrender\data\\results\\temp\\eyeDepth.jpg", c)
                imsave("D:\Render\jrender\data\\results\\temp\\LightDepth.jpg", a)
                imsave("D:\Render\jrender\data\\results\\temp\\DepthMapUV.jpg", uv)
                a[a > self.far] = 0'''
                a = LightDepth.copy()
                a[a > self.far] = 0
                #imsave("D:\Render\jrender\data\\results\\temp\\worldcoords.jpg",jt.clamp((worldcoords+1)/2,0,1)) 
                shading = eyeDepth-LightDepth < 0.05

                #imsave("D:\Render\jrender\data\\results\\temp\\shading.jpg", jt.clamp(eyeDepth - LightDepth, 0, 1))
                # exit()

                shading = shading.unsqueeze(2)

            shading = shading.float32()
            filter_w = jt.ones([7,7],"float32")/49 
            shading = conv_for_image(shading,filter_w,1)
            #imsave("D:\Render\jrender\data\\results\\temp\\shading.jpg", shading)
            H = jt.normalize(V + L, dim=2)
            cosine = nn.relu(jt.sum(L * N, dim=2)).unsqueeze(2)
            diffuse = light.intensity * light_color.unsqueeze(0).unsqueeze(0) * cosine * shading
            specular = jt.pow(nn.relu(jt.sum(H * N, dim=2)), 10).unsqueeze(2) * \
                light_color.unsqueeze(0).unsqueeze(0) * shading

            #imsave("D:\Render\jrender\data\\results\\temp\\kd.jpg",KD) 

            color += diffuse + specular

        color *= KD

        color = self.SSAO(color)

        #color = self.SSR(color)

        color = jt.clamp(color,0,1)

        return color

    def SM(self, lights):
        MRT=self.MRT
        depth_map = []
        face_vertices = MRT.get("worldcoords")
        for light in lights:
            if light.type == "point":
                eye = light.position
                direction = light.direction
                viewing_angle = 45
                proj_vertices = self.vp_transform(vertices=face_vertices, eye=eye,
                                                  camera_direction=direction, viewing_angle=viewing_angle, camera_mode="look", perspective=True, up=light.up)
                self.Rasterize(proj_vertices, proj_vertices)
                depth_map.append([self.rasterize.save_vars[4][:, 0, :, :].squeeze(0), viewing_angle])

                #temp = self.rasterize.save_vars[4][:, 0, :, :].squeeze(0).copy()
                #temp[temp == 10000000] = 0
                #imsave("D:\Render\jrender\data\\results\\temp\\shadow_map.jpg", temp[:,::-1]) 

            elif light.type == "directional":
                direction = light.direction
                eye = light.position
                viewing_scale = 0.9
                proj_vertices = self.vp_transform(vertices=face_vertices, eye=eye,
                                                  camera_direction=direction, viewing_scale=viewing_scale, camera_mode="look", perspective=False, up=light.up)
                depth_map.append([self.Rasterize(proj_vertices, proj_vertices)[:, :, 2], viewing_scale])

                #temp = self.Rasterize(proj_vertices, proj_vertices)[:, :, 2]
                #imsave("D:\Render\jrender\data\\results\\temp\\shadow_map.jpg", temp[:,::-1])



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
            self._proj_vertices = self.vp_transform(self._proj_vertices)
            self.proj_vertices_update = False
            self.MRT["render_update"][0] = False
        return self._proj_vertices

    @property
    def world_buffer(self):
        if self.world_buffer_update == True or self.MRT["render_update"][1] == True:
            face_normals = jt.matmul(self.MRT.get("normals").unsqueeze(2), self.view_rotate_m()).squeeze(2)
            self._normal_buffer = self.Rasterize(
                self.proj_vertices, face_normals)
            aggrs_info = self.rasterize.save_vars[4]

            #alpha = aggrs_info[:, 1, :, :].squeeze(0) == -1
            z = aggrs_info[:, 0, :, :].squeeze(0)
            self._faces_ind_buffer = aggrs_info[:,1,:,:].squeeze(0).int32()
            #z[alpha] = 0            

            image_size = self.rasterize.image_size
            x = jt.repeat((2*jt.arange(0, image_size)+1)/image_size-1, [image_size, 1])
            y = x[::, ::-1].transpose()
            width = math.tan(self.viewing_angle/180.*math.pi)
            self._world_buffer = jt.stack([x*z*width, y*z*width, z], dim=2)

            self.normal_buffer_update = False
            self.world_buffer_update = False
            self.MRT["render_update"][1] = False
            self.MRT["render_update"][2] = False
        return self._world_buffer

    @property
    def normal_buffer(self):
        if self.normal_buffer_update == True or self.MRT["render_update"][2] == True:
            face_normals = jt.matmul(self.MRT.get("normals").unsqueeze(2), self.view_rotate_m()).squeeze(2)
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
    def faces_ind_buffer(self):
        return self._faces_ind_buffer

    @property
    def lights(self):
        if self.MRT["render_update"][4] == True:
            self._lights = copy.deepcopy(self.MRT["lights"])
            self.lights_transform = True
        if self.lights_transform == True:
            for light in self._lights:

                light.direction = jt.matmul(jt.array(light.direction).unsqueeze(0),
                                            self.view_rotate_m()).numpy().tolist()
                light.position = jt.matmul(jt.array(light.position).unsqueeze(
                    0)-jt.array(self.eye).unsqueeze(0), self.view_rotate_m()).numpy().tolist()
                light.up = jt.matmul(jt.array(light.up).unsqueeze(0),
                                     self.view_rotate_m()).numpy().tolist()

            self.lights_transform = False
            self.MRT["render_update"][4] = False
        return self._lights

    

    def SSR(self,color):
        MRT = self.MRT
        ssr_faces = []
        for obj in MRT["objs"]:
            if obj.reflection_type == "mirror":
                i = MRT["name_dic"][obj.material_name]
                ssr_faces += MRT["obj_faces"][f"{i}"]
        ssr_faces = jt.array(ssr_faces).int32()
        world_buffer = self.world_buffer
        normal_buffer = self.normal_buffer
        faces_ind_buffer = self.faces_ind_buffer
        width = math.tan(self.viewing_angle/180.*math.pi)
        step = 1
        return SSR_cuda(color,world_buffer,normal_buffer,faces_ind_buffer,ssr_faces,width,step,level_intersect=0)
    
    def SSAO(self,color):        
        depth = self.world_buffer[:,:,2]
        normal_buffer = self.normal_buffer
        faces_ind_buffer = self.faces_ind_buffer
        width = math.tan(self.viewing_angle/180.*math.pi)
        occlusion = SSAO_cuda(depth,faces_ind_buffer,normal_buffer,width,sample_num=64,sample_range_r=0.1)

        filter_w = jt.ones([5,5],"float32")/25 
        occlusion = conv_for_image(occlusion,filter_w,0)
        color *= (1- occlusion).unsqueeze(2)
        imsave("D:\Render\jrender\data\\results\\temp\\rand.jpg", color)
        return

    def SSDO(self,color):
        return
        

