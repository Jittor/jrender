
from queue import Empty
import re
import jittor as jt
from jittor import nn
from skimage.io import imsave
import math
import copy
from typing import List
import time
import cv2

from ..renderer.transform.transform import Transform
from ..renderer.dr import *

from .utils import *
from .RenderDesc import *
from ..Scene.objects import *


class Render():

    MRT: MultipleRenderTargets
    GeometryDesc: GeometryDescption
    MaterialDesc: MaterialDescption
    IlluminationDesc: IlluminationDescption
    _lights: List[Light]

    def __init__(self, image_size=256, background_color=[0, 0, 0], near=0.1, far=100,
                 camera_mode='look',
                 K=None, R=None, t=None, dist_coeffs=None, orig_size=512,
                 perspective=True, viewing_angle=30, viewing_scale=1.0,
                 eye=None, camera_direction=[0, 0, 1], threshold=2e-2, up=[0, 1, 0], MSAA=False
                 ):

        self.transform = Transform(camera_mode,
                                   K, R, t, dist_coeffs, orig_size,
                                   perspective, viewing_angle, viewing_scale,
                                   eye, camera_direction)

        # view setting
        self.eye = eye
        self.camera_direction = camera_direction
        self.viewing_angle = viewing_angle
        self.viewing_scale = viewing_scale
        self.camera_mode = camera_mode
        self.perspective = perspective
        self.up = up
        self.near = near
        self.far = far
        self.fill_back = True

        # scene setting
        self.threshold = threshold

        # g_buffer
        self._world_buffer = None
        self._normal_buffer = None
        self._KD_buffer = None
        self._faces_ind_buffer = None
        self._proj_vertices = None
        self.world_buffer_update = True
        self.normal_buffer_update = True
        self.KD_buffer_update = True
        self.proj_vertices_update = True

        # render_setting
        self.MRT = None
        self.GeometryDesc = None
        self.MaterialDesc = None
        self.IlluminationDesc = None
        self.lights_transform = True
        self._lights = None

        # PBR
        self._albedo_buffer = None
        self.albedo_buffer_update = True
        self._metallic_roughness_buffer = None
        self.metallic_roughness_buffer_update = True

        # rasterize setting
        self.texture_type = "vertex"
        self.image_size = image_size
        self.background_color = background_color
        self.MSAA = MSAA

        self.rasterize = SoftRasterizeFunction(image_size,
                                        background_color=background_color, near=near, far=far, texture_type=self.texture_type, dist_func="barycentric", aggr_func_rgb="hard")

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

    def Rasterize(self, face_proj, face_info, MSAA=False, fill_back=True, texture_type = "vertex"):
        if len(face_info) == 0:
            return jt.array([])
        self.rasterize.fill_back = fill_back
        self.rasterize.texture_type = texture_type
        if MSAA:
            self.rasterize.image_size = int(self.image_size * 2)
        image = self.rasterize(face_proj.unsqueeze(0), face_info.unsqueeze(0))
        if MSAA:
            self.rasterize.image_size = int(self.rasterize.image_size / 2)
            image = nn.pool(image, 2, "mean", stride=2)
        image = jt.transpose(image.squeeze(0)[:3, :, :], (1, 2, 0))
        self.rasterize.fill_back = self.fill_back
        self.rasterize.texture_type = self.texture_type
        return image

    def Rasterize_depth(self, face_proj):
        textures = jt.ones_like(face_proj)
        rasterize = RasterizeFunction(image=self.image_size,
                                      background_color=self.background_color, near=self.near, far=self.far, texture_type="vertex", dist_func="hard", aggr_func_rgb="none")
        rasterize(face_proj.unsqueeze(0), textures.unsqueeze(0))
        return rasterize.save_vars[4][:, 0, :, :].squeeze(0)

    def vertex_shader():
        return

    def fragment_shader(self):
        worldcoords = self.world_buffer
        N = self.normal_buffer
        color = jt.zeros_like(worldcoords)
        textures = self.KD_buffer

        for light in self.lights:
            if light.type == "ambient":
                color += light.intensity * light.color.unsqueeze(0).unsqueeze(0) * textures
                continue

            V = jt.normalize(self.eye-worldcoords, dim=2)
            if light.type == "directional":
                light_color = jt.array(light.color, "float32")
                L = -jt.normalize(jt.array(light.direction, "float32"), dim=0).unsqueeze(0)

            elif light.type == "point":
                light_color = jt.array(light.color, "float32")
                L = jt.normalize((jt.array(light.position, "float32") - worldcoords), dim=2)

            elif light.type == "area":
                light_color = jt.array(light.color, "float32")
                L = jt.normalize((jt.array(light.position, "float32") - worldcoords), dim=2)

            H = jt.normalize(V + L, dim=2)
            cosine = nn.relu(jt.sum(L * N, dim=2)).unsqueeze(2)
            shading = self.light_visibility(light)

            # blinn_phong shading
            if self.IlluminationDesc.shading == "blinn_phong":
                Ns = 10
                diffuse = light.intensity * light_color.unsqueeze(0).unsqueeze(0) * cosine
                specular = jt.pow(nn.relu(jt.sum(H * N, dim=2)), Ns).unsqueeze(2) * \
                    light_color.unsqueeze(0).unsqueeze(0)
                color += (diffuse + specular) * textures * shading

            # Cook-Torrance BRDF
            elif self.IlluminationDesc.shading == "Cook_Torrance":
                F0 = jt.array((0.04, 0.04, 0.04), "float32")
                albedo = self.albedo_buffer
                metallic = self.metallic_roughness_buffer[:, :, 0].unsqueeze(2)
                roughness = self.metallic_roughness_buffer[:, :, 1].unsqueeze(2)
                with_specular = self.metallic_roughness_buffer[:, :, 2].unsqueeze(2)
                F0 = F0 * (1 - metallic) + albedo * metallic
                radiance = light.intensity * (light_color.unsqueeze(0).unsqueeze(0) * cosine)

                NDF = GGX(N, H, roughness)
                G = GeometrySmith(N, V, L, roughness)
                F = fresnelSchlick(nn.relu(jt.sum(H * V, dim=2)), F0)
                KS = F
                KD = 1.0 - KS
                KD *= (1.0 - metallic)

                diffuse = KD * radiance * shading
                numerator = NDF * G * F
                denominator = (4.0 * nn.relu(jt.sum(N * V, dim=2)) * nn.relu(jt.sum(N * L, dim=2))).unsqueeze(2)
                specular = numerator / jt.clamp(denominator, 0.01) * radiance * shading * with_specular

                color += diffuse * textures + specular * jt.ones_like(textures)
        
        color = self.SSSR(color)
        color = self.SSR(color)
        #color = self.SSAO(color)
        color = jt.clamp(color, 0, 1)
        color = FXAA_cuda(color)

        return color

    def light_visibility(self, light: Light):

        # eyeDepth: distance from light_plane to the shading point
        # LightDepth: distance from light_plane to the first hitting point
        # shading: visibilty

        if light.type == "ambient" or light.shadow == False:
            return 1
        if light.type == "directional":
            proj_to_light_v = self.vp_transform(vertices=self.world_buffer, eye=light.position,
                                                camera_direction=light.direction, viewing_scale=light.viewing_scale, perspective=False, camera_mode="look", up=light.up)
            eyeDepth = proj_to_light_v[:, :, 2]
            DepthMapUV = jt.stack([(proj_to_light_v[:, :, 0]+1.)/2, 1-(proj_to_light_v[:, :, 1]+1.)/2], dim=2)
            light.DepthMap.uv = DepthMapUV
            LightDepth = light.DepthMap.query_uv
            # PCF filter
            shading = (eyeDepth - LightDepth) < 0.04
            shading = shading.unsqueeze(2).float32()
            filter_w = jt.ones([7, 7], "float32")/49
            shading = conv_for_image(shading, filter_w, 1)

        elif light.type == "point":
            proj_to_light_v = self.vp_transform(
                vertices=self.world_buffer, eye=light.position, camera_direction=light.direction, viewing_angle=light.viewing_angle, perspective=True, camera_mode="look", up=light.up)
            eyeDepth = proj_to_light_v[:, :, 2]
            DepthMapUV = jt.stack([(proj_to_light_v[:, :, 0]+1.)/2, 1-(proj_to_light_v[:, :, 1]+1.)/2], dim=2)
            light.DepthMap.uv = DepthMapUV
            LightDepth = light.DepthMap.query_uv
            # PCF filter
            shading = eyeDepth-LightDepth < 0.02
            shading = shading.unsqueeze(2)
            shading = shading.float32()
            filter_w = jt.ones([7, 7], "float32")/49
            shading = conv_for_image(shading, filter_w, 1)
            imsave("D:\Render\jrender\data\\results\\temp\\lightDepth.jpg", LightDepth)
            imsave("D:\Render\jrender\data\\results\\temp\\shading.jpg",shading)
            #LightDepth[LightDepth > self.far] = 0
            #imsave("D:\Render\jrender\data\\results\\temp\\lightDepth.jpg", LightDepth)
            # exit()

        # TODO: VSSM
        elif light.type == "area":      
            proj_to_light_v = self.vp_transform(
                vertices=self.world_buffer, eye=light.position, camera_direction=light.direction, viewing_angle=light.viewing_angle, perspective=True, camera_mode="look", up=light.up)
            eyeDepth = proj_to_light_v[:, :, 2]
            DepthMapUV = jt.stack([(proj_to_light_v[:, :, 0]+1.)/2, 1-(proj_to_light_v[:, :, 1]+1.)/2], dim=2)
            light.DepthMap.uv = DepthMapUV
            DepthMap = light.DepthMap.image
            LightDepth = light.DepthMap.query_uv
            # VSSM
            #eyeDepth[eyeDepth > self.far] = 0
            #eyeDepth[eyeDepth < self.near] = 0
            # imsave("D:\Render\jrender\data\\results\\temp\\eyeDepth.jpg",eyeDepth)
            SM = eyeDepth-LightDepth < 0.04
            SM = Texture.generate_SAT(SM).int32()
            SAT = Texture.generate_SAT(DepthMap)
            SAT2 = Texture.generate_SAT(DepthMap * DepthMap)
            #mipmap, index = Texture.generate_mipmap(DepthMap)
            #mipmap2, index = Texture.generate_mipmap(DepthMap * DepthMap)
            shading = VSSM_cuda(eyeDepth, SAT, SAT2, DepthMapUV, light, SM)
            #shading = VSSM_cuda_mip(eyeDepth, mipmap, mipmap2, index, DepthMapUV, light)
            #shading = jt.clamp(shading,0,1)
            #shading[shading > 50] = 0
            #shading[shading < self.near] = 0
            #filter_w = jt.ones([13, 13], "float32")/169
            #shading = conv_for_image(shading, filter_w, 1)
            imsave("D:\Render\jrender\data\\results\\temp\\shading.bmp", shading)
            shading = shading.unsqueeze(2)

        return shading

    def generate_DepthMap(self, light: Light):
        MRT = self.MRT
        face_vertices = MRT.worldcoords
        if light.type == "point" or light.type == "area":
            eye = light.position
            direction = light.direction
            viewing_angle = light.viewing_angle
            proj_vertices = self.vp_transform(vertices=face_vertices, eye=eye,
                                              camera_direction=direction, viewing_angle=viewing_angle, camera_mode="look", perspective=True, up=light.up)
            self.Rasterize(proj_vertices, proj_vertices, fill_back=light.fillback)
            DM = self.rasterize.save_vars[4][:, 0, :, :].squeeze(0)
            DM[DM > light.far] = light.far + 1
            imsave("D:\Render\jrender\data\\results\\temp\\DM.jpg", DM)
            return DM

        elif light.type == "directional":
            direction = light.direction
            eye = light.position
            viewing_scale = light.viewing_scale
            proj_vertices = self.vp_transform(vertices=face_vertices, eye=eye,
                                              camera_direction=direction, viewing_scale=viewing_scale, camera_mode="look", perspective=False, up=light.up)
            DM = self.Rasterize(proj_vertices, proj_vertices, fill_back=light.fillback)[:, :, 2]
            DM[DM > light.far] = light.far + 1
            return DM
            #temp = self.Rasterize(proj_vertices, proj_vertices)[:, :, 2]
            #imsave("D:\Render\jrender\data\\results\\temp\\DM.jpg", temp[:,::-1])

        return None

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
        if self.proj_vertices_update == True or self.GeometryDesc.proj_v_update == True:
            self._proj_vertices = self.MRT.worldcoords
            self._proj_vertices = self.vp_transform(self._proj_vertices)
            self.proj_vertices_update = False
            self.GeometryDesc.proj_v_update = False
        return self._proj_vertices

    @property
    def world_buffer(self):
        if self.world_buffer_update == True or self.GeometryDesc.wcoord_update == True:
            face_normals = jt.matmul(self.MRT.normals.unsqueeze(2), self.view_rotate_m()).squeeze(2)
            self._normal_buffer = self.Rasterize(
                self.proj_vertices, face_normals)
            aggrs_info = self.rasterize.save_vars[4]
            #alpha = aggrs_info[:, 1, :, :].squeeze(0) == -1
            z = aggrs_info[:, 0, :, :].squeeze(0)
            self._faces_ind_buffer = aggrs_info[:, 1, :, :].squeeze(0).int32()
            #z[alpha] = 0
            #imsave("D:\Render\jrender\data\\results\\temp\\normal_buffer.jpg", (self._normal_buffer))
            image_size = self.rasterize.image_size
            x = jt.repeat((2*jt.arange(0, image_size)+1)/image_size-1, [image_size, 1])
            y = x[::, ::-1].transpose()
            width = math.tan(self.viewing_angle/180.*math.pi)
            self._world_buffer = jt.stack([x*z*width, y*z*width, z], dim=2)

            self.normal_buffer_update = False
            self.world_buffer_update = False
            self.GeometryDesc.wcoord_update = False
            self.GeometryDesc.normal_update = False
        return self._world_buffer

    @property
    def normal_buffer(self):
        if self.normal_buffer_update == True or self.GeometryDesc.normal_update == True:
            face_normals = jt.matmul(self.MRT.normals.unsqueeze(2), self.view_rotate_m()).squeeze(2)
            self._normal_buffer = self.Rasterize(
                self.proj_vertices, face_normals)
            self.normal_buffer_update = False
            self.GeometryDesc.normal_update = False
        return self._normal_buffer

    @property
    def KD_buffer(self):
        if self.KD_buffer_update == True or self.MaterialDesc.KD_update == True:
            KD = self.MRT.KD
            self._KD_buffer = self.Rasterize(
                self.proj_vertices, KD, texture_type="surface")
            self.KD_buffer_update = False
            self.MaterialDesc.KD_update = False
        return self._KD_buffer

    @property
    def albedo_buffer(self):
        if self.MaterialDesc.PBR == False:
            return self._albedo_buffer
        if self.albedo_buffer_update == True or self.MaterialDesc.albedo_update == True:
            albedo = self.MRT.albedo
            self._albedo_buffer = self.Rasterize(
                self.proj_vertices, albedo)
            self.albedo_buffer_update = False
            self.MaterialDesc.albedo_update = False
        return self._albedo_buffer

    @property
    def metallic_roughness_buffer(self):
        if self.MaterialDesc.PBR == False:
            return self._metallic_roughness_buffer
        if self.metallic_roughness_buffer_update == True or self.MaterialDesc.metallic_roughness_update == True:
            metallic_roughness = self.MRT.metallic_roughness
            #metallic_roughness = jt.concat([metallic_roughness, jt.ones([
            #    metallic_roughness.shape[0], metallic_roughness.shape[1], 1], "float32")], dim=2)
            self._metallic_roughness_buffer = self.Rasterize(
                self.proj_vertices, metallic_roughness)
            self.metallic_roughness_buffer_update = False
            self.MaterialDesc.metallic_roughness_update = False
        return self._metallic_roughness_buffer

    @property
    def faces_ind_buffer(self):
        return self._faces_ind_buffer

    @property
    def lights(self):
        if self.IlluminationDesc.light_update == True:
            self._lights = copy.deepcopy(self.IlluminationDesc.lights)
            self.lights_transform = True
        if self.lights_transform == True:
            for light in self._lights:
                if light.shadow:
                    light.DepthMap = Texture(self.generate_DepthMap(light))
                light.direction = jt.matmul(jt.array(light.direction).unsqueeze(0),
                                            self.view_rotate_m()).numpy().tolist()
                light.position = jt.matmul(jt.array(light.position).unsqueeze(
                    0)-jt.array(self.eye).unsqueeze(0), self.view_rotate_m()).numpy().tolist()
                light.up = jt.matmul(jt.array(light.up).unsqueeze(0),
                                     self.view_rotate_m()).numpy().tolist()

            self.lights_transform = False
            self.IlluminationDesc.light_update = False
        return self._lights

    def SSR(self, color):
        ssr_faces = []
        for obj in self.MaterialDesc.objects:
            if obj.reflection_type == "mirror":
                i = self.GeometryDesc.name_dic[obj.material_name]
                ssr_faces += self.GeometryDesc.obj_faces[f"{i}"]
        if len(ssr_faces) == 0:
            return color
        ssr_faces = jt.array(ssr_faces).int32()
        world_buffer = self.world_buffer
        normal_buffer = self.normal_buffer
        faces_ind_buffer = self.faces_ind_buffer
        width = math.tan(self.viewing_angle/180.*math.pi)
        time1 = time.time()
        reflect = SSR_cuda_naive2(color, world_buffer, normal_buffer, faces_ind_buffer, ssr_faces, width, self.far, step=1)
        reflect = FXAA_cuda(reflect)
        time2 = time.time()
        print(time2 - time1)
        imsave("D:\Render\jrender\data\\results\\temp\\debug.bmp", reflect)
        return color + reflect

    def SSSR(self, color):
        ssr_faces = []
        for obj in self.MaterialDesc.objects:
            if obj.reflection_type == "glossy":
                i = self.GeometryDesc.name_dic[obj.material_name]
                ssr_faces += self.GeometryDesc.obj_faces[f"{i}"]
        if len(ssr_faces) == 0:
            return color        
        ssr_faces = jt.array(ssr_faces).int32()
        world_buffer = self.world_buffer
        normal_buffer = self.normal_buffer
        faces_ind_buffer = self.faces_ind_buffer
        roughness_buffer = self.metallic_roughness_buffer[:, :, 1].unsqueeze(2)
        width = math.tan(self.viewing_angle/180.*math.pi)
        time1 = time.time()
        reflect = SSSR_cuda(color, world_buffer, normal_buffer, roughness_buffer,
                     faces_ind_buffer, ssr_faces, width, self.far, step=1, level_intersect=0, spp=256)
        time2 = time.time()
        print(time2 - time1)
        imsave("D:\Render\jrender\data\\results\\temp\\reflect.bmp", reflect)
        imsave("D:\Render\jrender\data\\results\\temp\\sssr.bmp", color + reflect)
        reflect = jt.clamp(reflect,0,1).numpy() * 255
        reflect = numpy.uint8(reflect)
        reflect = cv2.bilateralFilter(reflect,d=10,sigmaColor=20,sigmaSpace=10)
        reflect = jt.array(reflect)/255.
        color = color + reflect
        imsave("D:\Render\jrender\data\\results\\output_render\\SSSR.jpg", color[:,::-1,:])
        #imsave("D:\Render\jrender\data\\results\\temp\\sssr_bl.bmp", out_reflect)
        return color
        

    def SSAO(self, color):
        depth = self.world_buffer[:, :, 2]
        normal_buffer = self.normal_buffer
        faces_ind_buffer = self.faces_ind_buffer
        width = math.tan(self.viewing_angle/180.*math.pi)
        occlusion =  SSAO_cuda(depth, faces_ind_buffer, normal_buffer, width, sample_num=1024, sample_range_r=0.45)
        ambient = 1 - occlusion
        imsave("D:\Render\jrender\data\\results\\temp\\ambient.jpg", ambient[:,::-1])
        filter_w = jt.ones([5, 5], "float32")/25
        occlusion = conv_for_image(occlusion, filter_w, 0)
        color *= ambient.unsqueeze(2)
        return color

    def SSDO(self, color):
        depth = self.world_buffer[:, :, 2]
        normal_buffer = self.normal_buffer
        faces_ind_buffer = self.faces_ind_buffer
        width = math.tan(self.viewing_angle/180.*math.pi)
        color = SSDO_cuda(color, depth, faces_ind_buffer, normal_buffer, width, sample_num=1024, sample_range_r=0.3)
        #imsave("D:\Render\jrender\data\\results\\temp\\rand.jpg", color)
        return color

