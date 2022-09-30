from os import terminal_size
import jittor as jt
from jittor import nn
from jrender.renderer.dr.softras.soft_rasterize import SoftRasterizeFunction
from jrender.io.utils.load_textures import _load_textures_for_softras
from jrender.renderer.utils.gaussian_blur import gaussian_blur
from jrender.renderer.utils.ToStretchMap import computeStretchMap
from . import *


def lighting(faces, textures, intensity_ambient=0.5, intensity_directional=0.5,
             color_ambient=(1, 1, 1), color_directional=(1, 1, 1), direction=(0, 1, 0)):

    bs, nf = faces.shape[:2]

    # arguments
    # make sure to convert all inputs to float tensors
    color_ambient = jt.array(color_ambient, "float32")
    color_directional = jt.array(color_directional, "float32")
    direction = jt.array(direction, "float32")

    if len(color_ambient.shape) == 1:
        color_ambient = color_ambient.unsqueeze(0)
    if len(color_directional.shape) == 1:
        color_directional = color_directional.unsqueeze(0)
    if len(direction.shape) == 1:
        direction = direction.unsqueeze(0)

    # create light
    light = jt.zeros((bs, nf, 3), "float32")

    # ambient light
    if intensity_ambient != 0:
        light += intensity_ambient * color_ambient.unsqueeze(1)

    # directional light
    if intensity_directional != 0:
        faces = faces.reshape((bs * nf, 3, 3))
        v10 = faces[:, 0] - faces[:, 1]
        v12 = faces[:, 2] - faces[:, 1]

        normals = jt.normalize(jt.cross(v10, v12), eps=1e-5)
        normals = normals.reshape((bs, nf, 3))

        if len(direction.shape) == 2:
            direction = direction.unsqueeze(1)
        cos = nn.relu(jt.sum(normals * direction, dim=2))
        # may have to verify that the next line is correct
        light += intensity_directional * \
            (color_directional.unsqueeze(1) * cos.unsqueeze(2))
    # apply
    light = light.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2)
    textures *= light
    return textures


class AmbientLighting(nn.Module):
    def __init__(self, light_intensity=0.5, light_color=(1, 1, 1)):
        super(AmbientLighting, self).__init__()

        self.light_intensity = light_intensity
        self.light_color = light_color

    def execute(self, light):
        return ambient_lighting(light, self.light_intensity, self.light_color)


class DirectionalLighting(nn.Module):
    def __init__(self, light_intensity=0.5, light_color=(1, 1, 1), light_direction=(0, 1, 0), Gbuffer='None', transform=None):
        super(DirectionalLighting, self).__init__()

        self.light_intensity = light_intensity
        self.light_color = light_color
        self.light_direction = light_direction
        self.Gbuffer = Gbuffer
        self.transform = transform

    def execute(self, diffuseLight, specularLight, normals, positions=None, eye=None, with_specular=False, metallic_textures=None, roughness_textures=None):
        return directional_lighting(diffuseLight, specularLight, normals,
                                    self.light_intensity, self.light_color,
                                    self.light_direction, positions, eye, with_specular, metallic_textures, roughness_textures, self.Gbuffer, self.transform)


# subsurface scattering based on the texture_space
def SSS(diffuseLight, specular, mesh):
    irradiance = (mesh.textures*diffuseLight).sqrt()
    # rasterize to texture_space
    image_size = 1024
    rasterize = SoftRasterizeFunction(image_size=image_size,
                                      background_color=[0, 0, 0], near=1, far=100,
                                      fill_back=True, eps=1e-5,
                                      sigma_val=1e-5, dist_func='euclidean', dist_eps=1e-5,
                                      gamma_val=1e-4, aggr_func_rgb='softmax', aggr_func_alpha='prod',
                                      texture_type='surface')
    worldcoord_textures = (mesh.face_vertices+1)/2*255.
    coords = (mesh.face_texcoords-0.5)*2
    coords = jt.concat([coords, jt.full(
        (coords.shape[0], coords.shape[1], coords.shape[2], 1), 1.5)], dim=3)
    Worldcoord_Map = rasterize(coords, worldcoord_textures)
    Worldcoord_Map = jt.transpose(Worldcoord_Map.squeeze(0)[:3, :, :], (1, 2, 0))
    irradiance_Map = rasterize(coords, irradiance)
    irradiance_Map = jt.transpose(irradiance_Map.squeeze(0)[:3, :, :], (1, 2, 0))
    specular_Map = rasterize(coords, specular)
    specular_Map = jt.transpose(specular_Map.squeeze(0)[:3, :, :], (1, 2, 0))

    # compute stretch_map
    stretchU, stretchV = computeStretchMap(Worldcoord_Map*7)
    stretchU = jt.clamp(stretchU, 0.0, 1.0)
    stretchV = jt.clamp(stretchV, 0.0, 1.0)

    # rgb blur

    Gaussian_basis = [0, 16, 64, 256]

    diffuse_sqrt = irradiance_Map.clone()
    final_map = jt.zeros_like(irradiance_Map)
    tap_num = 7

    irradiance_Map_R = irradiance_Map[:, :, 0]
    irradiance_Map_G = irradiance_Map[:, :, 1]
    irradiance_Map_B = irradiance_Map[:, :, 2]

    Gaussian_weight_r = [0.2, 0.3, 0.5]
    Gaussian_weight_g = [0.5, 0.4, 0.1]
    Gaussian_weight_b = [0.6, 0.4, 0.0]
    Gaussian_weight = [Gaussian_weight_r, Gaussian_weight_g, Gaussian_weight_b]

    irradiance_rgb = []
    irradiance_rgb.append(irradiance_Map_R)
    irradiance_rgb.append(irradiance_Map_G)
    irradiance_rgb.append(irradiance_Map_B)

    for j in range(3):
        stretchU_cpy = stretchU.clone()
        stretchV_cpy = stretchV.clone()
        irradiance_basis_rgb = []
        for i in range(len(Gaussian_basis)-1):
            v = Gaussian_basis[i+1]-Gaussian_basis[i]
            stretchU_cpy = gaussian_blur(stretchU_cpy, tap_num, v, stretchU_cpy, 1)
            irradiance_rgb[j] = gaussian_blur(irradiance_rgb[j], tap_num, v, stretchU_cpy, 1)
            stretchV_cpy = gaussian_blur(stretchV_cpy, tap_num, v, stretchV_cpy, 0)
            irradiance_rgb[j] = gaussian_blur(irradiance_rgb[j], tap_num, v, stretchV_cpy, 0)
            irradiance_basis_rgb.append(irradiance_rgb[j].clone())
        for k, image in enumerate(irradiance_basis_rgb):
            final_map[:, :, j] += Gaussian_weight[j][k]*image

    final_map *= diffuse_sqrt
    final_map += specular_Map

    final_map = final_map[::-1, :, :]
    is_update = jt.ones((mesh.faces.shape[1])).int()
    final_textures = jt.ones((mesh.faces.shape[1], mesh.texture_res**2, 3), dtype="float32")
    final_textures = _load_textures_for_softras(
        final_map, mesh.face_texcoords.squeeze(0), final_textures, is_update).unsqueeze(0)

    return final_textures


class Lighting(nn.Module):
    def __init__(self, light_mode='surface',
                 intensity_ambient=0.5, color_ambient=[1, 1, 1],
                 intensity_directionals=0.5, color_directionals=[1, 1, 1],
                 directions=[0, 1, 0], Gbuffer='None', transform=None):
        super(Lighting, self).__init__()

        if light_mode not in ['surface', 'vertex']:
            raise ValueError('Lighting mode only support surface and vertex')

        self.Gbuffer = Gbuffer
        self.transform = transform
        self.light_mode = light_mode
        self.ambient = AmbientLighting(intensity_ambient, color_ambient)
        self.directionals = nn.ModuleList([DirectionalLighting(intensity_directionals,
                                                               color_directionals,
                                                               directions, Gbuffer, transform)])

    def execute(self, mesh, eyes=None):
        if self.Gbuffer == "albedo":
            return mesh
        if self.Gbuffer == "normal" or self.Gbuffer == "depth":
            mesh.textures = jt.ones_like(mesh.textures)
        if self.light_mode == 'surface':
            if mesh.normal_textures is not None:
                diffuseLight = jt.zeros(mesh.textures.shape)
                specularLight = jt.zeros(mesh.textures.shape)
                diffuseLight = self.ambient(diffuseLight)
                for directional in self.directionals:
                    [diffuseLight, specularLight] = directional(diffuseLight, specularLight, mesh.surface_ResNormals, (jt.sum(
                        mesh.face_vertices, dim=2) / 3.0), eyes, mesh.with_specular, mesh.metallic_textures, mesh.roughness_textures)
            else:
                diffuseLight = jt.zeros(mesh.faces.shape)
                specularLight = jt.zeros(mesh.faces.shape)
                diffuseLight = self.ambient(diffuseLight)
                for directional in self.directionals:
                    [diffuseLight, specularLight] = directional(diffuseLight, specularLight, mesh.surface_normals, (jt.sum(
                        mesh.face_vertices, dim=2) / 3.0), eyes, mesh.with_specular, mesh.metallic_textures, mesh.roughness_textures)
                diffuseLight = diffuseLight.unsqueeze(2)
                specularLight = specularLight.unsqueeze(2)
            if len(mesh.textures.shape) == 4 and mesh.with_SSS:
                mesh.textures = jt.clamp(
                    SSS(diffuseLight, specularLight, mesh), 0.0, 1.0)
            elif len(mesh.textures.shape) == 4 and mesh.with_SSS == False:
                mesh.textures = jt.clamp(mesh.textures * diffuseLight +
                                         jt.ones_like(mesh.textures) * specularLight, 0.0, 1.0)
            elif len(mesh.textures.shape) == 6:
                mesh.textures = jt.clamp(mesh.textures * diffuseLight.unsqueeze(2).unsqueeze(2) +
                                         jt.ones_like(mesh.textures) * specularLight.unsqueeze(2).unsqueeze(2), 0.0, 1.0)

        elif self.light_mode == 'vertex':
            diffuseLight = jt.zeros(mesh.vertices.shape)
            specularLight = jt.zeros(mesh.vertices.shape)
            diffuseLight = self.ambient(diffuseLight)
            for directional in self.directionals:
                [diffuseLight, specularLight] = directional(
                    diffuseLight, specularLight, mesh.vertex_normals, mesh.vertices, eyes, mesh.with_specular, mesh.metallic_textures, mesh.roughness_textures)
            if len(mesh.textures.shape) == 4:
                mesh.textures = jt.clamp(mesh.textures * diffuseLight.unsqueeze(2) +
                                         jt.ones_like(mesh.textures) * specularLight.unsqueeze(2), 0.0, 1.0)
            elif len(mesh.textures.shape) == 6:
                mesh.textures = jt.clamp(mesh.textures * diffuseLight.unsqueeze(2).unsqueeze(2).unsqueeze(2) +
                                         jt.ones_like(mesh.textures) * specularLight.unsqueeze(2).unsqueeze(2).unsqueeze(2), 0.0, 1.0)

        return mesh
