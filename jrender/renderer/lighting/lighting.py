import numpy as np
import jittor as jt
from jittor import nn

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
        light += intensity_directional * (color_directional.unsqueeze(1) * cos.unsqueeze(2))
    # apply
    light = light.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2)
    textures *= light
    return textures


class AmbientLighting(nn.Module):
    def __init__(self, light_intensity=0.5, light_color=(1,1,1)):
        super(AmbientLighting, self).__init__()

        self.light_intensity = light_intensity
        self.light_color = light_color

    def execute(self, light):
        return ambient_lighting(light, self.light_intensity, self.light_color)



class DirectionalLighting(nn.Module):
    def __init__(self, light_intensity=0.5, light_color=(1,1,1), light_direction=(0,1,0), Gbuffer='None', transform=None):
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


class Lighting(nn.Module):
    def __init__(self, light_mode='surface',
                 intensity_ambient=0.5, color_ambient=[1,1,1],
                 intensity_directionals=0.5, color_directionals=[1,1,1],
                 directions=[0,1,0], Gbuffer='None', transform=None):
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
            diffuseLight = jt.zeros(mesh.faces.shape)
            specularLight = jt.zeros(mesh.faces.shape)
            diffuseLight = self.ambient(diffuseLight)
            for directional in self.directionals:
                [diffuseLight, specularLight] = directional(diffuseLight, specularLight, mesh.surface_normals, (jt.sum(mesh.face_vertices, dim=2) / 3.0), eyes, mesh.with_specular, mesh.metallic_textures, mesh.roughness_textures) 
            if len(mesh.textures.shape) == 4: 
                mesh.textures = jt.clamp(mesh.textures * diffuseLight.unsqueeze(2) + jt.ones_like(mesh.textures) * specularLight.unsqueeze(2), 0.0, 1.0) 
            elif len(mesh.textures.shape) == 6:
                mesh.textures = jt.clamp(mesh.textures * diffuseLight.unsqueeze(2).unsqueeze(2).unsqueeze(2) + jt.ones_like(mesh.textures) * specularLight.unsqueeze(2).unsqueeze(2).unsqueeze(2), 0.0, 1.0)

        elif self.light_mode == 'vertex':
            diffuseLight = jt.zeros(mesh.vertices.shape)
            specularLight = jt.zeros(mesh.vertices.shape)
            diffuseLight = self.ambient(diffuseLight)
            for directional in self.directionals:
                [diffuseLight, specularLight] = directional(diffuseLight, specularLight, mesh.vertex_normals, mesh.vertices, eyes, mesh.with_specular, mesh.metallic_textures, mesh.roughness_textures)
            if len(mesh.textures.shape) == 4:
                mesh.textures = jt.clamp(mesh.textures * diffuseLight.unsqueeze(2) + jt.ones_like(mesh.textures) * specularLight.unsqueeze(2), 0.0, 1.0)
            elif len(mesh.textures.shape) == 6:
                mesh.textures = jt.clamp(mesh.textures * diffuseLight.unsqueeze(2).unsqueeze(2).unsqueeze(2) + jt.ones_like(mesh.textures) * specularLight.unsqueeze(2).unsqueeze(2).unsqueeze(2), 0.0, 1.0) 

        return mesh
