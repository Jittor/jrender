import numpy as np
import jittor as jt
from jittor import nn

def GGX(N, H, roughness):
    a = roughness * roughness
    a2 = a * a
    NdotH = nn.relu(jt.sum(N * H, dim = 2))
    NdotH2 = NdotH * NdotH

    num = a2
    denom = (NdotH2 * (a2 - 1.0) + 1.0)
    denom = 3.1415 * denom * denom

    return num / denom

def SchlickGGX(NdotV, roughness):
    r = roughness + 1.0
    k = (r * r) / 8.0

    num = NdotV
    denom = NdotV * (1.0 - k) + k
    
    return num / denom

def GeometrySmith(N, V, L, roughness):
    NdotV = nn.relu(jt.sum(N * V, dim = 2))
    NdotL = nn.relu(jt.sum(N * L, dim = 2))
    ggx2 = SchlickGGX(NdotV, roughness)
    ggx1 = SchlickGGX(NdotL, roughness)
    
    return ggx1 * ggx2

def fresnelSchlick(cosTheta, F0):
    return F0 + (1.0 - F0).unsqueeze(0).unsqueeze(1) * jt.pow(1.0 - cosTheta,5).unsqueeze(2)

def directional_lighting(diffuseLight, specularLight, normals, light_intensity=0.5, light_color=(1,1,1), light_direction=(0,1,0), positions=None, eye=None):
    eye = jt.array(eye, "float32")
    light_color = jt.array(light_color, "float32")
    light_direction = jt.array((0.0,0.707,0.707), "float32")
    #light_direction = jt.normalize(light_direction)

    if len(light_color.shape) == 1:
        light_color = light_color.unsqueeze(0)
    if len(light_direction.shape) == 1:
        light_direction = light_direction.unsqueeze(0)
    
    cosine = nn.relu(jt.sum(normals * light_direction, dim=2))
   
    #Microfacet model
    if(eye is not None) and (positions is not None):
        N = normals
        V = jt.normalize(eye - positions,dim=2)
        L = light_direction
        H = jt.normalize(V + L,dim=2)

        #Default Setting
        metallic = 0.3
        roughness = 0.5
        F0 = jt.array((0.04, 0.04, 0.04), "float32")
        albedo = jt.array((1.0, 1.0, 1.0), "float32")

        F0 = F0 * (1 - metallic) + albedo * metallic
        radiance = light_intensity * (light_color.unsqueeze(1) * cosine.unsqueeze(2))

        #Cook-Torrance BRDF
        NDF = GGX(N, H, roughness).unsqueeze(2)
        G = GeometrySmith(N, V, L, roughness).unsqueeze(2)
        F = fresnelSchlick(nn.relu(jt.sum(H * V, dim=2)), F0)
        
        KS = F
        KD = 1.0 - KS
        KD *= (1.0 - metallic)

        diffuseLight += KD * radiance
        numerator = NDF * G * F
        denominator = (4.0 * nn.relu(jt.sum(N * V, dim=2)) * nn.relu(jt.sum(N * L, dim=2))).unsqueeze(2)
        specular = numerator / jt.clamp(denominator, 0.01)
        specularLight += specular * radiance * 20.0
    else:
        diffuseLight += light_intensity * (light_color.unsqueeze(1) * cosine.unsqueeze(2))

    return [diffuseLight, specularLight]
