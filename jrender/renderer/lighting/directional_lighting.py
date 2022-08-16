import numpy as np
import jittor as jt
from jittor import nn

def GGX(N, H, roughness):
    a = roughness * roughness
    a2 = a * a
    if len(N.shape)==4:
        NdotH = nn.relu(jt.sum(N * H, dim = 3))
        NdotH2 = (NdotH * NdotH).unsqueeze(3)
    else:
        NdotH = nn.relu(jt.sum(N * H, dim = 2))
        NdotH2 = (NdotH * NdotH).unsqueeze(2)

    num = a2
    denom = (NdotH2 * (a2 - 1.0) + 1.0)
    denom = 3.1415 * denom * denom

    return num / denom

def SchlickGGX(NdotV, roughness):
    r = roughness + 1.0
    k = (r * r) / 8.0
    if len(roughness.shape)==4:
        NdotV = NdotV.unsqueeze(3)
    else:
        NdotV = NdotV.unsqueeze(2)
    num = NdotV
    denom = NdotV * (1.0 - k) + k
    
    return num / denom

def GeometrySmith(N, V, L, roughness):
    if len(N.shape)==4:
        NdotV = nn.relu(jt.sum(N * V, dim = 3))
        NdotL = nn.relu(jt.sum(N * L, dim = 3))
        ggx2 = SchlickGGX(NdotV, roughness)
        ggx1 = SchlickGGX(NdotL, roughness)
    else:
        NdotV = nn.relu(jt.sum(N * V, dim = 2))
        NdotL = nn.relu(jt.sum(N * L, dim = 2))
        ggx2 = SchlickGGX(NdotV, roughness)
        ggx1 = SchlickGGX(NdotL, roughness)
    
    return ggx1 * ggx2

def fresnelSchlick(cosTheta, F0):
    
    if len(F0.shape)==4:
        return F0 + (1.0 - F0) * jt.pow(1.0 - cosTheta,5).unsqueeze(3)
    else:   
        return F0 + (1.0 - F0) * jt.pow(1.0 - cosTheta,5).unsqueeze(2)

def directional_lighting(diffuseLight, specularLight, normals, light_intensity=0.5, light_color=(1,1,1), light_direction=(0,1,0), positions=None, eye=None, with_specular=False, metallic_textures=None, roughness_textures=None, Gbuffer="None", transform=None):
    eye = jt.array(eye, "float32")
    light_color = jt.array(light_color, "float32")
    light_direction = jt.normalize(jt.array(light_direction, "float32"), dim=0)

    if len(light_color.shape) == 1:
        light_color = light_color.unsqueeze(0)
    if len(light_direction.shape) == 1:
        light_direction = light_direction.unsqueeze(0)

    if len(normals.shape)==4:
        cosine = nn.relu(jt.sum(normals * light_direction, dim=3))
        positions=positions.unsqueeze(2)                  
    else:
        cosine = nn.relu(jt.sum(normals * light_direction, dim=2))
    if with_specular and len(normals.shape)!=4:
        if len(metallic_textures.shape) == 4:
            total = metallic_textures.shape[2] * 1.0
            metallic_textures = jt.sum(metallic_textures, dim=2) / total                      
            roughness_textures = jt.sum(roughness_textures, dim=2) / total
        elif len(metallic_textures.shape) == 6:
            total = metallic_textures.shape[2] * metallic_textures.shape[3] * metallic_textures.shape[4] * 1.0
            metallic_textures = jt.sum(metallic_textures, dim=2)
            metallic_textures = jt.sum(metallic_textures, dim=2)
            metallic_textures = jt.sum(metallic_textures, dim=2)
            metallic_textures = metallic_textures / total
            roughness_textures = jt.sum(roughness_textures, dim=2)
            roughness_textures = jt.sum(roughness_textures, dim=2)
            roughness_textures = jt.sum(roughness_textures, dim=2)
            roughness_textures = roughness_textures / total

    #Microfacet model
    if with_specular and (eye is not None) and (positions is not None) and (metallic_textures is not None) and (roughness_textures is not None):
        N = normals
        if len(normals.shape) == 4:
            if len(eye.shape) == 2:
                eye = eye.unsqueeze(1).unsqueeze(2)
            V = jt.normalize(eye - positions,dim=3)
            L = light_direction
            H = jt.normalize(V + L,dim=3)
        else:
            if len(eye.shape) == 2:
                eye = eye.unsqueeze(1)
            V = jt.normalize(eye - positions,dim=2)
            L = light_direction
            H = jt.normalize(V + L,dim=2)
        #Default Setting
        metallic = metallic_textures
        roughness = roughness_textures
        F0 = jt.array((0.04, 0.04, 0.04), "float32")
        albedo = jt.array((1.0, 1.0, 1.0), "float32")
        if len(normals.shape)==4:
            F0 = F0.unsqueeze(0).unsqueeze(1).unsqueeze(2) * (1 - metallic) + albedo.unsqueeze(0).unsqueeze(1).unsqueeze(2) * metallic             
            radiance = light_intensity * (light_color.unsqueeze(1).unsqueeze(2) * cosine.unsqueeze(3))
        else:
            F0 = F0.unsqueeze(0).unsqueeze(1) * (1 - metallic) + albedo.unsqueeze(0).unsqueeze(1) * metallic            
            radiance = light_intensity * (light_color.unsqueeze(1) * cosine.unsqueeze(2))

        #Cook-Torrance BRDF
        NDF = GGX(N, H, roughness)
        G = GeometrySmith(N, V, L, roughness)
        if len(normals.shape)==4:
            F = fresnelSchlick(nn.relu(jt.sum(H * V, dim=3)), F0)
        else :
            F = fresnelSchlick(nn.relu(jt.sum(H * V, dim=2)), F0)
        KS = F
        KD = 1.0 - KS
        KD *= (1.0 - metallic)
        
        diffuseLight += KD * radiance
        numerator = NDF * G * F
        if len(normals.shape)==4:
            denominator = (4.0 * nn.relu(jt.sum(N * V, dim=3)) * nn.relu(jt.sum(N * L, dim=3))).unsqueeze(3)
        else:
            denominator = (4.0 * nn.relu(jt.sum(N * V, dim=2)) * nn.relu(jt.sum(N * L, dim=2))).unsqueeze(2)
        specular = numerator / jt.clamp(denominator, 0.01)
        specularLight += specular * radiance
    else:
        if len(normals.shape)==4:
            diffuseLight += light_intensity * (light_color.unsqueeze(1).unsqueeze(2) * cosine.unsqueeze(3))
        else:
            diffuseLight += light_intensity * (light_color.unsqueeze(1) * cosine.unsqueeze(2))
    if Gbuffer == "normal":
        specularLight *= 0.0
        diffuseLight = normals * 0.5 + 0.5
    elif Gbuffer == "depth":
        specularLight *= 0.0
        viewpos = transform.tranpos(positions)
        diffuseLight = viewpos/jt.max(viewpos[...,2])
        diffuseLight[...,0] = viewpos[...,2]/jt.max(viewpos[...,2])
        diffuseLight[...,1] = viewpos[...,2]/jt.max(viewpos[...,2])
    return [diffuseLight, specularLight]
