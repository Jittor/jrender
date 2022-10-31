import jittor as jt
from jittor import nn

# N:[,,3] H:[,,3]
def GGX(N, H, roughness):
    a = roughness * roughness
    a2 = a * a

    NdotH = nn.relu(jt.sum(N * H, dim = 2))
    NdotH2 = (NdotH * NdotH).unsqueeze(2)

    num = a2
    denom = (NdotH2 * (a2 - 1.0) + 1.0)
    denom = 3.1415 * denom * denom

    return num / denom

def SchlickGGX(NdotV, roughness):
    r = roughness + 1.0
    k = (r * r) / 8.0
    NdotV = NdotV.unsqueeze(2)
    num = NdotV
    denom = NdotV * (1.0 - k) + k
    
    return num / denom

def GeometrySmith(N, V, L, roughness):
    NdotV = nn.relu(jt.sum(N * V, dim = 2))
    NdotL = nn.relu(jt.sum(N * L, dim = 2))
    ggx2 = SchlickGGX(NdotV, roughness)
    ggx1 = SchlickGGX(NdotL, roughness)
    
    return ggx1 * ggx2

#cosTheta :[,,]
def fresnelSchlick(cosTheta, F0):
    return F0 + (1.0 - F0) * jt.pow(1.0 - cosTheta,5).unsqueeze(2)
