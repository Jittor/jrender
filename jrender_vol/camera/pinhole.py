import jittor as jt
import numpy as np
from jittor import nn

# Pinhole Camera Rays
def pinhole_get_rays(H, W, focal, c2w, intrinsic = None):
    i, j = jt.meshgrid(jt.linspace(0, W-1, W), jt.linspace(0, H-1, H))
    i = i.t()
    j = j.t()
    if intrinsic is None:
        dirs = jt.stack([-(i-W*.5)/focal, -(j-H*.5)/focal, jt.ones_like(i)], -1).unsqueeze(-2)
    else:
        i+=0.5
        j+=0.5
        dirs = jt.stack([i, j, jt.ones_like(i)], -1).unsqueeze(-2)
        dirs = jt.sum(dirs * intrinsic[:3,:3], -1).unsqueeze(-2)
    # Rotate ray directions from camera frame to the world frame
    rays_d = jt.sum(dirs * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


# Pinhole Camera Ndc Rays
def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t.unsqueeze(-1) * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = jt.stack([o0,o1,o2], -1)
    rays_d = jt.stack([d0,d1,d2], -1)

    return rays_o, rays_d

