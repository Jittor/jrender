import jittor as jt
from jittor import nn
import numpy as np

def integrator(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=jt.nn.relu: 1.-jt.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = jt.concat([dists, jt.array(np.array([1e10]).astype(np.float32)).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    dists = dists * jt.norm(rays_d.unsqueeze(-2), p=2, dim=-1)

    rgb = jt.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = jt.init.gauss(raw[...,3].shape, raw.dtype) * raw_noise_std
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * jt.cumprod(jt.concat([jt.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = jt.sum(weights.unsqueeze(-1) * rgb, -2)  # [N_rays, 3]

    depth_map = jt.sum(weights * z_vals, -1)
    disp_map = 1./jt.maximum(1e-10 * jt.ones_like(depth_map), depth_map / jt.sum(weights, -1))
    acc_map = jt.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map.unsqueeze(-1))

    return rgb_map, disp_map, acc_map, weights, depth_map
