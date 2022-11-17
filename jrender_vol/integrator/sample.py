import jittor as jt
from jittor import nn
import numpy as np

# Hierarchical sampling
def sample(N_rays, N_samples, lindisp, perturb, near, far):
    t_vals = jt.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = jt.concat([mids, z_vals[...,-1:]], -1)
        lower = jt.concat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = jt.random(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    return z_vals


# Hierarchical sampling pdf
def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / jt.sum(weights, -1, keepdims=True)
    cdf = jt.cumsum(pdf, -1)
    cdf = jt.concat([jt.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = jt.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = jt.random(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    inds = jt.searchsorted(cdf, u, right=True)
    below = jt.maximum(jt.zeros_like(inds-1), inds-1)
    above = jt.minimum((cdf.shape[-1]-1) * jt.ones_like(inds), inds)
    inds_g = jt.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = jt.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = jt.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom[denom<1e-5] = 1.0
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
