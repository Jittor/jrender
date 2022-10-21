import jittor as jt

from .cuda import voxelization as voxelization_cuda

def voxelize_sub1(faces, size, dim):
    bs = faces.size(0)
    nf = faces.size(1)
    if dim == 0:
        faces = faces[:, :, :, [2, 1, 0]]
    elif dim == 1:
        faces = faces[:, :, :, [0, 2, 1]]
    voxels = jt.zeros((bs, size, size, size)).int()
    res = voxelization_cuda.voxelize_sub1(faces, voxels)[0]
    if dim == 0: return res.transpose((0,3,2,1))
    elif dim == 1: return res.transpose((0,1,3,2))
    elif dim == 2: return res

def voxelize_sub2(faces, size):
    bs = faces.size(0)
    nf = faces.size(1)
    voxels = jt.zeros((bs, size, size, size)).int()
    return voxelization_cuda.voxelize_sub2(faces, voxels)[0]

def voxelize_sub3(faces, voxels):
    bs = voxels.size(0)
    vs = voxels.size(1)
    visible = jt.zeros_like(voxels).int32()
    voxels, visible = voxelization_cuda.voxelize_sub3(faces, voxels, visible)

    sum_visible = visible.sum()

    while True:
        voxels, visible = voxelization_cuda.voxelize_sub4(faces, voxels, visible)
        if visible.sum() == sum_visible:
            break
        else:
            sum_visible = visible.sum()
    return 1 - visible


def voxelization(faces, size, normalize=False):
    faces = faces.clone()
    if normalize:
        pass
    else:
        faces *= size

    voxels0 = voxelize_sub1(faces, size, 0)
    voxels1 = voxelize_sub1(faces, size, 1)
    voxels2 = voxelize_sub1(faces, size, 2)
    voxels3 = voxelize_sub2(faces, size)

    voxels = voxels0 + voxels1 + voxels2 + voxels3
    voxels = (voxels > 0).int()
    voxels = voxelize_sub3(faces, voxels)

    return voxels
