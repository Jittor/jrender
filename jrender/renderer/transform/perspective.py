import math
import jittor as jt

def perspective(vertices, angle=30.):
    '''
    Compute perspective distortion from a given angle
    '''
    if len(vertices.shape) != 3:
        raise ValueError('vertices Tensor should have 3 dimensions')
    angle = jt.array([angle / 180 * math.pi]).float32()
    width = jt.tan(angle)
    width = width.unsqueeze(-1) 
    z = vertices[:, :, 2]
    x = vertices[:, :, 0] / z / width
    y = vertices[:, :, 1] / z / width
    vertices = jt.contrib.concat((x.unsqueeze(-1),y.unsqueeze(-1),z.unsqueeze(-1)), dim=2)
    return vertices