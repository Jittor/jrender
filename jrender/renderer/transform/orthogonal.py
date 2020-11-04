import jittor as jt

def orthogonal(vertices, scale):
    '''
    Compute orthogonal projection from a given angle
    To find equivalent scale to perspective projection
    set scale = focal_pixel / object_depth  -- to 0~H/W pixel range
              = 1 / ( object_depth * tan(half_fov_angle) ) -- to -1~1 pixel range
    '''
    if (len(vertices.shape) != 3):
        raise ValueError('vertices Tensor should have 3 dimensions')
    z = vertices[:, :, 2]
    x = vertices[:, :, 0] * scale
    y = vertices[:, :, 1] * scale
    vertices = jt.stack((x,y,z), dim=2)
    return vertices