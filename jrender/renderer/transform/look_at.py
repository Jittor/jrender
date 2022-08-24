import jittor as jt

def look_at(vertices, eye, at=[0, 0, 0], up=[0, 1, 0]):
    """"Look at" transformation of vertices. The z axis is changed to (at - eye). Original vertices are transformed to the new axis.
    """
    if len(vertices.shape) != 3:
        raise ValueError('vertices Tensor should have 3 dimensions')
    
    at = jt.array(at).float32()
    up = jt.array(up).float32()
    if isinstance(eye, tuple):
        eye = jt.array(list(eye)).float32()
    else:
        eye = jt.array(eye).float32()

    batch_size = vertices.shape[0]
    if len(eye.shape) == 1:
        eye = eye.broadcast([batch_size] + eye.shape)
    if len(at.shape) == 1:
        at = at.broadcast([batch_size] + at.shape)
    if len(up.shape) == 1:
        up = up.broadcast([batch_size] + up.shape)

    # create new axes
    # eps is chosen as 0.5 to match the chainer version
    z_axis = jt.normalize(at - eye, eps=1e-5)
    x_axis = jt.normalize(jt.cross(up, z_axis), eps=1e-5)
    y_axis = jt.normalize(jt.cross(z_axis, x_axis), eps=1e-5)

    # create rotation matrix: [bs, 3, 3]
    r = jt.contrib.concat((x_axis.unsqueeze(1), y_axis.unsqueeze(1), z_axis.unsqueeze(1)), dim=1)
    # apply
    # [bs, nv, 3] -> [bs, nv, 3] -> [bs, nv, 3]
    if vertices.shape != eye.shape:
        eye = eye.unsqueeze(1)
    vertices = vertices - eye

    vertices = jt.matmul(vertices.unsqueeze(2), r.transpose(0,2,1)).squeeze(2)
    return vertices
