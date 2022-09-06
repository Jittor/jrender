import jittor as jt

def look(vertices, eye, direction=[0, 1, 0], up=None, coordinate ="right"):
    """
    "Look" transformation of vertices.
    """
    if len(vertices.shape) != 3:
        raise ValueError('vertices Tensor should have 3 dimensions')

    direction = jt.array(direction).float32()
    if isinstance(eye, tuple):
        eye = jt.array(list(eye)).float32()
    else:
        eye = jt.array(eye).float32()


    if up is None:
        up = jt.array([0, 1, 0]).float32()
    if isinstance(up, list):
        up = jt.array(up).float32()
    if len(eye.shape) == 1:
        eye = eye.unsqueeze(0)
    if len(direction.shape) == 1:
        direction = direction.unsqueeze(0)
    if len(up.shape) == 1:
        up = up.unsqueeze(0)
 

    # create new axes
    z_axis = jt.normalize(direction, eps=1e-5)

    if jt.abs(jt.sum(up * z_axis)) > 1-1e-4:
        raise ValueError("camera_direction and camera_up can not be the same")

    if coordinate == "right":
        x_axis = jt.normalize(jt.cross(up, z_axis), eps=1e-5)
        y_axis = jt.normalize(jt.cross(z_axis, x_axis), eps=1e-5)
    elif coordinate =="left":
        x_axis = jt.normalize(jt.cross(z_axis,up), eps=1e-5)
        y_axis = jt.normalize(jt.cross(x_axis, z_axis), eps=1e-5)

    # create rotation matrix: [bs, 3, 3]
    r = jt.contrib.concat((x_axis.unsqueeze(1), y_axis.unsqueeze(1), z_axis.unsqueeze(1)), dim=1)

    # apply
    # [bs, nv, 3] -> [bs, nv, 3] -> [bs, nv, 3]
    if vertices.shape != eye.shape:
        eye = eye.unsqueeze(1)
    vertices = vertices - eye
    vertices = jt.matmul(vertices.unsqueeze(2), r.transpose(0,2,1)).squeeze(2)
    return vertices
