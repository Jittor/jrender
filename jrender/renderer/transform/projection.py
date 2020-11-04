import jittor as jt

def projection(vertices, K, R, t, dist_coeffs, orig_size, eps=1e-9):
    '''
    Calculate projective transformation of vertices given a projection matrix
    Input parameters:
    K: batch_size * 3 * 3 intrinsic camera matrix
    R, t: batch_size * 3 * 3, batch_size * 1 * 3 extrinsic calibration parameters
    dist_coeffs: vector of distortion coefficients
    orig_size: original size of image captured by the camera
    Returns: For each point [X,Y,Z] in world coordinates [u,v,z] where u,v are the coordinates of the projection in
    pixels and z is the depth
    '''

    # instead of P*x we compute x'*P'
    vertices = jt.matmul(vertices, R.transpose((0,2,1))[0]) + t
    x, y, z = vertices[:, :, 0], vertices[:, :, 1], vertices[:, :, 2]
    x_ = x / (z + eps)
    y_ = y / (z + eps)

    # Get distortion coefficients from vector
    k1 = dist_coeffs[:,0].unsqueeze(1)
    k2 = dist_coeffs[:,1].unsqueeze(1)
    p1 = dist_coeffs[:,2].unsqueeze(1)
    p2 = dist_coeffs[:,3].unsqueeze(1)
    k3 = dist_coeffs[:,4].unsqueeze(1)

    # we use x_ for x' and x__ for x'' etc.
    x_2 = x_.sqr()
    y_2 = y_.sqr()
    r = jt.sqrt(x_2 + y_2)
    r2 = r.sqr()
    r4 = r2.sqr()
    r6 = r4 * r2

    tmp = k1*(r2) + k2*(r4) + k3*(r6) + 1
    x__ = x_* tmp + 2*p1*x_*y_ + p2*(r2 + 2*x_2)
    y__ = y_* tmp + p1*(r2 + 2*y_2) + 2*p2*x_*y_

    vertices = jt.stack([x__, y__, jt.ones(z.shape)], dim=-1)
    vertices = jt.matmul(vertices, K.transpose((0,2,1))[0])
    u, v = vertices[:, :, 0], vertices[:, :, 1]
    v = orig_size - v
    # map u,v from [0, img_size] to [-1, 1] to use by the renderer
    u = 2 * (u - orig_size / 2.) / orig_size
    v = 2 * (v - orig_size / 2.) / orig_size
    vertices = jt.stack([u, v, z], dim=-1)
    return vertices