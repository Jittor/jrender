import jittor as jt

def projection(vertices, P, dist_coeffs, orig_size):
    '''
    Calculate projective transformation of vertices given a projection matrix
    P: 3x4 projection matrix
    dist_coeffs: vector of distortion coefficients
    orig_size: original size of image captured by the camera
    '''
    
    vertices = jt.concat([vertices, jt.ones_like(vertices[:,:,0].unsqueeze(-2))], dim=-1)
    vertices = jt.matmul(vertices, P.transpose((0,2,1))[0])
    x, y, z = vertices[:, :, 0], vertices[:, :, 1], vertices[:, :, 2]
    x_ = x / (z + 1e-5)
    y_ = y / (z + 1e-5)

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

    x__ = 2 * (x__ - orig_size / 2.) / orig_size
    y__ = 2 * (y__ - orig_size / 2.) / orig_size
    vertices = jt.stack([x__, y__, z], dim=-1)
    return vertices