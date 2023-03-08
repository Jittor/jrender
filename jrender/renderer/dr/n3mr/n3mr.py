import jittor as jt
from jittor import nn
from jittor import Function
from .cuda import rasterize as rasterize_cuda

DEFAULT_IMAGE_SIZE = 256
DEFAULT_ANTI_ALIASING = True
DEFAULT_NEAR = 0.1
DEFAULT_FAR = 100
DEFAULT_EPS = 1e-4
DEFAULT_BACKGROUND_COLOR = (0, 0, 0)

class RasterizeFunction(Function):
    '''
    Definition of differentiable rasterize operation
    Some parts of the code are implemented in CUDA
    Currently implemented only for cuda Tensors
    '''
    def __init__(self, image_size, near, far, eps, background_color, return_rgb=False, return_alpha=False, return_depth=False):
        self.image_size = image_size
        self.near = near
        self.far = far
        self.eps = eps
        self.background_color = background_color
        self.return_rgb = return_rgb
        self.return_alpha = return_alpha
        self.return_depth = return_depth

    def grad(self, grad_rgb_map, grad_alpha_map, grad_depth_map):
        faces, textures, face_index_map, weight_map, depth_map, rgb_map, alpha_map, face_inv_map, sampling_index_map, sampling_weight_map = self.save_vars

        grad_faces = jt.zeros(faces.shape).float32()
        if self.return_rgb:
            grad_textures = jt.zeros(textures.shape).float32()
        else:
            grad_textures = jt.zeros(1).float32()

        # get grad_outputs
        if self.return_rgb:
            if grad_rgb_map is None:
                grad_rgb_map = jt.zeros(rgb_map.shape)
        else:
            grad_rgb_map = jt.zeros(1).float32()
        if self.return_alpha:
            if grad_alpha_map is None:
                grad_alpha_map = jt.zeros(alpha_map.shape)
        else:
            grad_alpha_map = jt.zeros(1).float32()
        if self.return_depth:
            if grad_depth_map is None:
                grad_depth_map = jt.zeros(depth_map.shape)
        else:
            grad_depth_map = jt.zeros(1).float32()

        # backward pass
        grad_faces = self.backward_pixel_map(faces, face_index_map, rgb_map, alpha_map, grad_rgb_map, grad_alpha_map, grad_faces)
        grad_textures = self.backward_textures(face_index_map, sampling_weight_map, sampling_index_map, grad_rgb_map, grad_textures)
        grad_faces = self.backward_depth_map(faces, depth_map, face_index_map, face_inv_map, weight_map, grad_depth_map, grad_faces)[0]
        if self.texture_size is None:
            grad_textures = None
        
        if grad_faces is not None:
            grad_faces.sync()
        if grad_textures is not None:
            grad_textures.sync()

        return grad_faces, grad_textures

    def execute(self, faces, textures):
        self.batch_size, self.num_faces = faces.shape[:2]

        if self.return_rgb:
            self.texture_size = textures.shape[2]
        else:
            # initializing with dummy values
            textures = jt.array([0]).float32()
            self.texture_size = None

        face_index_map = jt.empty((self.batch_size, self.image_size, self.image_size)).int()

        weight_map = jt.empty((self.batch_size, self.image_size, self.image_size, 3))

        depth_map = jt.empty((self.batch_size, self.image_size, self.image_size)) * self.far

        if self.return_rgb:
            rgb_map = jt.empty((self.batch_size, self.image_size, self.image_size, 3)).float()
            sampling_index_map = jt.empty((self.batch_size, self.image_size, self.image_size, 8)).int()
            sampling_weight_map = jt.empty((self.batch_size, self.image_size, self.image_size, 8))
        else:
            rgb_map = jt.zeros(1)
            sampling_index_map = jt.zeros(1).int()
            sampling_weight_map = jt.zeros(1)

        if self.return_alpha:
            alpha_map = jt.empty((self.batch_size, self.image_size, self.image_size))
        else:
            alpha_map = jt.zeros(1)

        if self.return_depth:
            face_inv_map = jt.empty((self.batch_size, self.image_size, self.image_size, 3, 3))
        else:
            face_inv_map = jt.zeros(1)

        # faces -> face_index_map, weight_map, depth_map, face_inv_map
        face_index_map, weight_map, depth_map, face_inv_map = self.forward_face_index_map(faces, face_index_map, weight_map, depth_map, face_inv_map)

        # faces, textures, face_index_map, weight_map, depth_map -> rgb_map, sampling_index_map, sampling_weight_map
        rgb_map, sampling_index_map, sampling_weight_map = self.forward_texture_sampling(faces, textures, face_index_map, weight_map, depth_map, rgb_map, sampling_index_map, sampling_weight_map)

        rgb_map = self.forward_background(face_index_map, rgb_map)

        alpha_map = self.forward_alpha_map(alpha_map, face_index_map)

        self.save_vars = faces, textures, face_index_map, weight_map, depth_map, rgb_map, alpha_map, face_inv_map, sampling_index_map, sampling_weight_map

        rgb_r, alpha_r, depth_r = jt.array([]), jt.array([]), jt.array([])
        if self.return_rgb:
            rgb_r = rgb_map
        if self.return_alpha:
            alpha_r = alpha_map
        if self.return_depth:
            depth_r = depth_map
        return rgb_r, alpha_r, depth_r

    def forward_face_index_map(self, faces, face_index_map, weight_map, depth_map, face_inv_map):
        faces_inv = jt.empty(faces.shape)
        return rasterize_cuda.forward_face_index_map(faces, face_index_map, weight_map, depth_map, face_inv_map, faces_inv, self.image_size, self.near, self.far, int(self.return_rgb), int(self.return_alpha), int(self.return_depth))

    def forward_texture_sampling(self, faces, textures, face_index_map, weight_map, depth_map, rgb_map, sampling_index_map, sampling_weight_map):
        if not self.return_rgb:
            return rgb_map, sampling_index_map, sampling_weight_map
        else:
            return rasterize_cuda.forward_texture_sampling(faces, textures, face_index_map, weight_map, depth_map, rgb_map, sampling_index_map, sampling_weight_map, self.image_size, self.eps)

    def forward_background(self, face_index_map, rgb_map):
        if self.return_rgb:
            background_color = jt.array(self.background_color).float()
            mask = (face_index_map >= 0).float().unsqueeze(-1)
            if len(background_color.shape) == 1:
                rgb_map = rgb_map * mask + (1-mask) * background_color.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            elif len(background_color.shape) == 2:
                rgb_map = rgb_map * mask + (1-mask) * background_color.unsqueeze(1).unsqueeze(1)
        return rgb_map

    def forward_alpha_map(self, alpha_map, face_index_map):
        if self.return_alpha:
            return (face_index_map >= 0).float()
        return alpha_map

    def backward_pixel_map(self, faces, face_index_map, rgb_map, alpha_map, grad_rgb_map, grad_alpha_map, grad_faces):
        if (not self.return_rgb) and (not self.return_alpha):
            return grad_faces
        else:
            return rasterize_cuda.backward_pixel_map(faces, face_index_map, rgb_map, alpha_map, grad_rgb_map, grad_alpha_map, grad_faces, self.image_size, self.eps, int(self.return_rgb), int(self.return_alpha))

    def backward_textures(self, face_index_map, sampling_weight_map, sampling_index_map, grad_rgb_map, grad_textures):
        if not self.return_rgb:
            return grad_textures
        else:
            return rasterize_cuda.backward_textures(face_index_map, sampling_weight_map, sampling_index_map, grad_rgb_map, grad_textures, self.num_faces)

    def backward_depth_map(self, faces, depth_map, face_index_map, face_inv_map, weight_map, grad_depth_map, grad_faces):
        if not self.return_depth:
            return [grad_faces]
        else:
            return rasterize_cuda.backward_depth_map(faces, depth_map, face_index_map, face_inv_map, weight_map, grad_depth_map, grad_faces, self.image_size)

class Rasterize(nn.Module):
    '''
    Wrapper around the autograd function RasterizeFunction
    Currently implemented only for cuda Tensors
    '''
    def __init__(self, image_size, near, far, eps, background_color,
                 return_rgb=False, return_alpha=False, return_depth=False):
        super(Rasterize, self).__init__()
        self.image_size = image_size
        self.image_size = image_size
        self.near = near
        self.far = far
        self.eps = eps
        self.background_color = background_color
        self.return_rgb = return_rgb
        self.return_alpha = return_alpha
        self.return_depth = return_depth

    def execute(self, faces, textures):
        return RasterizeFunction(self.image_size, self.near, self.far, self.eps, self.background_color, self.return_rgb, self.return_alpha, self.return_depth)(faces, textures)

def rasterize_rgbad(
        faces,
        textures=None,
        image_size=DEFAULT_IMAGE_SIZE,
        anti_aliasing=DEFAULT_ANTI_ALIASING,
        near=DEFAULT_NEAR,
        far=DEFAULT_FAR,
        eps=DEFAULT_EPS,
        background_color=DEFAULT_BACKGROUND_COLOR,
        return_rgb=True,
        return_alpha=True,
        return_depth=True,
):
    """
    Generate RGB, alpha channel, and depth images from faces and textures (for RGB).

    Args:
        faces (jittor.Var): Faces. The shape is [batch size, number of faces, 3 (vertices), 3 (XYZ)].
        textures (jittor.Var): Textures.
            The shape is [batch size, number of faces, texture size, texture size, texture size, 3 (RGB)].
        image_size (int): Width and height of rendered images.
        anti_aliasing (bool): do anti-aliasing by super-sampling.
        near (float): nearest z-coordinate to draw.
        far (float): farthest z-coordinate to draw.
        eps (float): small epsilon for approximated differentiation.
        background_color (tuple): background color of RGB images.
        return_rgb (bool): generate RGB images or not.
        return_alpha (bool): generate alpha channels or not.
        return_depth (bool): generate depth images or not.

    Returns:
        dict:
            {
                'rgb': RGB images. The shape is [batch size, 3, image_size, image_size].
                'alpha': Alpha channels. The shape is [batch size, image_size, image_size].
                'depth': Depth images. The shape is [batch size, image_size, image_size].
            }

    """
    if textures is None:
        inputs = [faces, None]
    else:
        inputs = [faces, textures]

    if anti_aliasing:
        # 2x super-sampling
        rgb, alpha, depth = Rasterize(image_size * 2, near, far, eps, background_color, return_rgb, return_alpha, return_depth)(*inputs)
    else:
        rgb, alpha, depth = Rasterize(image_size, near, far, eps, background_color, return_rgb, return_alpha, return_depth)(*inputs)
    
    # transpose & vertical flip
    if return_rgb:
        rgb = rgb.permute((0, 3, 1, 2))
        # may need to look at this again because it seems to be very slow
        rgb = rgb[:, :, list(reversed(range(rgb.shape[2]))), :]
    if return_alpha:
        alpha = alpha[:, list(reversed(range(alpha.shape[1]))), :]
    if return_depth:
        depth = depth[:, list(reversed(range(depth.shape[1]))), :]

    if anti_aliasing:
        # 0.5x down-sampling
        if return_rgb:
            rgb = nn.pool(rgb, 2, "mean", stride=2)
        if return_alpha:
            alpha = nn.pool(alpha.unsqueeze(1), 2, "mean", stride=2)
        if return_depth:
            depth = nn.pool(depth.unsqueeze(1), 2, "mean", stride=2)

    ret = {
        'rgb': rgb if return_rgb else None,
        'alpha': alpha if return_alpha else None,
        'depth': depth if return_depth else None,
    }

    return ret


def rasterize(
        faces,
        textures,
        image_size=DEFAULT_IMAGE_SIZE,
        anti_aliasing=DEFAULT_ANTI_ALIASING,
        near=DEFAULT_NEAR,
        far=DEFAULT_FAR,
        eps=DEFAULT_EPS,
        background_color=DEFAULT_BACKGROUND_COLOR,
):
    """
    Generate RGB images from faces and textures.

    Args:
        faces: see `rasterize_rgbad`.
        textures: see `rasterize_rgbad`.
        image_size: see `rasterize_rgbad`.
        anti_aliasing: see `rasterize_rgbad`.
        near: see `rasterize_rgbad`.
        far: see `rasterize_rgbad`.
        eps: see `rasterize_rgbad`.
        background_color: see `rasterize_rgbad`.

    Returns:
        ~jittor.Var: RGB images. The shape is [batch size, 3, image_size, image_size].

    """
    return rasterize_rgbad(
        faces, textures, image_size, anti_aliasing, near, far, eps, background_color, True, False, False)['rgb']

def rasterize_silhouettes(
        faces,
        image_size=DEFAULT_IMAGE_SIZE,
        anti_aliasing=DEFAULT_ANTI_ALIASING,
        near=DEFAULT_NEAR,
        far=DEFAULT_FAR,
        eps=DEFAULT_EPS,
):
    """
    Generate alpha channels from faces.

    Args:
        faces: see `rasterize_rgbad`.
        image_size: see `rasterize_rgbad`.
        anti_aliasing: see `rasterize_rgbad`.
        near: see `rasterize_rgbad`.
        far: see `rasterize_rgbad`.
        eps: see `rasterize_rgbad`.

    Returns:
        ~jittor.Var: Alpha channels. The shape is [batch size, image_size, image_size].

    """
    return rasterize_rgbad(faces, None, image_size, anti_aliasing, near, far, eps, None, False, True, False)['alpha']


def rasterize_depth(
        faces,
        image_size=DEFAULT_IMAGE_SIZE,
        anti_aliasing=DEFAULT_ANTI_ALIASING,
        near=DEFAULT_NEAR,
        far=DEFAULT_FAR,
        eps=DEFAULT_EPS,
):
    """
    Generate depth images from faces.

    Args:
        faces: see `rasterize_rgbad`.
        image_size: see `rasterize_rgbad`.
        anti_aliasing: see `rasterize_rgbad`.
        near: see `rasterize_rgbad`.
        far: see `rasterize_rgbad`.
        eps: see `rasterize_rgbad`.

    Returns:
        ~jittor.Var: Depth images. The shape is [batch size, image_size, image_size].

    """
    return rasterize_rgbad(faces, None, image_size, anti_aliasing, near, far, eps, None, False, False, True)['depth']
