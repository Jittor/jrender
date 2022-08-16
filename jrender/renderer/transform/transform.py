
import math
import numpy as np
import jittor as jt
from jittor import nn

from ..utils import *
from . import *


class Projection(nn.Module):
    def __init__(self, K, R, t, dist_coeffs=None, orig_size=512):
        super(Projection, self).__init__()

        self.K = K
        self.R = R
        self.t = t
        self.dist_coeffs = dist_coeffs
        self.orig_size = orig_size
        self._eye = None

        if isinstance(self.K, np.ndarray):
            self.K = jt.array(self.K).float32()
        if isinstance(self.R, np.ndarray):
            self.R = jt.array(self.R).float32()
        if isinstance(self.t, np.ndarray):
            self.t = jt.array(self.t).float32()
        if dist_coeffs is None:
            self.dist_coeffs = jt.array([[0., 0., 0., 0., 0.]]).repeat(self.K.shape[0], 1)

    def execute(self, vertices):
        vertices = projection(vertices, self.K, self.R, self.t, self.dist_coeffs, self.orig_size)
        return vertices


class LookAt(nn.Module):
    def __init__(self, perspective=True, viewing_angle=30, viewing_scale=1.0, eye=None):
        super(LookAt, self).__init__()

        self.perspective = perspective
        self.viewing_angle = viewing_angle
        self.viewing_scale = viewing_scale
        self._eye = eye

        if self._eye is None:
            self._eye = [0, 0, -(1. / math.tan(math.radians(self.viewing_angle)) + 1)]

    def execute(self, vertices):
        vertices = look_at(vertices, self._eye)
        # perspective transformation
        if self.perspective:
            vertices = perspective(vertices, angle=self.viewing_angle)
        else:
            vertices = orthogonal(vertices, scale=self.viewing_scale)
        return vertices


class Look(nn.Module):
    def __init__(self, camera_direction=[0, 0, 1], perspective=True, viewing_angle=30, viewing_scale=1.0, eye=None):
        super(Look, self).__init__()

        self.perspective = perspective
        self.viewing_angle = viewing_angle
        self.viewing_scale = viewing_scale
        self._eye = eye
        self.camera_direction = camera_direction

        if self._eye is None:
            self._eye = [0, 0, -(1. / math.tan(math.radians(self.viewing_angle)) + 1)]

    def execute(self, vertices):
        vertices = look(vertices, self._eye, self.camera_direction)
        # perspective transformation
        if self.perspective:
            vertices = perspective(vertices, angle=self.viewing_angle)
        else:
            vertices = orthogonal(vertices, scale=self.viewing_scale)
        return vertices


class Transform(nn.Module):
    def __init__(self, camera_mode='projection', K=None, R=None, t=None, dist_coeffs=None, orig_size=512,
                 perspective=True, viewing_angle=30, viewing_scale=1.0,
                 eye=None, camera_direction=[0, 0, 1]):
        super(Transform, self).__init__()

        self.camera_mode = camera_mode
        if self.camera_mode == 'projection':
            self.transformer = Projection(K, R, t, dist_coeffs, orig_size)
        elif self.camera_mode == 'look':
            self.transformer = Look(camera_direction, perspective, viewing_angle, viewing_scale, eye)
        elif self.camera_mode == 'look_at':
            self.transformer = LookAt(perspective, viewing_angle, viewing_scale, eye)
        else:
            raise ValueError('Camera mode has to be one of projection, look or look_at')

        self.eye=eye
        self.camera_direction=camera_direction
        self.viewing_angle=viewing_angle

    def execute(self, mesh):
        mesh.vertices = self.transformer(mesh.vertices)
        return mesh

    def tranpos(self, pos):
        pos = self.transformer(pos)
        return pos

    def set_eyes_from_angles(self, distances, elevations, azimuths):
        if self.camera_mode not in ['look', 'look_at']:
            raise ValueError('Projection does not need to set eyes')
        self.transformer._eye = get_points_from_angles(distances, elevations, azimuths)

    def set_eyes(self, eyes):
        if self.camera_mode not in ['look', 'look_at']:
            raise ValueError('Projection does not need to set eyes')
        self.transformer._eye = eyes

    def view_transform(self, vertices):
        if self.camera_mode == 'look':
            vertices = look_at(vertices, self.eye)
        elif self.camera_mode == 'look_at':
            vertices = look(vertices, self.eye, self.camera_direction)
        return vertices

    def projection_transform(self,vertices):
        return perspective(vertices, self.viewing_angle)

    @property
    def eyes(self):
        return self.transformer._eye
