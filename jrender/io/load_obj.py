import os
import jittor as jt
import numpy as np
from skimage.io import imread

from .utils._load_obj_for_n3mr import load_obj as load_obj_for_n3mr
from .utils._load_obj_for_softras import load_obj as load_obj_for_softras

def load_obj(filename_obj, normalization=False, load_texture=False, dr_type='softras', texture_res=4, texture_type='surface', texture_wrapping='REPEAT', use_bilinear=True):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x) and faces (f x x x).

    In the official softras and n3mr versions, the load_obj method is different.
    """
    assert dr_type in ['softras', 'n3mr']

    if dr_type == 'softras':
        return load_obj_for_softras(filename_obj, normalization=normalization, load_texture=load_texture, texture_res=texture_res, texture_type=texture_type)
    elif dr_type == 'n3mr':
        return load_obj_for_n3mr(filename_obj, normalization=normalization, texture_res=texture_res, load_texture=load_texture, texture_wrapping=texture_wrapping, use_bilinear=use_bilinear)