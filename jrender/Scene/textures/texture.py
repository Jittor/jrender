
import jittor as jt
from jittor import transform
import numpy as np
import numba
import math
from .utils.sample2D import sample2D
from skimage.io import imread,imsave
from PIL import Image


class Texture():  #image:[height,width,(channels)]  uv:[...,2]
    
    def __init__(self,image = None,uv = None,sampler = sample2D):
        self._image = None
        self.height = None
        self.width = None

        if image is not None:
            self.image = image
        self._uv = uv
        self._query_uv = None
        self.query_uv_update = True
        self.sampler = sampler
        self._mipmap = None            #sampling with mipmaps is usually handled by cuda
        self.mipmap_update = True

    @property
    def query_uv(self):
        if self.uv is None:
            raise ValueError("The texture has not been attached to uvs")
        if self.query_uv_update:
            self._query_uv = self.sampler(self.image,self.uv,default=99999)
            self.query_uv_update = False
        return self._query_uv

    @property
    def uv(self):
        return self._uv

    @uv.setter
    def uv(self,_uv):
        self._uv = _uv
        self.update()
        return
    
    @property
    def image(self):
        return self._image

    @image.setter
    def image(self,image):
        if len(image.shape) == 2:
            self._image = image
            self.channel = 1
        else:
            self._channel = image.shape[2]
        self.height = self._image.shape[0]
        self.width = self._image.shape[1]
        self.update()
        return
    
    def update(self):
        self.query_uv_update = True
        self.mipmap_update = True 
    
    def generate_SAT(self):
        if self.channel == 1:
            if len(self.image.shape) == 3:
                image = image.squeeze(2)
            image = self.image
            row = jt.zeros([1,self.width])
            col = jt.zeros([self.height+1,1])
            SAT = jt.concat([row,image],dim = 0)
            SAT = jt.concat([col,SAT],dim = 1)
            SAT = np.array(SAT)
            return generate_SAT_fast(SAT)
        else:
            raise ValueError("generate_SAT only suppots 1 channel")

    @classmethod
    def generate_mipmap(self,image):  #[sz,sz]
        height = image.shape[0]
        width = image.shape[0]
        maxMipmapLevel = math.floor(math.log2(min(height,width)))
        level = 0
        mipmap = image.reshape((1,width * height))
        index = [0, width * height]
        while level < maxMipmapLevel:
            width = int(round(width/2))
            height = int(round(height/2))
            trans = transform.Compose([transform.Resize((height,width),Image.BILINEAR),transform.ToTensor()])
            image = trans(image)
            image = jt.array(image).squeeze(0)
            #image[image > 100] = 0
            #image[image < 0.1] = 0
            #imsave("D:\Render\jrender\data\\results\\temp\\mipmap.jpg",image) 
            mipmap = jt.concat([mipmap,image.reshape((1,width * height))], dim = 1)
            index.append(index[level + 1] + width * height)
            level = level + 1
        index = jt.array(index).int32()
        return mipmap,index

    @classmethod
    def from_path(cls,path):
        if path is None:
            return None
        image = imread(path).astype(np.float32)/255.
        return cls(image)

    @classmethod
    def generate_SAT(self,image):  #image: [sz,sz]  type: jt.array
        width = image.shape[1]
        height = image.shape[0]
        row = jt.zeros([1,width])
        col = jt.zeros([height+1,1])
        SAT = jt.concat([row,image],dim = 0)
        SAT = jt.concat([col,SAT],dim = 1)
        SAT = np.array(SAT)
        SAT = generate_SAT_fast(SAT)
        return jt.array(SAT[1:,1:]).float32()

class Sampler():
    def __init__(self,sampler=sample2D):
        self.sampler=sampler

    def execute(self,image,uv):
        return self.sampler(image,uv)

@numba.jit(nopython = True)
def generate_SAT_fast(data):
    for i in range(data.shape[0]-1):
        for j in range(data.shape[1]-1):
            data[i+1,j+1] = data[i,j+1] + data[i+1,j] + data[i+1,j+1] - data[i,j]
    return data