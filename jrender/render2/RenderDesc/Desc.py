

class GeometryDescption():
    def __init__(self, obj_faces=None, name_dic=None):
        self.proj_v_update = True
        self.wcoord_update = True
        self.normal_update = True
        self.obj_faces = obj_faces
        self.name_dic = name_dic

    def reset(self):
        self.proj_v_update = True
        self.wcoord_update = True
        self.normal_update = True


class MaterialDescption():

    def __init__(self, objects=None,PBR=False):
        self.objects = objects
        self.KD_update = True
        self.albedo_update = True
        self.metallic_roughness_update = True
        self._PBR = PBR

    @property
    def PBR(self):
        return self._PBR

    @PBR.setter
    def PBR(self, PBR):
        if PBR == True:
            self.albedo_update = True
            self.metallic_roughness_update = True
        elif PBR == False:
            self.albedo_update = False
            self.metallic_roughness_update = False
        self._PBR = PBR


class IlluminationDescption():
    
    def __init__(self, lights=None, shading="blinn_phong"):
        self.lights = lights
        self.light_update = True
        self.shading = shading


