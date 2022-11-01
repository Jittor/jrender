import jittor as jt

# vertices : [nf,3,3]
class BBox():
    def __init__(self,vertices):
        self.max = jt.max(vertices,dims=(0,1))
        self.min = jt.max(vertices,dims=(0,1))
        self.center = (self.max - self.min) / 2