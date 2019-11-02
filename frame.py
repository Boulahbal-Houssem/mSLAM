import numpy as np 


class Frame(object):
    def __init__(self,img,kps=None,pos=np.eye(4),pts3d=None, dense3d = None):
        self.img   = img 
        self.kps   = kps
        self.pos   = pos
        self.pts3d = pts3d
        self.dense3d = dense3d
