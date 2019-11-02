import numpy as np 
import cv2

class Frame(object):
    def __init__(self,img_path,kps=None,pos=np.eye(4),pts3d=None, dense3d = None):
        self.img_path   = img_path 
        self.kps   = kps
        self.pos   = pos
        self.pts3d = pts3d
        self.dense3d = dense3d

    def get_image(self):
        return cv2.imread(self.img_path,1)
    def get_resized_image(self):
        im = cv2.imread(self.img_path,1)
        return cv2.resize(im,(im.shape[0]//2 , im.shape[1]//2  ))