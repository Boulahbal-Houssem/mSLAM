import cv2
import numpy as np
import os
from features_extractor import FeaturesExtractor
import position_estimator
from frame import Frame
import math
import viewer



def set_camera_intrinsics(fx,fy,cx,cy):
    K = np.eye(4)
    K[0,0] = fx * 1280 
    K[1,1] = fy * 1024
    K[0,2] = cx * 1280 - 0.5  
    K[1,2] = cy * 1024 - 0.5 
    return K
class SLAM(object):
    def __init__(self,k,images_path,W,H):
        self.K = k
        self.images_path = images_path
        self.W = W
        self.H = H
        self.Kinv = np.linalg.inv(self.K)
        self.features_extractor = FeaturesExtractor()

    def run(self):
        sequence = self.read_sequence(self.images_path)
        frames = []
        for img_path in sequence:
            image = cv2.imread(img_path)
            frames.append(Frame(image))
            self.get_pos(frames[-1])
            viewer.display_frame(frames[-1])



    def read_sequence(self,path):
        ret =[]
        for name in os.listdir(path):
            image_name  = path + name
            ret.append(image_name)
        ret.sort()
        return ret

    def get_pos(self,frame):
        #img = cv2.resize(img, (self.W,self.H) )

        # extract features
        keypt_new , keypt_old, _, __ = self.features_extractor.extract_features(frame.img)
        frame.kps = keypt_new

        keypt_new = np.array(cv2.KeyPoint_convert(keypt_new))
        keypt_old = np.array(cv2.KeyPoint_convert(keypt_old))

        if(keypt_new.shape[0]==0):
            return 
        # updating frame by getting the camera postion and points triangulation
        position_estimator.update_frame(frame,keypt_new,keypt_old,self.K)





if __name__ == "__main__":
    K = set_camera_intrinsics(0.535719308086809, 0.669566858850269, 0.493248545285398, 0.500408664348414)
    W = 1280
    H = 1024
    path = "./images/images/"
    mslam = SLAM(K,path,W,H)
    mslam.run()
