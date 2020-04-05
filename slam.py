import cv2
import numpy as np
import os
from features_extractor import FeaturesExtractor
import position_estimator
from frame import Frame
import math
import viewer
from mapp import MapViewer ,Mapp

class SLAM(object):
    def __init__(self,k,images_path,W,H):
        self.K = k
        self.images_path = images_path
        self.W = W
        self.H = H
        self.Kinv = np.linalg.inv(self.K)
        self.features_extractor = FeaturesExtractor()
        self.slammap = Mapp()

    def run(self):
        cap = cv2.VideoCapture("test_countryroad.mp4")
        #while cap.isOpened():
        path = self.read_sequence("./images/images/")
        for img in path:
            image = cv2.imread(img)
            image = cv2.resize(image, (self.W,self.H) )    

            if(! self.update_frame(image))
                continue
            viewer.display_frame(self.slammap.frames[-1],image)
            self.slammap.display_map()
                        
    def update_frame(self,image):
        # extract features
        keypt_old, keypt_new, _, __ = self.features_extractor.extract_features(image)
        keypt_new = np.array(cv2.KeyPoint_convert(keypt_new))
        keypt_old = np.array(cv2.KeyPoint_convert(keypt_old))
        
        if(keypt_new.shape[0]==0):
            return False
        
        position_estimator.update_frame(self.slammap,keypt_new,keypt_old,self.K)
        return True

    def read_sequence(self,path):
        ret =[]
        for name in os.listdir(path):
            image_name  = path + name
            ret.append(image_name)
        ret.sort()
        return ret

def set_camera_intrinsics(fx,fy,cx,cy,W,H):
    K = np.eye(4)
    K[0,0] = fx * W 
    K[1,1] = fy * H
    K[0,2] = cx * W - 0.5  
    K[1,2] = cy * H - 0.5 
    return K

if __name__ == "__main__":
    W = 1280//2
    H = 1024//2
    K = set_camera_intrinsics(0.535719308086809, 0.669566858850269, 0.493248545285398, 0.500408664348414,W,H)

    W, H = 1920//2, 1080//2
    F = 270
    K = np.array([[F,0,W//2,0],[0,F,H//2,0],[0,0,1,0],[0,0,0,1]])
    Kinv = np.linalg.inv(K)  

    path = "./images/images/"
    mslam = SLAM(K,path,W,H)
    mslam.run()
    cv2.destroyWindow("image") 