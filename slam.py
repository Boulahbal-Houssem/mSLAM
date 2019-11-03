import cv2
import numpy as np
import os
from features_extractor import FeaturesExtractor
import position_estimator
from frame import Frame
import math
import viewer
from mapp import Mapp
#cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)



def set_camera_intrinsics(fx,fy,cx,cy):
    K = np.eye(4)
    K[0,0] = fx * (1280//2) 
    K[1,1] = fy * (1024//2)
    K[0,2] = cx * (1280//2) - 0.5  
    K[1,2] = cy * (1024//2) - 0.5 
    return K
class SLAM(object):
    def __init__(self,k,images_path,W,H):
        self.K = k
        self.images_path = images_path
        self.W = W//2
        self.H = H//2
        self.Kinv = np.linalg.inv(self.K)
        self.features_extractor = FeaturesExtractor()
        self.stereo = cv2.StereoSGBM_create(numDisparities =112, blockSize = 16, preFilterCap=4, minDisparity=0,
                                            P1 = 600, P2 = 2400, disp12MaxDiff = 10, uniquenessRatio = 1,
                                            speckleWindowSize = 150, speckleRange = 2 )
        self.slam_map = Mapp()

    def run(self):
        sequence = self.read_sequence(self.images_path)
        frames = []
        for img_path in sequence:
            frames.append(Frame(img_path))
            self.get_pos(frames[-1])
            if(len(frames)>1):
                frames[-1].pos = np.array(frames[-1].pos.dot(frames[-2].pos))
                viewer.display_frame(frames[-1])

                '''disparity = self.stereo.compute(frames[-1].get_resized_image(),frames[-2].get_resized_image())
                frames[-1].dense3d = disparity
                cv2.imshow("disparity",disparity)'''
                self.slam_map.display_map(frames)

                



    def read_sequence(self,path):
        ret =[]
        for name in os.listdir(path):
            image_name  = path + name
            ret.append(image_name)
        ret.sort()
        return ret

    def get_pos(self,frame):
        image = cv2.imread(frame.img_path)
        image = cv2.resize(image, (self.W,self.H) )
        # extract features
        keypt_old, keypt_new, _, __ = self.features_extractor.extract_features(image)
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
    cv2.destroyWindow("image") 
