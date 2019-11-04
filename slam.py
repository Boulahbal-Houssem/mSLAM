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



def set_camera_intrinsics(fx,fy,cx,cy,W,H):
    K = np.eye(4)
    K[0,0] = fx * W 
    K[1,1] = fy * H
    K[0,2] = cx * W - 0.5  
    K[1,2] = cy * H - 0.5 
    return K
class SLAM(object):
    def __init__(self,k,images_path,W,H):
        self.K = k
        self.images_path = images_path
        self.W = W
        self.H = H
        self.Kinv = np.linalg.inv(self.K)
        self.features_extractor = FeaturesExtractor()
        self.stereo = cv2.StereoSGBM_create(numDisparities =112, blockSize = 16, preFilterCap=4, minDisparity=0,
                                            P1 = 600, P2 = 2400, disp12MaxDiff = 10, uniquenessRatio = 1,
                                            speckleWindowSize = 150, speckleRange = 2 )
        self.slam_map = Mapp()

    def run(self):
        cap = cv2.VideoCapture("test_countryroad.mp4")
        frames = [Frame(np.eye(4))]
        while cap.isOpened():
            world_transformation = frames[-1].pos
            frames.append(Frame(world_transformation))
            _,image = cap.read()
            image = cv2.resize(image, (self.W,self.H) )    

            self.get_pos(frames[-1],image)
            viewer.display_frame(frames[-1],image)

            '''disparity = self.stereo.compute(frames[-1].get_resized_image(),frames[-2].get_resized_image())
            frames[-1].dense3d = disparity
            cv2.imshow("disparity",disparity)'''
            if(len(frames)>2):
                self.slam_map.display_map(frames)


    def get_pos(self,frame,image):
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
    W = 1280//2
    H = 1024//2
    K = set_camera_intrinsics(0.535719308086809, 0.669566858850269, 0.493248545285398, 0.500408664348414,W,H)
    path = "./images/images/"
    mslam = SLAM(K,path,W,H)
    mslam.run()
    cv2.destroyWindow("image") 
