import cv2
import numpy as np
import os
from features_extractor import FeaturesExtractor
import position_estimator
cv2.namedWindow('image', cv2.WINDOW_NORMAL)


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
        for img_path in sequence:
            image = cv2.imread(img_path)
            R, T = self.get_pos(image)
            print ("Rotation " + str(R))
            print("translation : "  + str(T))

    def read_sequence(self,path):
        ret =[]
        for name in os.listdir(path):
            image_name  = path + name
            ret.append(image_name)
        ret.sort()
        return ret

    def get_pos(self,img):
        #img = cv2.resize(img, (self.W,self.H) )

        # extract features
        keypt_new , keypt_old, _, __ = self.features_extractor.extract_features(img)
        if(keypt_new.shape[0]==0):
            return np.eye(3),np.array([0,0,0])
        # normlize the matched poitns
        keypt_new_ = np.array(cv2.KeyPoint_convert(np.array(keypt_new)))
        keypt_old_ = np.array(cv2.KeyPoint_convert(np.array(keypt_old)))
    

        # getting the camera pos(t) with respect to pos(t-1)
        R,t= position_estimator.getPos(keypt_new_, keypt_old_)
        coord = np.eye(4)
        coord[0:3,0:3] = R
        coord[0:3,3]   = t
        coord = np.dot( np.dot(self.K.T , coord) , self.K )

        #display matched points 
        img2 = cv2.drawKeypoints(img,keypt_new, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("image",img2)
        cv2.waitKey(0)
        return coord[0:3,0:3] , coord[0:3,3]




if __name__ == "__main__":
    W = 1280//2
    H = 1024//2
    F = 1338
    K = np.array([[F,0,W//2,0],[0,F,H//2,0],[0,0,1,0],[0,0,0,1]])
    path = "./images/images/"
    mslam = SLAM(K,path,W,H)
    mslam.run()
