#!/usr/bin/env python3
import cv2
import numpy as np
import os
from featres_extractor import FeaturesExtractor
from utils import utils
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

f = FeaturesExtractor()

def read_sequence(path):
    ret =[]
    for name in os.listdir(path):
        image_name  = path + name
        ret.append(image_name)
    ret.sort()
    return ret

def get_pos_fromframe(img):
    #img = cv2.resize(img, (img.shape[0]//2,img.shape[1]//2) )
    keypt_new , keypt_old, _, __ = f.extract_features(img)
    if(keypt_new.shape[0]==0):
        return np.eye(3),np.array([0,0,0])
    keypt_new , keypt_old , model  = utils.filter_fundamebtal_matrix(keypt_new , keypt_old)
    if(model.size == 0):
        return np.eye(3),np.array([0,0,0])
    R, T = utils.get_pos(model)
    img2 = cv2.drawKeypoints(img,keypt_new, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("image",img2)
    cv2.waitKey(0)
    return R , T




if __name__ == "__main__":
    sequence = read_sequence("./images/images/")
    for image_path in sequence:
        image = cv2.imread(image_path)
        R, T = get_pos_fromframe(image)
        print ("Rotation " + str(R))
        print("translation : "  + str(T))
