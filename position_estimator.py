import numpy as np
import cv2
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform


def getPos(kps1,kps2):
        _, __, model = get_essential_m(kps1,kps2)
        print(model.params)
        R,t = extract_RT(model.params)
        return R,t
        
def get_essential_m(kps1,kps2):
        if(len(kps1)<8):
                return kps1,kps2,None
        data  = (kps1,kps2)
        print(data[0].shape)
        model, inliers = ransac(data,FundamentalMatrixTransform, min_samples=8,
                                residual_threshold=1, max_trials=50)
        inlier_keypoints_left = kps1[inliers]
        inlier_keypoints_right = kps2[inliers]
        print(model)
        return inlier_keypoints_left, inlier_keypoints_right , model

def extract_RT(E):
        W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
        U,d,Vt = np.linalg.svd(E)
        assert np.linalg.det(U) > 0
        if np.linalg.det(Vt) < 0:
                Vt *= -1.0
        R = np.dot(np.dot(U, W), Vt)
        if np.sum(R.diagonal()) < 0:
                R = np.dot(np.dot(U, W.T), Vt)
        t = U[:, 2]
        return R, t
