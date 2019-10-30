import numpy as np
import cv2
from skimage import data
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform
import featres_extractor

W = 1280//2
H = 1024//2

F = 1338
K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])
Kinv = np.linalg.inv(K)

def add_ones(x):
        return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
def normalize(pts):
        return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]

def denormalize(pt):
        ret = np.dot(K, np.array([pt[0], pt[1], 1.0]))
        return int(round(ret[0])), int(round(ret[1]))
avg =[]
def filter_fundamebtal_matrix(kps1,kps2):
    if(len(kps1)<0):
            return kps1,kps2
    kps1 = np.array(kps1)
    kps2 = np.array(kps2)
    kps1_point	=normalize( np.array(cv2.KeyPoint_convert(kps1)) )
    kps2_point	=normalize( np.array(cv2.KeyPoint_convert(kps2)) )
    model, inliers = ransac((kps1_point,kps2_point),
                        EssentialMatrixTransform, min_samples=8,
                        residual_threshold=1, max_trials=50)
    inlier_keypoints_left = kps1[inliers]
    inlier_keypoints_right = kps2[inliers]
    S,V,D = np.linalg.svd(model.params) 

    avg.append( np.array([np.sqrt(2) / ( (V[1]+V[2])/2)]))
    print(V)
    return inlier_keypoints_left, inlier_keypoints_right , model.params

def get_pos(E):
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
