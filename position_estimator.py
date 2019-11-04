import numpy as np
import cv2
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

def add_ones(x):
  return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def update_frame(frame,kps1,kps2,K):
        local_transform = estimate_postion(kps1, kps2 , K[:3,:3])
        frame.pos = frame.world_transformation.dot(local_transform)

        local_pts = triangulatePoints(frame.pos,K,kps1,kps2)
        local_pts = add_ones(local_pts).T
        frame.pts3d = frame.world_transformation.dot(local_pts).T[:,:3]


def estimate_postion(kps1, kps2 , K):
        kps1_norm = cv2.undistortPoints(np.expand_dims(kps1, axis=1), cameraMatrix=K, distCoeffs=None)
        kps2_norm = cv2.undistortPoints(np.expand_dims(kps2, axis=1), cameraMatrix=K, distCoeffs=None)

        E, mask = cv2.findEssentialMat(kps1_norm,kps2_norm, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=3.0)
        points, R, t, mask = cv2.recoverPose(E, kps1_norm, kps2_norm)
        ret = np.eye(4)
        ret[:3,:3] =R
        ret[0:3,3] = t.transpose()
        return ret 
def triangulatePoints(M_r,K,kps1,kps2):        
        M_l = np.eye(4)
        P_l = np.dot(K,  M_l)[:3,:4]
        P_r = np.dot(K,  M_r)[:3,:4]
        point_4d_hom = cv2.triangulatePoints(P_l, P_r, np.expand_dims(kps1, axis=1), np.expand_dims(kps2, axis=1))
        point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
        point_3d = point_4d[:3, :].T
        return np.array(point_3d)