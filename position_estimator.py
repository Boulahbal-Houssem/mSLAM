import numpy as np
import cv2
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform
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
        ret = np.eye(4)
        kps1_norm = np.array(cv2.undistortPoints(np.expand_dims(kps1, axis=1), cameraMatrix=K, distCoeffs=None))
        kps2_norm = np.array(cv2.undistortPoints(np.expand_dims(kps2, axis=1), cameraMatrix=K, distCoeffs=None))
        if kps1_norm.shape[0] <8: 
                return ret
        model, inliers = ransac(([kps1_norm[:,0,:],kps2_norm[:,0,:]]),
                                EssentialMatrixTransform, min_samples=8,
                                residual_threshold=1, max_trials=50)       
        kps1 = kps1[inliers]
        kps2 = kps2[inliers]

        points, R, t, mask = cv2.recoverPose(model.params, kps1_norm[inliers], kps2_norm[inliers])
        ret[:3,:3] =R
        ret[0:3,3] = t.transpose()
        
        return ret 
def triangulatePoints(M_r,K,kps1,kps2):        
        M_l = np.eye(4)
        P_l = np.dot(K,  M_l)
        P_r = np.dot(K,  M_r)
        point_4d_hom = triangulate(P_l, P_r,kps1,kps2)
        print(point_4d_hom.shape)
        point_4d_hom /= point_4d_hom[:,3:] 
        point_3d = point_4d_hom[:3, :].T
        good_pts = point_3d[:,2]>0 
        return np.array(point_3d[good_pts])

def triangulate(pose1, pose2, pts1, pts2):
        ret = np.zeros((pts1.shape[0], 4))
        pose1 = np.linalg.inv(pose1)
        pose2 = np.linalg.inv(pose2)
        for i, p in enumerate(zip(pts1, pts2)):
                A = np.zeros((4,4))
                A[0] = p[0][0] * pose1[2] - pose1[0]
                A[1] = p[0][1] * pose1[2] - pose1[1]
                A[2] = p[1][0] * pose2[2] - pose2[0]
                A[3] = p[1][1] * pose2[2] - pose2[1]
                _, _, vt = np.linalg.svd(A)
                ret[i] = vt[3]
        return ret