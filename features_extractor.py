import cv2
import numpy as np



class FeaturesExtractor():
    def __init__ (self):
        self.last_kps = None
        self.last_des = None
        self.orb  = cv2.ORB_create()
        self.matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

    def extract_features(self,img):
        keypt_old =[]
        keypt_new =[]
        des_old   =[]
        des_new   =[]
        # Detection of keypoints
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        shi_thomasi_corner = cv2.goodFeaturesToTrack(gray,200,0.0001,10)
        kpts =[]
        for pt in shi_thomasi_corner:
            kpts.append(cv2.KeyPoint(pt[0][0], pt[0][1],1) )

        # Compute descriptor for each keypoint
        current_kpt,current_des = self.orb.compute(img,kpts)
        current_des = np.float32(current_des)
        #check initialisation
        if self.last_kps == None: 
            self.last_kps = current_kpt
            self.last_des = current_des
            return np.array(keypt_old) , np.array(keypt_new) , des_old , des_new

        knnmatch = self.matcher.knnMatch(current_des,self.last_des,k=2)
    
        for m,n in knnmatch:
            if m.distance < 0.7*n.distance:
                keypt_old.append(self.last_kps[m.trainIdx])
                keypt_new.append(current_kpt[m.queryIdx])
                des_old.append(self.last_des[m.trainIdx])
                des_old.append(current_des[m.queryIdx])
        self.last_kps = current_kpt
        self.last_des = current_des
        return np.array(keypt_old) , np.array(keypt_new) , des_old , des_new
        