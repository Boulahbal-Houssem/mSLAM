import numpy as np 
import cv2

class Frame(object):
    
    def __init__(self, frameid, world_transformation, kps=None, pos=np.eye(4), pts3d=np.array([])):
        self.id = frameid
        self.world_transformation = world_transformation
        self.kps   = kps
        self.pos   = pos
        self.pts   = []
        self.pts3d = pts3d

class Point:
    def __init__(self,pts,frame,idx,mapp):
        self.pts = pts
        self.frame = frame
        self.idx = idx
        self.mapp = mapp
    
    def add_observation(self,frame,idx):
        frame.pts.append(self)
        self.frame.append(frame)
        self.idx.append(frame)