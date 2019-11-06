import OpenGL.GL as gl
import pangolin
import numpy as np
from multiprocessing import Process, Queue
class Mapp(object):
    def __init__(self):
        self.queue = Queue()
        self.state = None
        self.scam  = None
        self.dscam = None
        self.handler =None
        self.k =[]
        p = Process(target=self.display_thread,args=())
        p.daemon = True
        p.start()
        
    def display_thread(self):
        self.init_viewer(1280//2,1024//2)
        while(1):
            self.refresh_viewer()


    def init_viewer(self,w=1024,h=768):
        pangolin.CreateWindowAndBind('Main', 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)
        self.tree = pangolin.Renderable()
        # Define Projection and initial ModelView matrix
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(w, h, 420, 420, 320, 240, 0.2, 200),
            pangolin.ModelViewLookAt(0, -10, -8,
                                     0, 0, 0,
                                    0, -1, 0))
        self.tree.Add(pangolin.Axis())

        self.handler = pangolin.Handler3D(self.scam)
            # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -w/h)
        self.dcam.SetHandler(self.handler)

    def refresh_viewer(self):
        while(not self.queue.empty()):
            self.state = self.queue.get()
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            self.dcam.Activate(self.scam)

            gl.glPointSize(1)
            gl.glColor3f(1.0, 0.0, 0.0)
            pangolin.DrawPoints(self.state[0])

            gl.glPointSize(1)
            gl.glColor3f(0.0, 0.0, 1.0)
            pangolin.DrawCameras(self.state[1])
            self.tree.Render()
            pangolin.FinishFrame()
 
    def display_map(self,frames):
        camera_pose = []
        point_3d    = []
        for frame in frames:
            for pt in frame.pts3d:
                point_3d.append(pt)
            camera_pose.append(frame.pos)
        self.queue.put((np.array(point_3d), np.array(camera_pose)))