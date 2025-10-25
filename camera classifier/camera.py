import cv2
import numpy as np

class Camera:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise Exception("Could not open camera.")
        
    def __del__(self):
        if self.camera.isOpened():
            self.camera.release()
            
    def get_frame(self):
        if self.camera.isOpened():
            ret, frame = self.camera.read()
            
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return None
    