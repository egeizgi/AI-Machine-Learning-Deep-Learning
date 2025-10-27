# model.py
from sklearn.svm import LinearSVC
import numpy as np
import cv2
import os

class Model:
    def __init__(self):
        self.model = LinearSVC()
        self.is_trained = False 

    def _load_images_for_class(self, cls_dir):
        X = []
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff')):
                continue
            img = cv2.imread(os.path.join(cls_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (150, 150), interpolation=cv2.INTER_AREA)
            X.append(img.reshape(-1))
        return X

    def train(self, counters):
        X, y = [], []

        if os.path.isdir('1'):
            X1 = self._load_images_for_class('1')
            X += X1
            y += [1] * len(X1)
        if os.path.isdir('2'):
            X2 = self._load_images_for_class('2')
            X += X2
            y += [2] * len(X2)
        if os.path.isdir('3'):
            X3 = self._load_images_for_class('3')
            X += X3
            y += [3] * len(X3)

        if len(X) == 0:
            print("No training images found. Please capture images for both classes before training.")
            self.is_trained = False
            return False
        if len(set(y)) < 2:
            print("Need at least one image for EACH class (1 and 2). Please capture more images.")
            self.is_trained = False
            return False

        X = np.vstack(X)
        y = np.array(y)
        self.model.fit(X, y)
        self.is_trained = True  
        print(f"Model trained successfully on {len(y)} samples.")
        return True             

    def predict(self, frame):
        if not self.is_trained:
            return None         
        if frame is None or not frame[0]:
            return None
        rgb = frame[1]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (150, 150), interpolation=cv2.INTER_AREA)
        feat = gray.reshape(1, -1)
        pred = self.model.predict(feat)
        return int(pred[0])
