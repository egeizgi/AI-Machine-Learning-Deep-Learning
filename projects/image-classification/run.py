import cv2 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

layers = tf.keras.layers
models = tf.keras.models

model = models.load_model('image_classifier_model.h5')

img = cv2.imread('test_images/car.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img, cmap = plt.cm.binary)

prediction = model.predict(np.array([img]) / 255.0)
index = np.argmax(prediction)
print(f'Prediction is {class_names[index]}')