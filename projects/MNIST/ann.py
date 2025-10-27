"""
MNIST veri setinde toplam 10 class vardır. Resimler 28x28 boyutunda ve grayscaledir.
60000 train, 10000 test seti vardır.
"""
#import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

#load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

"""
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

X_train shape: (60000, 28, 28)
y_train shape: (60000,)
"""
img = x_train[0]
stages = {'original': img}

#histogram eşitleme
eq = cv2.equalizeHist(img)
stages['histogram esitleme'] = eq

#gaussian blur
blur = cv2.GaussianBlur(eq, (5,5), 0)
stages['gaussian blur'] = blur

#canny ile kenar tespiti
edges = cv2.Canny(blur, 50, 150)
stages['canny kenarlari'] = edges

#görselleştirme
fig, axes = plt.subplots(2, 2, figsize=(6, 6))
axes = axes.flat
for ax, (title, im) in zip(axes, stages.items()):
    ax.imshow(im, cmap = 'gray')
    ax.set_title(title)
    ax.axis('off')
plt.suptitle("MNIST Image Processing Stages")
plt.tight_layout()
plt.show()

#preprocessing fonksiyonu
def preprocess_images(img):
    """
    1 - Histogram Eşitleme
    2 - Gaussian Blur
    3 - Canny Kenarları
    4 - Flattening
    5 - Normalizasyon
    """
    img_eq = cv2.equalizeHist(img) #1
    img_blur = cv2.GaussianBlur(img_eq, (5,5), 0) #2
    img_edges = cv2.Canny(img_blur, 50, 150) #3
    features = img_edges.flatten() / 255.0 #4-5
    return features

num_train = 10000
num_test = 2000

X_train = np.array([preprocess_images(img) for img in x_train[:num_train]]) 
y_train_sub = y_train[:num_train]

X_test = np.array([preprocess_images(img) for img in x_test[:num_test]])
y_test_sub = y_test[:num_test]

#ann model tanımlama
model = Sequential([
    Dense(128, activation='relu', input_shape = (784,)),
    Dropout(0.5),
    Dense(64, activation = 'relu'),
    Dense(10, activation = 'softmax')
])

#compile model
model.compile(
    optimizer = Adam(learning_rate=0.001),
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
model.summary()

#ann model eğitimi
history = model.fit(
    X_train, y_train_sub,
    validation_data = (X_test, y_test_sub),
    epochs = 50,
    batch_size = 32,
    verbose = 2
)

#modeli evrimleştir
test_loss, test_accuracy = model.evaluate(X_test, y_test_sub)
print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}')

#eğitimi görselleştir
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label = 'Training Loss')
plt.plot(history.history['val_loss'], label = 'Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label = 'Training Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()