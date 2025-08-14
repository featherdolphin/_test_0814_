import numpy as np
import cv2
from tensorflow.keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(x_test.shape)

print(y_train[0])

cv2.imshow("win", x_train[0])
cv2.waitKey(0)