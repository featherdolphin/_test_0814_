import numpy as np
import cv2
from tensorflow.keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

#mg_1 = cv2.imread("316_px_Lenna.jpg")

#mg_1 = cv2.resize(img_1,(500,500), interpolation = cv2.INTER_NEAREST)

print(x_train.shape)
print(x_test.shape)

print(y_train[0])

for i in range(0, 10):
    img = cv2.resize(x_train[i],(280,280))
    cv2.imshow("win", img)
    cv2.waitKey(0)