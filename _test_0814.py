import numpy as np
import cv2
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.modle Sequential
from tensorflow.keras.layers Dens, Flatten

(x_train, y_train),(x_test, y_test) = mnist.load_data()

new_x_train = x_train.astype("float32")/255.0
new_x_test = x_test.astype("float32")/255.0

new_y_train = to_categorical(y_train)
new_y_test = to_categorical(y_test)

print(new_y_train[0])

model = Sequential(
    [
        Flatten(input_shape = (28,28)),
        Dense(128)
    ]
)


"""
print(x_train.shape)
print(x_test.shape)



for i in range(10):
    img = cv2.resize(x_train[i],(280,280))
    cv2.imshow("win", img)
    cv2.waitKey(0)
"""