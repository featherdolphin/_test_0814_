import numpy as np
import cv2
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

(x_train, y_train),(x_test, y_test) = mnist.load_data()

new_x_train = x_train.astype("float32")/255.0
new_x_test = x_test.astype("float32")/255.0

new_y_train = to_categorical(y_train)
new_y_test = to_categorical(y_test)

print(new_y_train[0])

model = Sequential(
    [
        Flatten(input_shape = (28,28)),
        Dense(128,activation = "sigmoide"),
        Dense(64,activation = "relue"),
        Dense(10,activation = "softmax"),
    ]
)
model.summary()
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
model.fit(new_x_train, new_y_train, epochs = 120, batch_size = 100, validation_split = 0.2)

model.save("ngoo's_nn_model_2.h5")

loss, accuracy = model.evaluate(new_x_test, new_y_test)
print(accuracy)
predictions = model.predict(new_x_test[:5])
for i in range(len(predictions)):
    print(predictions[i])
    cv2.imshow("win", x_test[i])
    cv2.waitKey(0)

"""
print(x_train.shape)
print(x_test.shape)



for i in range(10):
    img = cv2.resize(x_train[i],(280,280))
    cv2.imshow("win", img)
    cv2.waitKey(0)
"""