import tensorflow as tf
import cv2
from tensorflow.keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

new_x_test = x_test.astype("float32")/255.0
model = tf.keras.models.load_model("ngoo's_nn_model_(epochs_70).h5")
predictions = model.predict(new_x_test[:5])
for i in range(len(predictions)):
    print(predictions[i])
    cv2.imshow("win", x_test[i])
    cv2.waitKey(0)