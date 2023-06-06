import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import np_utils
from tensorflow.keras import datasets, layers, models
import cv2

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# normalizing the RGB values of the images to be between the range of 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

num_classes = len(class_names)
train_images_shape = train_images.shape[1:]

# print(train_images.shape)
# print(train_images[:5])

# train_labels are unnecessarily two-dimensional - we can reshape it to a single dimension
# print(train_labels.shape)
# print(train_labels[:5])

train_labels = train_labels.reshape(-1, )


# print(train_labels.shape)
# print(train_labels[:5])


def sample_plot(images, labels, index):
    plt.figure(figsize=(4, 4))
    plt.imshow(images[index])
    plt.xlabel(class_names[labels[index]], fontsize=13)
    plt.show()


# sample_plot(train_images, train_labels, 9)

cnn = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(3000, activation="relu"),
    layers.Dense(1000, activation="relu"),
    layers.Dense(10, activation="sigmoid"),
])

cnn.compile(optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])

print("CNN with 10 epochs")
cnn.fit(train_images, train_labels, epochs=10)

test_loss, test_accuracy = cnn.evaluate(test_images, test_labels)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
