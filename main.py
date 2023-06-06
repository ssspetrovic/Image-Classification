import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

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

print(train_images.shape)
# print(train_images[:5])

# train_labels are unneccesarily two-dimensional - we can reshape it to a single dimension
print(train_labels.shape)
print(train_labels[:5])

train_labels = train_labels.reshape(-1, )
print(train_labels.shape)
print(train_labels[:5])


def sample_plot(images, labels, index):
    plt.figure(figsize=(4, 4))
    plt.imshow(images[index])
    plt.xlabel(class_names[labels[index]], fontsize=13)
    plt.show()


sample_plot(train_images, train_labels, 9)
