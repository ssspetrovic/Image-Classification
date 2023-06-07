import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.utils import np_utils
from keras import datasets, layers, models

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


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


print("\nPrimer jednog kanala ulazne slike:")
print(train_images[0][0][:3])
print("\nPrimer nekoliko izgleda oblika ulaznih labela:")
print(train_labels[:5])

# Normalizovanje RGB vrednosti slika tako da budu u opsegu od 0 do 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Pretvaranje vrednosti labela u jednodimenzionalni niz umesto dvodimenzionalnog
train_labels = train_labels.reshape(
    -1,
)


print("\nPrimer jednog kanala ulazne slike:")
print(train_images[0][0][:3])
print("\nPrimer nekoliko izgleda oblika ulaznih labela:")
print(train_labels[:5])


num_classes = len(class_names)
train_images_shape = train_images.shape[1:]


def sample_plot(images, labels, n):
    fig, axes = plt.subplots(1, n, figsize=(n * 4, 4))
    for i in range(n):
        axes[i].imshow(images[i])
        label_index = int(labels[i])
        axes[i].set_xlabel(class_names[label_index], fontsize=13)


sample_plot(train_images, train_labels, 3)
plt.show()

cnn = models.Sequential(
    [
        layers.Conv2D(
            filters=64,
            padding="same",
            kernel_size=(3, 3),
            activation="relu",
            input_shape=train_images_shape,
        ),
        layers.Dropout(0.2),
        layers.Conv2D(
            filters=128, padding="same", kernel_size=(3, 3), activation="relu"
        ),
        layers.Dropout(0.2),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(
            filters=256, padding="same", kernel_size=(3, 3), activation="relu"
        ),
        layers.Dropout(0.2),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

cnn.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

cnn.fit(
    train_images, train_labels, validation_data=(test_images, test_labels), epochs=20
)
cnn.evaluate(test_images, test_labels)
