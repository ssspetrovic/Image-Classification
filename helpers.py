def sample_plot(images, labels, n, cmap=None):
    fig, axes = plt.subplots(1, n, figsize=(n * 4, 4))
    for i in range(n):
        axes[i].imshow(images[i], cmap=cmap)
        label_index = int(labels[i])
        axes[i].set_xlabel(class_names[label_index], fontsize=13)


def grayscale(images):
    grayscale_images = []
    for image in images:
        image_uint8 = np.uint8(image)
        grayscale_image = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
        grayscale_images.append(grayscale_image)
    return np.array(grayscale_images)

def augment(images):
    augmented_images = []
    for image in images:
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)

        angle = np.random.randint(-90, 90)
        rows, cols, _ = image.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

        augmented_images.append(image)

    return np.array(augmented_images)
