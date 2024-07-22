import matplotlib.pyplot as plt
import skimage
import numpy as np
from skimage import data
from skimage import io
import image_gaussian_noise as ign


# Generate vertices based on batches of pixels
# Choose batch_size in {9, 25, 49, ...}
def vertex_generation(batch_size):

    shift = int(batch_size ** (1/2) // 2)
    image = skimage.io.imread('noisy_image.png')

    n = image.shape[0]
    m = image.shape[1]

    print("Loaded " + str(n) + "x" + str(m) + " image.")
    # for i in range(n):
    #     print(image[i])


    vertices = []

    for i in range(n):
        for j in range(m):
            # Center pixel
            if (i - shift >= 0) and (j - shift >= 0) and (i + shift <= n - 1) and (j + shift <= m - 1):
                batch = np.zeros(batch_size)
                index = 0
                for k in range(-shift, shift + 1):
                    for l in range(-shift, shift + 1):
                        batch[index] = image[i + k, j + l]
                        index = index + 1
                vertices.append(batch)

            # Edge and corner pixels
            else:
                batch = np.zeros(batch_size)
                index = 0
                for k in range(-shift, shift + 1):
                    for l in range(-shift, shift + 1):
                        if (i + k < 0) or (j + l < 0) or (i + k > n - 1) or (j + l > m - 1):
                            batch[index] = image[i][j]
                        else:
                            batch[index] = image[i + k, j + l]
                        index = index + 1
                vertices.append(batch)

    # plt.figure(figsize=(8, 8))
    # plt.imshow(image, cmap=plt.cm.gray)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()
    return vertices

