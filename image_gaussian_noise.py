from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import datetime

initial_image = Image.open('venice_100.jpg')
width, height = initial_image.size

image_array = np.array(initial_image)

# We can control the amount of Gaussian noise added to each pixel by specifying the variance value.
# A higher value means noisier pixels.
v = 300
# Generate Gaussian noise (loc,scale,size)
# mean = 0, because Gaussian distribution
# scale - the standard deviation of the distribution
# size -  the shape of the noise array, which matches the shape of the initial array
noise = np.random.normal(0, np.sqrt(v), image_array.shape)

noisy_image_array = image_array + noise

# To ensure that the pixel values in the noisy_image_array remain within a valid range (0-255 for uint8 images)
noisy_image_array = np.clip(noisy_image_array, 0, 255)

# Convert the noisy NumPy array back to an image
noisy_image = Image.fromarray(noisy_image_array.astype(np.uint8))

# pixel_values = noisy_image.ravel()
pixel_values = noisy_image_array.flatten()

#  Visualization
now = datetime.datetime.now()

# plt.figure(figsize=(8, 8))
# plt.imshow(initial_image, cmap=plt.cm.gray)
# plt.title('Initial image'+ "\n time: " + str(now.strftime("%d-%m-%Y %H:%M:%S")))
# plt.axis('off')
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 8))
# plt.imshow(noisy_image, cmap=plt.cm.gray)
# plt.title('Noisy image'+ "\n time: " + str(now.strftime("%d-%m-%Y %H:%M:%S")))
# plt.axis('off')
# plt.tight_layout()
# plt.show()

# Save the image
noisy_image.save('noisy_image.png')
# noisy_image.show()
