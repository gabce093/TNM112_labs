import os

from keras.src.applications.densenet import layers
from keras.src.layers import Conv3D
from keras.src.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D
import numpy as np
import tensorflow as tf
import scipy
import numpy as np
import cv2


def random_cover(image):
    # Generate random coordinates for the top-left corner of the covered region
    top_left_x = np.random.randint(0, image.shape[1])
    top_left_y = np.random.randint(0, image.shape[0])

    # Generate random dimensions for the covered region
    width = np.random.randint(10, 30)
    height = np.random.randint(10, 30)

    # Set the pixels in the covered region to a constant value (e.g., zero for black)
    image[top_left_y:top_left_y + height, top_left_x:top_left_x + width, :] = 0

    return image


def pre_processing(image):
    image = random_cover(image)
    image /= 255.0
    return image


def pre_processing_org(image):
    image /= 255.0
    return image


image_size = (64, 64)

datagen_cover = ImageDataGenerator(
    preprocessing_function=pre_processing,
    # validation_split=0.2  # Split for validation if needed
)

datagen_org = ImageDataGenerator(
    preprocessing_function=pre_processing_org,

)

covered_data = datagen_cover.flow_from_directory(
    directory='vegetable_images/train',
    target_size=image_size,
    batch_size=1,
    class_mode=None,
    seed=None,
    shuffle=False,
)

org_data = datagen_org.flow_from_directory(
    directory='vegetable_images/train',
    target_size=image_size,
    batch_size=1,
    class_mode=None,
    seed=None,
    shuffle=False,
)

#print(org_data[2000].shape)

combined_data = zip(covered_data, org_data)

# först bild
# andra in och out
# Batch
# print(training_data[0][2][0].shape)

model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding="same"),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu', padding="same"),
    layers.MaxPooling2D((2, 2), padding='same'),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same'),
])

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

steps_per_epoch = min(len(covered_data), len(org_data))

model.fit(combined_data, epochs=1, steps_per_epoch=steps_per_epoch)

# model.evaluate(model, training_data[0])


regenerated = model.predict(covered_data)

import matplotlib.pyplot as plt

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original image
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(org_data[i][0])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Destroyed image
    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(covered_data[i][0])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Regenerated image
    ax = plt.subplot(3, n, i + 2 * n + 1)
    plt.imshow(regenerated[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

model.save("test1.keras")