import os

import keras.models
from keras.src.applications.densenet import layers
from keras.src.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt

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

def createData(path, batch_size, image_size):
    datagen_cover = ImageDataGenerator(preprocessing_function=pre_processing)

    covered_data = datagen_cover.flow_from_directory(
        directory=path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=None,
        seed=None,
        shuffle=False,
    )

    datagen_org = ImageDataGenerator(preprocessing_function=pre_processing_org)

    org_data = datagen_org.flow_from_directory(
        directory=path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=None,
        seed=None,
        shuffle=False,
    )
    return zip(covered_data, org_data)

epochs = 1
image_size = (128, 128)
combined_data = createData('vegetable_images/train', batch_size=1, image_size=image_size)

model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(image_size[0], image_size[1], 3)),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
    #layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu', padding="same"),
    layers.MaxPooling2D((2, 2), padding='same'),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    #layers.UpSampling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same'),
])

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

steps_per_epoch = 15000

log = model.fit(combined_data, epochs=epochs, steps_per_epoch=steps_per_epoch)
#model.save("test.keras")

#model.evaluate(model, training_data[0])

#model = keras.models.load_model("128_org.keras")

#regenerated = model.predict(covered_data, steps=10)

n = 10
plt.figure(figsize=(20, 4))

for i in range(n):
    img_pair = next(combined_data)
    destroyed = img_pair[0]
    original = img_pair[1]
    regenerated = model.predict(destroyed)[0]
    # Original image
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(original[0])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Destroyed image
    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(destroyed[0])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Regenerated image
    ax = plt.subplot(3, n, i + 2 * n + 1)
    plt.imshow(regenerated)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

