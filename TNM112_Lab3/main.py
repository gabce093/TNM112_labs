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



datagen = ImageDataGenerator(
    preprocessing_function=pre_processing,
   # validation_split=0.2  # Split for validation if needed
)

image_size = (64, 64)

covered_data = datagen.flow_from_directory(
    directory='vegetable_images/train',
    target_size=image_size,
    batch_size=10,
    subset='training'
)

org_data = tf.keras.utils.image_dataset_from_directory(
    directory='vegetable_images/train',
    labels=None,
    image_size = image_size,
)

normalization_layer = tf.keras.layers.Rescaling(1./255)

org_data = org_data.map(lambda x: (normalization_layer(x)))
covered_data = org_data.map(lambda x: (normalization_layer(x)))

#x_data = np.concatenate(org_data, covered_data)

#först bild
#andra in och out
#Batch
#print(training_data[0][2][0].shape)

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

model.fit(covered_data, org_data, epochs=1)

#model.evaluate(model, training_data[0])

regenerated = model.predict(training_data[0])

import matplotlib.pyplot as plt
n = 5
plt.figure(figsize=(20, 4))
for i in range(n):
  # Original image
  ax = plt.subplot(3, n, i + 1)
  plt.imshow(training_data[i][1][0])
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # Destroyed image
  ax = plt.subplot(3, n, i + n + 1)
  plt.imshow(training_data[i][0][0])
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # Regenerated image
  ax = plt.subplot(3, n, i + 2*n + 1)
  plt.imshow(regenerated[i])
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

plt.show()
