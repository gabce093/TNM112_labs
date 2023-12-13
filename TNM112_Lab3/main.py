# This is a sample Python script.
from keras import Sequential
from keras.src.layers import Conv3D, Resizing, UpSampling3D, MaxPooling3D
from tensorflow.keras import layers
from tensorflow import keras



from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D
import numpy as np

training_data = keras.utils.image_dataset_from_directory(
    directory='vegetable_images/train',
    labels='inferred',
    label_mode='categorical',
)






