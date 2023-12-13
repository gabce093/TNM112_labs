# This is a sample Python script.
from keras import Sequential
from keras.src.layers import Conv3D, Resizing
from tensorflow.keras import layers
from tensorflow import keras



from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D
import numpy as np

training_data = keras.utils.image_dataset_from_directory(
    directory='vegetable_images/train/Bean',
    labels='inferred',
    label_mode='categorical',
)



model = Sequential()
model.add(Resizing(112, 112))
model.add(Conv2D(32, (3, 3), input_shape=(30, 30, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(training_data, batch_size=8, epochs=2)




