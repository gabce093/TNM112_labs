
import data_generator
from tensorflow import keras
from tensorflow.keras import layers
import util
from tensorflow.keras import regularizers


data = data_generator.DataGenerator()


def conv_block(x, N, channels, kernel_size, activation, padding='same'):
    for i in range(N):
        x = layers.Conv2D(channels, kernel_size=kernel_size, activation=activation, padding=padding)(x)
        # x = layers.Dropout(0.2)(x)
    return layers.MaxPooling2D(pool_size=(2, 2))(x)


epochs = 30
batch_size = 128

# Load the PatchCamyleon dataset
# In this dataset, we don't have labels for the test set.
# Do your development by monitoring the validation performance,
# and when you are finished you will run predictions on the test
# set and produce a CSV file that you can upload to Kaggle.
data = data_generator.DataGenerator()
data.generate(dataset='patchcam')

keras.backend.clear_session()

# TODO: Build your network here
x = layers.Input(shape=data.x_train.shape[1:])
aug_flip = layers.RandomFlip(mode="horizontal_and_vertical")(x)
conv1 = conv_block(aug_flip, N=2, channels=8, kernel_size=(3, 3), activation='relu', padding='same')
pool1 = layers.MaxPooling2D()(conv1)
conv2 = conv_block(pool1, N=2, channels=16, kernel_size=(3, 3), activation='relu', padding='same')

flat1 = layers.Flatten()(conv2)

dense1 = layers.Dense(512, kernel_regularizer=regularizers.L1L2(0.0001, 0.01),  activation='relu')(flat1)
#norm1 = layers.BatchNormalization()(dense1)

dense2 = layers.Dense(512, kernel_regularizer=regularizers.L1L2(0.0002, 0.001), activation='relu')(dense1)
drop2 = layers.Dropout(0.02)(dense2)
#norm2 = layers.BatchNormalization()(drop2)

dense3 = layers.Dense(512, kernel_regularizer=regularizers.L1L2(0.0002, 0.001),  activation='relu')(drop2)
#norm3 = layers.BatchNormalization()(dense3)
drop3 = layers.Dropout(0.02)(dense3)

dense4 = layers.Dense(512, kernel_regularizer=regularizers.L1L2(0.0002, 0.002),  activation='relu')(drop3)



y = layers.Dense(data.K, activation='softmax')(dense4)

model = keras.models.Model(inputs=x, outputs=y)
model.summary()

opt = keras.optimizers.Adam(weight_decay=0.05)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', 'AUC'])
log = model.fit(data.x_train, data.y_train_oh, batch_size=batch_size, epochs=epochs,
                validation_data=(data.x_valid, data.y_valid_oh), validation_freq=1,
                verbose=True)

util.evaluate(model, data)
util.plot_training(log)

# TODO: When you have finished your model development, you should
# run inference on the test set and export a CSV file that can be
# uploaded to Kaggle
# util.pred_test(model, data, name='your_submission.csv')