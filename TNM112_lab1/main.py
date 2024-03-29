import numpy as np
import mlp
import importlib
import data_generator
import keras_mlp
from tensorflow import keras
importlib.reload(mlp)

data = data_generator.DataGenerator()
data.generate(dataset='linear', N_train=512, N_test=512, K=2, sigma=0.05) # Task 1.3
model_k = keras_mlp.KerasMLP(data, verbose=True)
activation = 'softmax'

hidden_layers = 0
layer_width = 10
batch_size = 32
epochs = 20            # Number of epochs for the training
batch_size = 32    # Batch size to use in training
loss = keras.losses.MeanSquaredError() # Loss function
initial_learning_rate = 1.0
opt = keras.optimizers.Adam() # Optimizer
init = keras.initializers.glorot_normal()
# Setup the model with the specified hyper parameters
model_k.setup_model(hidden_layers=hidden_layers, layer_width=layer_width,
                    activation=activation, init=init)

# Compile the model with a loss function and optimizer
model_k.compile(loss_fn=loss, optimizer=opt)

# Train the model a certain number of epochs with a specified batch size
model_k.train(epochs=epochs, batch_size=batch_size)

# Plot the training progress
model_k.plot_training()

# Evaluate the model (loss and accuracy on the training and test data)

model_k.evaluate()
# Plot the dataset with decision boundaries generated by the trained model
#data.plot_classifier(model_k)
# Get the weight matrices and biases of the trained Keras model
#W, b = model_k.get_weights()

# Task 3: specify a weight matrix and a bias vector
W = [np.array([[-2, -2], [-1, -1]])]
b = [np.array([[0], [-1]])]

# This is our implementation of an MLP, which we set to use the dataset we generated
model = mlp.MLP(data)

# Assign the weights and biases to the MLP and specify the activation function
model.setup_model(W, b, activation=activation)

# Evaluate the model (accuracy on the training and test data)
model.evaluate()

# Plot the dataset with decision boundaries generated by our MLP
data.plot_classifier(model)