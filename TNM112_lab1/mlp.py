import numpy as np
import data_generator

# Different activations functions
def activation(x, activation):
    if activation == 'linear':
        return x
    elif activation == 'ReLU':
        return np.maximum(np.zeros(x.shape), x)
    elif activation=='sigmoid':
        return 1/(1+np.exp(-x))
    elif activation=='softmax':
        s = np.exp(x) / sum(np.exp(x))
        return s
    else:
        raise Exception("Activation function is not valid", activation) 

#-------------------------------
# Our own implementation of an MLP
#-------------------------------
class MLP:
    def __init__(
        self,
        dataset,         # DataGenerator
    ):
        self.dataset = dataset

    # Set up the MLP from provided weights and biases
    def setup_model(
        self,
        W,                   # List of weight matrices
        b,                   # List of bias vectors
        activation='linear'  # Activation function of layers
    ):
        self.activation = activation

        # TODO: specify the number of hidden layers based on the length of the provided lists
        self.hidden_layers = len(W) - 1
        print(self.hidden_layers)

        self.W = W
        self.b = b

        # TODO: specify the total number of weights in the model (both weight matrices and bias vectors)
        self.N = 0
        for i in range(len(self.W)):
            self.N += W[i].size
            self.N += b[i].size

        print('Number of hidden layers: ', self.hidden_layers)
        print('Number of model weights: ', self.N)

    # Feed-forward through the MLP
    def feedforward(
        self,
        x      # Input data points
    ):
        # TODO: specify a matrix for storing output values
        y = np.zeros((x.shape[0], self.dataset.K))

        # TODO: implement the feed-forward layer operations
        print(x.shape)
        for datapoint in range(x.shape[0]-1):
            h = x[datapoint]
            for layer in range(self.hidden_layers):

               #print("W[]layer=",self.W[layer])
               #print("h.size before=", h.shape)

               #print("b=",self.b[layer])
               h= (self.W[layer] @ h)
               if(layer == 0):
                   h = h[:,np.newaxis]
               #print("h.size before b=", h.shape)
               h = np.add(h,self.b[layer])
               #print("b.size=", self.b[layer].shape)
               h = activation(h, self.activation)
               #print("h.size after=", h.shape)

            h = (self.W[self.hidden_layers] @ h)
            h = h[:, np.newaxis]
            h = np.add(h, self.b[self.hidden_layers])
            y[datapoint,:] = activation(h, 'softmax').flatten()
            #print("y.shape=", y.shape)
        # 1. Specify a loop over all the datapoints
        # 2. Specify the input layer (2x1 matrix)
        # 3. For each hidden layer, perform the MLP operations
        #    - multiply weight matrix and output from previous layer
        #    - add bias vector
        #    - apply activation function
        # 4. Specify the final layer, with 'softmax' activation
        return y

    # Measure performance of model
    def evaluate(self):
        print('Model performance:')

        # TODO: formulate the training loss and accuracy of the MLP
        # Assume the mean squared error loss
        # Hint: For calculating accuracy, use np.argmax to get predicted class

        score = self.feedforward(self.dataset.x_train)
        train_pred = np.argmax(score, axis=1)
        #print(self.dataset.y_train.shape)
       # print("trainpred",train_pred.shape)
        train_loss = pow(train_pred-self.dataset.y_train,2).mean()
        train_acc = (self.dataset.y_train == train_pred).sum()/ train_pred.size
        print("\tTrain loss:     %0.4f"%train_loss)
        print("\tTrain accuracy: %0.2f"%train_acc)

        # TODO: formulate the test loss and accuracy of the MLP
        score = self.feedforward(self.dataset.x_test)
        train_pred = np.argmax(score, axis=1)
        test_loss =pow(train_pred-self.dataset.y_test,2).mean()
        test_acc =  (self.dataset.y_test == train_pred).sum()/ train_pred.size
        print("\tTest loss:      %0.4f"%train_loss)
        print("\tTest accuracy:  %0.2f"%test_acc)
