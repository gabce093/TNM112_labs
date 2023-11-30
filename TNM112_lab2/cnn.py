import numpy as np
from scipy import signal
import skimage
import data_generator

# Different activations functions
def activation(x, activation):
    if activation == 'linear':
        return x
    elif activation == 'relu':
        return np.maximum(np.zeros(x.shape), x)
    elif activation=='sigmoid':
        return 1/(1+np.exp(-x))
    elif activation=='softmax':
        return np.divide(np.exp(x),np.sum(np.exp(x)[0]))
    else:
        raise Exception("Activation function is not valid", activation) 

# 2D convolutional layer
def conv2d_layer(h,     # activations from previous layer, shape = [height, width, channels prev. layer]
                 W,     # conv. kernels, shape = [kernel height, kernel width, channels prev. layer, channels this layer]
                 b,     # bias vector
                 act    # activation function
):
    # TODO: implement the convolutional layer
    # 1. Specify the number of input and output channels
    CI = W.shape[2]# Number of input channels
    CO = W.shape[3]# Number of output channels
    
    # 2. Setup a nested loop over the number of output channels 
    #    and the number of input channels
    h_out = np.zeros((h.shape[0],h.shape[1],CO))
    for i in range(CI):
        for j in range(CO):
    # 3. Get the kernel mapping between channels i and j
            kernel = W[:,:,i,j];
            
    # 4. Flip the kernel horizontally and vertically (since
    #    We want to perform cross-correlation, not convolution.
    #    You can, e.g., look at np.flipud and np.fliplr
            kernel = np.fliplr(kernel)
            kernel = np.flipud(kernel)

    # 5. Run convolution (you can, e.g., look at the convolve2d
    #    function in the scipy.signal library)
    # 6. Sum convolutions over input channels, as described in the 
    #    equation for the convolutional layer
            image = h[:,:,i] # i or j? 
            #TODO: Pad with zeros 
            h_out[:,:,j] += signal.convolve2d(image,kernel,mode='same')
            
    # 7. Finally, add the bias and apply activation function
    for i in range(CO):
        h_out[:,:,i] += b[i] 
    h_out = activation(h_out,act) 
    
    return h_out 

# 2D max pooling layer
def pool2d_layer(h):  # activations from conv layer, shape = [height, width, channels]
    # TODO: implement the pooling operation
    # 1. Specify the height and width of the output
    sx = int(h.shape[0]/2)
    sy = int(h.shape[1]/2)

    # 2. Specify array to store output
    ho = np.zeros((sx,sy,h.shape[2]))

    # 3. Perform pooling for each channel.
    #    You can, e.g., look at the measure.block_reduce() function
    #    in the skimage library
    for i in range(ho.shape[2]):
        ho[:,:,i] = skimage.measure.block_reduce(h[:,:,i], block_size=2, func=np.max)
    return ho


# Flattening layer
def flatten_layer(h): # activations from conv/pool layer, shape = [height, width, channels]
    # TODO: Flatten the array to a vector output.
    # You can, e.g., look at the np.ndarray.flatten() function
    return np.ndarray.flatten(h)
    
    
# Dense (fully-connected) layer
def dense_layer(h,   # Activations from previous layer
                W,   # Weight matrix
                b,   # Bias vector
                act  # Activation function
):

    
    # TODO: implement the dense layer.
    # You can use the code from your implementation
    # in Lab 1. Make sure that the h vector is a [Kx1] array.
    ho = (W @ h)
    ho = np.add(ho,np.transpose(b))
    ho = activation(ho, act)
    return ho
    
#---------------------------------
# Our own implementation of a CNN
#---------------------------------
class CNN:
    def __init__(
        self,
        dataset,         # DataGenerator
        verbose=True     # For printing info messages
    ):
        self.verbose = verbose
        self.dataset = dataset

    # Set up the CNN from provided weights
    def setup_model(
        self,
        W,                   # List of weight matrices
        b,                   # List of bias vectors
        lname,               # List of layer names
        activation='relu'    # Activation function of layers
    ):
        self.activation = activation
        self.lname = lname

        self.W = W
        self.b = b

        # TODO: specify the total number of weights in the model
        #       (convolutional kernels, weight matrices, and bias vectors)
        self.N = 0
        
        for l in range(len(self.lname)):
            if self.lname[l] == 'conv':
                self.N += W[l].size + b[l].size
            elif self.lname[l] == 'dense':
                self.N += W[l].size + b[l].size
            
        print('Number of model weights: ', self.N)
        
    # Feedforward through the CNN of one single image
    def feedforward_sample(self, h):

        # Loop over all the model layers
        for l in range(len(self.lname)):
            act = self.activation
            
            if self.lname[l] == 'conv':
                h = conv2d_layer(h, self.W[l], self.b[l], act)
            elif self.lname[l] == 'pool':
                h = pool2d_layer(h)
            elif self.lname[l] == 'flatten':
                h = flatten_layer(h)
            elif self.lname[l] == 'dense':
                if l==(len(self.lname)-1):
                    act = 'softmax'
                h = dense_layer(h, self.W[l], self.b[l], act)
        return h

    # Feedforward through the CNN of a dataset
    def feedforward(self, x):
        # Output array
        y = np.zeros((x.shape[0],self.dataset.K))

        # Go through each image
        for k in range(x.shape[0]):
            if self.verbose and np.mod(k,1000)==0:
                print('sample %d of %d'%(k,x.shape[0]))

            # Apply layers to image
            y[k,:] = self.feedforward_sample(x[k])   
            
        return y

    # Measure performance of model
    def evaluate(self):
        print('Model performance:')

        # TODO: formulate the training loss and accuracy of the CNN.
        # Assume the cross-entropy loss.
        # For the accuracy, you can use the implementation from Lab 1.
        train_pred_score = self.feedforward(self.dataset.x_train) 
        train_pred = np.argmax(train_pred_score, axis=1)
        
        train_loss = 0
        for i in range(len(self.dataset.y_train)):
            train_loss -= np.log(train_pred_score[i,self.dataset.y_train[i]])
        train_loss = train_loss/len(self.dataset.y_train)
        train_acc = (self.dataset.y_train == train_pred).sum()/ train_pred.size * 100
        print("\tTrain loss:     %0.4f"%train_loss)
        print("\tTrain accuracy: %0.2f"%train_acc)

        # TODO: formulate the test loss and accuracy of the CNN

        
        test_pred_score = self.feedforward(self.dataset.x_test) 
        test_pred = np.argmax(test_pred_score, axis=1)
        
        test_loss = 0
        for i in range(len(self.dataset.y_test)):
            test_loss -= np.log(test_pred_score[i,self.dataset.y_test[i]])
        test_loss = test_loss/len(self.dataset.y_test)
        test_acc = (self.dataset.y_test == test_pred).sum()/ test_pred.size * 100
        print("\tTest loss:      %0.4f"%test_loss)
        print("\tTest accuracy:  %0.2f"%test_acc)
