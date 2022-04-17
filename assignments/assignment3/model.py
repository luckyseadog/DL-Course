import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization,
    softmax
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        
        self.conv1 = ConvolutionalLayer(input_shape[2], conv1_channels, filter_size=3, padding=1)
        self.relu1 = ReLULayer()
        self.maxpool1 = MaxPoolingLayer(4, 4)
        self.conv2 = ConvolutionalLayer(conv1_channels, conv2_channels, filter_size=3, padding=1)
        self.relu2 = ReLULayer()
        self.maxpool2 = MaxPoolingLayer(4, 4)
        self.flattern = Flattener()
        self.fullL = FullyConnectedLayer(input_shape[0] // 16 * input_shape[0] // 16 * conv2_channels, n_output_classes)
        
        


    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        self.params()['L1_W'].grad = np.zeros_like(self.params()['L1_W'].grad)
        self.params()['L1_B'].grad = np.zeros_like(self.params()['L1_B'].grad)
        self.params()['L2_W'].grad = np.zeros_like(self.params()['L2_W'].grad)
        self.params()['L2_B'].grad = np.zeros_like(self.params()['L2_B'].grad)
        self.params()['L3_W'].grad = np.zeros_like(self.params()['L3_W'].grad)
        self.params()['L3_B'].grad = np.zeros_like(self.params()['L3_B'].grad)
        
        z2 = self.conv1.forward(X)
        a2 = self.relu1.forward(z2)
        p2 = self.maxpool1.forward(a2)
        
        z3 = self.conv2.forward(p2)
        a3 = self.relu2.forward(z3)
        p3 = self.maxpool2.forward(a3)

        
        f = self.flattern.forward(p3)
        z4 = self.fullL.forward(f)
        loss, d_out = softmax_with_cross_entropy(z4, y)
        
        
        d_out = self.fullL.backward(d_out)
        d_out = self.flattern.backward(d_out)
        d_out = self.maxpool2.backward(d_out)
        d_out = self.relu2.backward(d_out)
        d_out = self.conv2.backward(d_out)
        d_out = self.maxpool1.backward(d_out)
        d_out = self.relu1.backward(d_out)
        d_out = self.conv1.backward(d_out)
        
        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        z2 = self.conv1.forward(X)
        a2 = self.relu1.forward(z2)
        p2 = self.maxpool1.forward(a2)
        
        z3 = self.conv2.forward(p2)
        a3 = self.relu2.forward(z3)
        p3 = self.maxpool2.forward(a3)
        
        f = self.flattern.forward(p3)
        z4 = self.fullL.forward(f)
        pred = np.argmax(softmax(z4), axis=1)
        return pred

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        result = {'L1_W': self.conv1.params()['W'],  'L1_B': self.conv1.params()['B'],
                  'L2_W': self.conv2.params()['W'],  'L2_B': self.conv2.params()['B'],
                  'L3_W': self.fullL.params()['W'],  'L3_B': self.fullL.params()['B'],}

        return result

    
