import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        # TODO Create necessary layers
        self.reg = reg
        self.layer1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu = ReLULayer()
        self.layer2 = FullyConnectedLayer(hidden_layer_size, n_output)
        
        

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        self.params()['L1_W'].grad = np.zeros_like(self.params()['L1_W'].grad)
        self.params()['L1_B'].grad = np.zeros_like(self.params()['L1_B'].grad)
        self.params()['L2_W'].grad = np.zeros_like(self.params()['L2_W'].grad)
        self.params()['L2_B'].grad = np.zeros_like(self.params()['L2_B'].grad)
        
        z2 = self.layer1.forward(X)
        a2 = self.relu.forward(z2)
        z3 = self.layer2.forward(a2)
        
        loss, d_out = softmax_with_cross_entropy(z3, y)
        
        d_out = self.layer2.backward(d_out)
        d_out = self.relu.backward(d_out)
        d_out = self.layer1.backward(d_out)
        
        
        

        
        reg_loss1, d1 = l2_regularization(self.layer2.W.value, self.reg)
        reg_loss2, d2 = l2_regularization(self.layer1.W.value, self.reg)
        reg_loss3, d3 = l2_regularization(self.layer2.B.value, self.reg)
        reg_loss4, d4 = l2_regularization(self.layer1.B.value, self.reg)
        
#         loss /= X.shape[0]
#         self.layer1.B.grad /= X.shape[0]
#         self.layer2.B.grad /= X.shape[0]
#         self.layer1.W.grad /= X.shape[0]
#         self.layer2.W.grad /= X.shape[0]
        
        loss+= reg_loss1
        self.layer2.W.grad += d1
        loss += reg_loss2
        self.layer1.W.grad += d2
        loss += reg_loss3
        self.layer2.B.grad += d3
        loss += reg_loss4
        self.layer1.B.grad += d4
        
        
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        z2 = self.layer1.forward(X)
        a2 = self.relu.forward(z2)
        z3 = self.layer2.forward(a2)
        pred = np.argmax(softmax(z3), axis=1)

        return pred
         

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params

        result = {'L1_W': self.layer1.params()['W'],  'L1_B': self.layer1.params()['B'],
                  'L2_W': self.layer2.params()['W'],  'L2_B': self.layer2.params()['B']}

        return result
