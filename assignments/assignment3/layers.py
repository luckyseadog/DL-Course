import numpy as np

def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''

    pred = predictions.copy()
    if len(pred.shape) > 1:
        pred -= np.amax(pred,axis=1).reshape(predictions.shape[0], 1)
        pred = np.exp(pred)
        pred  /= np.sum(pred, axis=1).reshape(pred.shape[0], 1)
    else:
        pred = pred.reshape(1, pred.shape[0])
        pred -= np.amax(pred)
        pred = np.exp(pred)
        pred /= np.sum(pred)
        pred = pred.reshape(pred.size)
    return pred
        



def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    
    
    if len(probs.shape) > 1:
        target_index = target_index.reshape(target_index.size, 1)
        loss = np.mean(-np.log(probs[np.arange(target_index.shape[0]), target_index.reshape(target_index.size)])) 
    else:
        target_index = np.array([target_index])
        target_index = target_index.reshape(target_index.size)
        loss = -np.log(probs[target_index[0]])
    
    
    return loss

def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    loss = (reg_strength * W ** 2).sum()
    grad = reg_strength * 2 * W
    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    target_index = np.array([target_index])
    target_index = target_index.reshape(target_index.size)
    flag = len(preds.shape) == 1
    
    if flag:
        preds = preds.reshape(1, preds.shape[0])
        
        
       
    
    predict = preds.copy() 
    probs = softmax(predict)
    loss = cross_entropy_loss(probs, target_index)
   
    d_preds = probs.copy()
    if not(len(preds.shape) == 1):
        d_preds[np.arange(d_preds.shape[0]), target_index.reshape(target_index.size)] -= 1 #target_index - dprobs
 
    else:
        d_preds = d_preds.reshape(d_preds.size)
        d_preds[target_index.reshape(target_index.size)[0]] -= 1
        
    
    d_preds /= d_preds.shape[0]
    
    if flag:
        d_preds = d_preds.reshape(dprobs.size)

    return loss, d_preds


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.val_X = X.copy()
        
        return np.maximum(X, 0)
        

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result = np.array(self.val_X > 0) * d_out
        
        assert d_out.shape == d_result.shape
        
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.1 * np.random.randn(n_input, n_output)) #0.001
        self.B = Param(0.1 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X.copy()
        
        return X @ self.W.value + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        self.W.grad = self.X.T @ d_out
        self.B.grad =  np.ones((1, d_out.shape[0])) @ d_out
        
        
        d_input = d_out @ self.W.value.T

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}



    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X
        
        if self.padding:
            self.X = np.zeros((batch_size, height + 2*self.padding, width + 2*self.padding, channels))
            self.X[:, self.padding:-self.padding, self.padding:-self.padding, :] = X
        
        
        batch_size, height, width, channels = self.X.shape

        out_height = height - self.filter_size + 1
        out_width = width - self.filter_size + 1
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        output = np.zeros((batch_size, out_height, out_width, self.out_channels))
        
        for batch in range(batch_size):
            for y in range(out_height):
                for x in range(out_width):
                    for ch in range(self.out_channels):
                        X_step = self.X[batch, y:y+self.filter_size, x:x+self.filter_size]
                        W_step = self.W.value[:, :, :, ch]
                        output[batch, y, x, ch] = X_step.flatten() @ W_step.flatten() + self.B.value[ch]
                        
                
                
#                 X_yx = X[:, y:y+self.filter_size, x:x+self.filter_size, :]
#                 X_yx = np.reshape(X_yx, (batch_size, self.filter_size*self.filter_size*channels))
#                 W_yx = np.reshape(self.W.value, (self.filter_size*self.filter_size*channels, self.out_channels))
#                 output[y, x] = X_yx @ W_yx + self.B.value
                
#         output = np.reshape(output, (batch_size, out_height, out_width, self.out_channels))
        
        
        return output


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        
        d_in = np.zeros((batch_size, height, width, channels))

        # Try to avoid having any other loops here too
        for batch in range(batch_size):
            for y in range(out_height):
                for x in range(out_width):
                    for ch in range(self.out_channels):
                        d_in[batch, y:y+self.filter_size, x:x+self.filter_size, :] += d_out[batch, y, x, ch] * \
                                                                                        self.W.value[:, :, :, ch]

                        self.W.grad[:, :, :, ch] += d_out[batch, y, x, ch] * \
                                                    self.X[batch, y:y+self.filter_size, x:x+self.filter_size, :] 
                        self.B.grad[ch] += d_out[batch, y, x, ch]
                        
                        
                
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
        if self.padding:
            d_in = d_in[:, self.padding:-self.padding, self.padding:-self.padding, :]
                

        return d_in

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        
        self.X = X
        batch_size, height, width, channels = X.shape
        
        out_height, out_width = (height - self.pool_size) // self.stride + 1, (width - self.pool_size) // self.stride + 1 
        
        X_out = np.zeros((batch_size, out_height, out_width, channels))
        
        for batch in range(batch_size):
            for y in range(out_height):
                for x in range(out_width):
                    for ch in range(channels):
                        X_out[batch, y, x, ch] =np.max(X[batch, y * self.stride:y * self.stride + self.pool_size, \
                                                        x * self.stride:x * self.stride + self.pool_size, ch])
                        
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        return X_out

    def backward(self, d_out):                                             
                                                       
                                                       
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, _ = d_out.shape
        
        d_in = np.zeros((batch_size, height, width, channels))
        
        for batch in range(batch_size):
            for y in range(out_height):
                for x in range(out_width):
                    for ch in range(channels):
                        flatten_ind = np.argmax(self.X[batch, y * self.stride:y * self.stride + self.pool_size, \
                                                        x * self.stride:x * self.stride + self.pool_size, ch].flatten())
                        
                        x_max = flatten_ind % self.pool_size
                        y_max = flatten_ind // self.pool_size
                        
                        
                        d_in[batch, y_max + y * self.stride, x_max + x * self.stride, ch] += d_out[batch, y, x, ch]
        
        
        
        
        
        return d_in

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        self.shape = X.shape
        batch_size, height, width, channels = self.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        X = np.reshape(X, (batch_size, height*width*channels))
        return X

    def backward(self, d_out):
        return np.reshape(d_out, self.shape)

    def params(self):
        # No params!
        return {}