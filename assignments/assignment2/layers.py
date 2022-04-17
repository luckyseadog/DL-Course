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
