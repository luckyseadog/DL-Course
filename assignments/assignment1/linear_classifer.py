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
    

def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''
    
    target_index = np.array([target_index])
    target_index = target_index.reshape(target_index.size)
    flag = len(predictions.shape) == 1
    
    if flag:
        predictions = predictions.reshape(1, predictions.shape[0])
        
        
       
    
    predict = predictions.copy() 
    probs = softmax(predict)
    loss = cross_entropy_loss(probs, target_index)
   
    dprobs = probs.copy()
    if not(len(predictions.shape) == 1):
        dprobs[np.arange(dprobs.shape[0]), target_index.reshape(target_index.size)] -= 1 #target_index - dprobs
 
    else:
        dprobs = dprobs.reshape(dprobs.size)
        dprobs[target_index.reshape(target_index.size)[0]] -= 1
        
    
    dprobs /= dprobs.shape[0]
    
    if flag:
        dprobs = dprobs.reshape(dprobs.size)

    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops

    return loss, dprobs

    
    

    


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    loss = (reg_strength * W ** 2).sum()
    grad = reg_strength * 2 * W

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)
    loss, dpred = softmax_with_cross_entropy(predictions, target_index)
    dW = X.T @ dpred
    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops


    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
                
            
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            
            for batch_ind in batches_indices:
                loss1, grad1 = linear_softmax(X[batch_ind], self.W, y[batch_ind])
                loss2, grad2 = l2_regularization(self.W, reg)
                loss = loss1 + loss2
                grad = grad1 + grad2
                self.W = self.W - learning_rate * grad
                
                
            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!

            # end
            print("Epoch %i, loss: %f" % (epoch, loss))
            loss_history.append(loss)

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)
        y_pred = np.argmax(softmax(X @ self.W), axis = 1)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops

        return y_pred



                
                                                          

            

                
