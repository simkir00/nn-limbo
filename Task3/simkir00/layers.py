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
    if (predictions.ndim == 1):
        ax = 0
    else:
        ax = 1
        
    probs = np.exp(predictions - np.max(predictions, axis = ax).reshape(-1,1))
    probs = probs / np.sum(probs, axis = ax).reshape(-1, 1)
    
    return probs

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
    loss = 0
    if (probs.ndim == 1):
        loss = -np.log(probs[target_index])
    else:
        for i in range(probs.shape[0]):
            loss -= np.log(probs[i, target_index[i]])
        loss /= probs.shape[0]
        
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
    # TODO: Copy from the previous assignment
    loss = reg_strength * np.sum(W**2)
    grad = 2 * reg_strength * W
    
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
    # TODO: Copy from the previous assignment
    probs = softmax(preds)
    loss = cross_entropy_loss(probs, target_index)
    
    if (probs.ndim == 1):
        probs[target_index] -= 1
    else:
        for i in range(probs.shape[0]):
            probs[i, target_index[i]] -= 1
        probs /= probs.shape[0]
    dprediction = probs
    
    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO copy from the previous assignment
        self.mask = (X > 0)
        return X * self.mask

    def backward(self, d_out):
        # TODO copy from the previous assignment
        d_result = d_out * self.mask
        return d_result

    def params(self):
        return {}
    
    def reset_grad(self):
        pass

class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = X.copy()
        res = X.dot(self.W.value) + self.B.value
        return res

    def backward(self, d_out):
        # TODO copy from the previous assignment
        
        self.W.grad = self.X.T.dot(d_out)
        
        E = np.ones(shape = (1, self.X.shape[0]))
        self.B.grad = E.dot(d_out)
        
        return d_out.dot(self.W.value.T)

    def params(self):
        return { 'W': self.W, 'B': self.B }
    
    def reset_grad(self):
        self.W.grad = np.zeros_like(self.W.value)
        self.B.grad = np.zeros_like(self.B.value)
    
    
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
        self.X = None


    def forward(self, X):
        batch_size, height, width, channels = X.shape
        filter_size = self.filter_size
        
        #Padding adding
        pad = self.padding
        self.X = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
       
        out_height = (height + 2 * pad) - filter_size + 1
        out_width = (width + 2 * pad) - filter_size + 1
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        
        W = self.W.value.reshape((filter_size ** 2) * self.in_channels, self.out_channels)
        res = np.zeros((batch_size, out_height, out_width, self.out_channels))
        
        for y in range(out_height):
            for x in range(out_width):
                I = self.X[:, y : (y + filter_size), x : (x + filter_size), :]
                I = I.reshape((batch_size, (filter_size ** 2) * self.in_channels))
                res[:, y, x, :] = I.dot(W) + self.B.value
        
        return res

    
    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        filter_size = self.filter_size
        pad = self.padding
        
        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        
        W = self.W.value.reshape((filter_size ** 2) * self.in_channels, self.out_channels)
        d_res = np.zeros_like(self.X)
        
        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                grad = d_out[:, y, x, :]
                d_res[:, y : (y + filter_size), x : (x + filter_size), :] += \
                    (grad.dot(W.T)).reshape(batch_size, filter_size, filter_size, self.in_channels)
                
                I = self.X[:, y : (y + filter_size), x : (x + filter_size), :]
                I = I.reshape(batch_size, (filter_size ** 2) * self.in_channels)
                
                self.W.grad += (I.T.dot(grad)).reshape(self.W.value.shape)
                self.B.grad += np.sum(grad, axis = 0)
        
        if pad:
            d_res = d_res[:, pad : -pad, pad : -pad, :]
        
        return d_res

    def params(self):
        return { 'W': self.W, 'B': self.B }
    
    def reset_grad(self):
        self.W.grad = np.zeros_like(self.W.value)
        self.B.grad = np.zeros_like(self.B.value)

        
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
        batch_size, height, width, channels = X.shape
        pool_size = self.pool_size
        st = self.stride
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        
        out_height = (height - pool_size) // self.stride + 1
        out_width = (width - pool_size) // self.stride + 1
        
        self.X = X
        res = np.zeros((batch_size, out_height, out_width, channels))
        
        for y in range(out_height):
            for x in range (out_width):
                window = self.X[:, y * st : (y * st + pool_size), x * st : (x * st + pool_size), :]
                res[:, y, x, :] = np.max(window, axis = (1, 2))
        
        return res

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        pool_size = self.pool_size
        st = self.stride
        
        out_height = (height - pool_size) // self.stride + 1
        out_width = (width - pool_size) // self.stride + 1
        d_res = np.zeros_like(self.X)
        
        for y in range(out_height):
            for x in range (out_width):
                grad = d_out[:, y, x, :][:, np.newaxis, np.newaxis, :]
                window = self.X[:, y * st : y * st + pool_size, x * st : (x * st + pool_size), :]
                mask = (window == np.max(window, axis = (1, 2))[:, np.newaxis, np.newaxis, :])
                # print(grad.shape, mask.shape)
                d_res[:, y * st: (y * st + pool_size), x * st: (x * st + pool_size), :] += grad * mask
                
        return d_res

    def params(self):
        return {}
    
    def reset_grad(self):
        pass

class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape
        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
    
    def reset_grad(self):
        pass