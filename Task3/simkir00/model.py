import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
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
        
        self.conv1 = ConvolutionalLayer(input_shape[2], conv1_channels, 3, 1)
        self.relu1 = ReLULayer()
        self.pool1 = MaxPoolingLayer(4, 4)
        
        self.conv2 = ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1)
        self.relu2 = ReLULayer()
        self.pool2 = MaxPoolingLayer(4, 4)
        
        self.flat = Flattener()
        self.fc = FullyConnectedLayer(4 * conv2_channels, n_output_classes)
        
        self.model = [self.conv1, self.relu1, self.pool1,
                      self.conv2, self.relu2, self.pool2,
                      self.flat, self.fc]

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
        for layer in self.model:
            layer.reset_grad()
#         params = self.params()
#         for key in params.keys():
#             params[key].grad = np.zeros_like(params[key].value)
        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        x = X.copy()
        for layer in self.model:
            x = layer.forward(x)
        
        loss, d_out = softmax_with_cross_entropy(x, y)
        
        for layer in reversed(self.model):
            d_out = layer.backward(d_out)
        
        return loss
        
    def predict(self, X):
        # You can probably copy the code from previous assignment
        pred = X
        for layer in self.model:
            pred = layer.forward(pred)
        pred = pred.argmax(axis = 1)
        
        return pred

    def params(self):
        # TODO: Aggregate all the params from all the layers
        # which have parameters
        result = {'Conv_W1': self.conv1.W, 'Conv_B1': self.conv1.B,
                  'Conv_W2': self.conv2.W, 'Conv_B2': self.conv2.B,
                  'FC_W': self.fc.W, 'FC_B':self.fc.B}

        return result
