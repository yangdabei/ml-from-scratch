from layer import Layer
import numpy as np

class FCLayer(Layer):

    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = input_data @ self.weights + self.bias
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        input_error = output_error @ self.weights
        weights_error = output_error @ self.input

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
