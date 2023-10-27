class Network():
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
    
    def add(self, layer):
        self.layers.append(layer)

    # loss function to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output
    def predict(self, input_data):
        samples = len(input_data)
        result = [] # what is this result?

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result
    
    # train network
    def train(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)

        for epoch in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output) # output is matrix
                
                # display
                err += self.loss(y_train[j], output)

                # dE/dY used in backprop to update weights and biases
                error = self.loss_prime(y_train[j], output) # scalar
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
                
            err /= samples
            print(f'epoch {epoch + 1}/{epochs}: error = {err}')
    