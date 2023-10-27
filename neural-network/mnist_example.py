import numpy as np

from network import Network
from fully_connected_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime
from loss import mse, mse_prime

from keras.datasets import mnist
from keras.utils import to_categorical

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape and normalise train data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32') / 255

# convert train output to one-hot encoding
y_train = to_categorical(y_train)

# reshape and normalise test data
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32') / 255

# convert test output to one-hot encoding
y_test = to_categorical(y_test)

# create network
net = Network()
net.add(FCLayer(28*28, 100))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(100, 50))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(50, 10))
net.add(ActivationLayer(tanh, tanh_prime))

# train on 1000 samples
net.use(mse, mse_prime)
net.train(x_train[:1000], y_train[:1000], epochs=35, learning_rate=0.1)
# TODO implement mini-batch gradient descent

# test on 3 samples
out = net.predict(x_test[0:3])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:3])