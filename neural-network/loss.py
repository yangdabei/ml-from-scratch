import numpy as np

# Loss functions used to calculate loss at final layer.

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2));

def mse_prime(y_true, y_pred):
    return 2*(y_pred - y_true) / y_true.size;