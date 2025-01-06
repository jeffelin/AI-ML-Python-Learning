

import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt 

# reading and uploading data 

data = pd.read_csv('')
c_data = np.array(data)
m,n = data.shape
np.random.shuffle(c_data)

# working with data 

data_dev = c_data[0:1000].T # first 1000 examples 
Y_dev = data[0] # first 
X_dev = data[1:n] # column 

data_train = data[1000:m].T # up to row m 
Y_train = data_train[0] # first 
X_train = data_train[1:n] # column 

# These above are all arrays 

# functions 

def init_parameters():
    W1 = np.random.rand(10,784) 
    b1 = np.random.rand(10,1) 
    W2 = np.random.rand(10,10) 
    b2 = np.random.rand(10,1)
    return W1, b1, W2, b2 

def ReLU(z):
    return np.maximum(z,0)

def softmax(A):
    return np.exp(A) / np.sum(np.exp(A))

def two_layer_foward_propagation(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1 
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

def one_hot(Y): 
    newY = np.zeros((Y.size, Y.max() +1 )) # creates correctly sized matrix size 10 
    newY[np.arange(Y.size), Y] = 1 # 0 to m training 
    Y_T = newY.T
    return Y_T

def derivative_ReLU(z): 
    return Z>0 

def backward_propagation(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    newY = one_hot(Y) 
    dZ2 = A2 - newY
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, 2)
    dZ1 = W2.T.dot(dZ2) * derivative_ReLU(Z1) 
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, 2)
    return dW1, dW2. db1, db2 

# still a little confused on the math as I probably would not be able to do it on my own 

def update_parameters(W1, b1, W2, b2, dW1, dW2, db1, db2, alpha):
    W1 = W1 - a*dW1
    b1 = b1 - a*db1
    W2 = W2 - a*dW2
    b2 = b2 - a*db2

# calling and aggregating the functions 

def get_predictions(A2): 
    return np.argmax(A2, 0)

def get_accuracy (predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    W1, W2, b1, b2 = init_parameters() 
    for i in range(iterations):
        Z1, A1, Z2, A2, = two_layer_foward_propagation( W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_propagation(Z1, A1, Z2, A2,X, Y)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, dW2, db1, db2, alpha)
        if i % 50 == 0:
            print("Iteration:", i)
            print("Accuracy",get_accuracy(get_predictions(A2), Y))

        return W1, W2, b1, b2

W1, W2, b1, b2 = gradient_descent(X_train, Y_train, 100, 0.1)


# def n_looped_fp(W, b, X): 
#     for n in n:
#         Z = W.dot(X) + b 
#         A = ReLU(Z)
#         if 

# You can observe here that it will be manually exchaustive to compute abitrary hidden layers. for loop would be nice, though the attempt is ultimately botched as you still need to compute all non-given Wn+1 and bn+1
