import numpy as np 
import math 
    
def spiral_data(points, classes):
        X = np.zeros((points*classes, 2))
        y = np.zeros(points*classes, dtype='uint8')
        for class_number in range(classes):
            ix = range(points*class_number, points*(class_number+1))
            r = np.linspace(0.0, 1, points)  # radius
            t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
            X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
            y[ix] = class_number
        return X, y


class Layers_Dense: 
        def __init__(self, n_inputs, n_neurons):
            self.weights = 0.10*np.random.randn(n_inputs, n_neurons) 
            self.biases = np.zeros((1, n_neurons))
        def forward(self, inputs):
            self.output = np.dot(inputs, self.weights) + self.biases 


class Activation_RELU: 
        def forward(self, inputs): 
            self.output = np.maximum(0, inputs)
    
class Activation_softmax: 
        def forward(self, inputs): 
            exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims= True))
            probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims= True)
            self.output = probabilities 
    
class Loss: 
    def calculate(self, output, y): 
        sample_losses = self.forward(output, y) 
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_Categorical_Cross_Entropy(Loss): 
    def forward(self, y_pred, y_true): 
        samples = len(y_pred) 
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape) ==1: 
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) ==2: 
            correct_confidences = np.sum((y_pred_clipped*y_true), axis =1) 
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods



# initializing 

points = 0
classes = 0 
d_1 = 1
d_2 = 1

X_training_dataset, y = spiral_data(points, classes)

dense_layer_1 = Layer_Dense(d_1,d_2) 
activation_1 = Activation_RELU()
dense_layer_2 = Layer_Dense(d_2,d_1) 
activation_2 = Activation_softmax() 

dense_layer_1.forward(X_training_dataset)
activation_1.forward(dense_layer_1.output)
dense_layer_2.forward(activation_1.output)
activation_2.forward(dense_layer_2.output)
print(activation_2.output)

loss_function = Loss_Categorical_Cross_Entropy() 
loss = loss_function.calculate(activation_2.output, y) 
print("Loss:", loss)