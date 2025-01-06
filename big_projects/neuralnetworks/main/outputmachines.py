import numpy as np 
import math 

def output_machine_1(): 
    inputs = [1,2,3,2.5]
    weights = [0.2,0.8,-0.5,1.0]
    bias = 2
    output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + inputs[3]*weights[3] + bias 
    return print(output)

def output_machine_2(): 
    inputs = [1,2,3,2.5]
    weights_1 = [0.2,0.8,-0.5,1.0]
    weights_2 = [0.5,-0.91,0.26,-0.5]
    weights_3 = [-0.26,-0.27,0.17,0.87]
    bias_1 = 2
    bias_2 = 3
    bias_3 = 0.5 
    output = [inputs[0]*weights_1[0] + inputs[1]*weights_1[1] + inputs[2]*weights_1[2] + inputs[3]*weights_1[3] + bias_1, inputs[0]*weights_2[0] + inputs[1]*weights_2[1] + inputs[2]*weights_2[2] + inputs[3]*weights_2[3] + bias_2, inputs[0]*weights_3[0] + inputs[1]*weights_3[1] + inputs[2]*weights_3[2] + inputs[3]*weights_3[3] + bias_3]
    return print(output)

def neuron(inputs, weights, biases, layer_outputs): 
  for neuron_weights, neuron_biases in zip(weights, biases):
        neuron_output = 0 
        for n_input, weight in zip(inputs, neuron_weights):
            neuron_output += n_input*weight
        neuron_output += neuron_biases
        layer_outputs.append(neuron_output)

def output_machine_3():
    inputs = [1,2,3,2.5]
    weights = [[0.2,0.8,-0.5,1.0], [0.5,-0.91,0.26,-0.5], [-0.26,-0.27,0.17,0.87]]
    biases = [2,3,0.5]
    layer_outputs = [] 
    neuron(inputs, weights, biases, layer_outputs)
    return print(layer_outputs)


import numpy as np 

def output_machine_4():
    inputs = [1,2,3,2.5]
    weights = [[0.2,0.8,-0.5,1.0], 
           [0.5,-0.91,0.26,-0.5], 
           [-0.26,-0.27,0.17,0.87]]
    biases = [2,3,0.5]
    output = np.dot(weights, inputs) + biases
    return print(output)

def output_machine_5(): 

    inputs = [[1,2,3,2.5],
          [2.0,5.0,-1.0,2.0],
          [-1.5,2.7,3.3,-0.8]] 
     
    weights_1 = [[0.2,0.8,-0.5,1.0], 
           [0.5,-0.91,0.26,-0.5], 
           [-0.26,-0.27,0.17,0.87]]

    biases_1 = [2,3,0.5]


    weights_2 = [[0.1,-0.14,0.5], 
           [-0.5,0.12,-0.33], 
           [-44,0.73,-0.13]]

    biases_2 = [-1,2,-0.5]

    layer_1_outputs = np.dot(inputs, np.array(weights_1).T) + biases_1
    layer_2_outputs = np.dot(layer_1_outputs, np.array(weights_2).T) + biases_2

    return print(layer_2_outputs)

def output_machine_6(): 

    X_training_dataset = [[1,2,3,2.5],
          [2.0,5.0,-1.0,2.0],
          [-1.5,2.7,3.3,-0.8]] 
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

    class Layer_Dense: 
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
        
    X_training_dataset, y = spiral_data(100,3) 
    dense_layer_1 = Layer_Dense(2,3) 
    activation_1 = Activation_RELU()
    dense_layer_2 = Layer_Dense(3,3) 
    activation_2 = Activation_softmax() 

    dense_layer_1.forward(X_training_dataset)
    activation_1.forward(dense_layer_1.output)
    dense_layer_2.forward(activation_1.output)
    activation_2.forward(dense_layer_2.output)
    print(activation_2.output[:5])

    loss_function = Loss_Categorical_Cross_Entropy() 
    loss = loss_function.calculate(activation_2.output, y) 
    print("Loss:", loss)




def ReLU(inputs): 
    outputs = []

    for i in inputs: 
        outputs.append(max(0,i))
    return print(outputs)

def softmax_raw_math(): 
    layer_outputs = [4.8,1.21,2.385]
    exp_values = [] 
    E = math.e
    for output in layer_outputs:
        exp_values.append(E**output)
    norm_base = sum(exp_values)
    norm_values = [] 
    for value in exp_values: 
        norm_values.append( value / norm_base)

    return print(sum(norm_values)) 

import numpy as np

def softmax(): 
    layer_outputs = [[4.8,1.21,2.385], 
                     [8.9,-1.81,0.2],
                     [1.41,1.051,0.026]] 
    exp_values = np.exp(layer_outputs)
    norm_values = exp_values / np.sum(exp_values,axis = 1, keepdims = True)
    print(norm_values)

import math 

def categorical_cross_entropy(): 
    softmax_output = [0.7,0.1,0.2] 
    target_output = [1,0,0] 
    loss = -(math.log(softmax_output[0])*target_output[0] + math.log(softmax_output[1])*target_output[1] + math.log(softmax_output[2])*target_output[2])
    new_loss = -math.log(softmax_output[0]*target_output[0])
    return print(loss) 

def implement_categorical_loss(): 
    softmax_outputs = np.array( [0.7,0.1,0.2],  [0.1,0.5,0.4],  [0.02,0.9,0.8])
    class_targets = [0,1,1] 
    return print(-np.log(softmax_outputs[[0,1,2], class_targets])) 

def testing(): 
    output_machine_1()
    output_machine_2() 
    output_machine_3()
    output_machine_4() 
    output_machine_5() 
    ReLU ([1,2,3,2.5]) 
    softmax_raw_math()
    softmax
    categorical_cross_entropy()
    output_machine_6() 
