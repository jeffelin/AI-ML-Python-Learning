
# Output Machines for testing cases and better efficiencies

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
  for neuron_weights, neuron_biases in zip(weights, biases): # zip combines two list into a list for that element. 
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

def output_machine_4(): # dot product with numpy 
    inputs = [1,2,3,2.5]
    weights = [[0.2,0.8,-0.5,1.0], 
           [0.5,-0.91,0.26,-0.5], 
           [-0.26,-0.27,0.17,0.87]]
    biases = [2,3,0.5]
    output = np.dot(weights, inputs) + biases
    return print(output)


# cross product multiplies each row of a matrix with the corresponding columns of another matrix 
# transpose inverts columns into the rows 

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

np.random.seed(0) 

def output_machine_6(): 

    X_training_dataset = [[1,2,3,2.5],
          [2.0,5.0,-1.0,2.0],
          [-1.5,2.7,3.3,-0.8]] 


# Biases = 0 might initialize to 0 but cannot do this all the time as it can keep looping *0 "called a dead network", so maybe it can be non zero values. 

# Weights = random so we initialize weights with random values from 0 to 1 to prevent explosion. 

# Making this an object. 
    class Layer_Dense: 
        def __init__(self, n_inputs, n_neurons):
            self.weights = 0.10*np.random.randn(n_inputs, n_neurons) # this is the shape that it makes of the array 
            self.biases = np.zeros((1, n_neurons))
            # no need the transpose b/c it maps correctly 
        def forward(self, inputs):
            self.output = np.dot(inputs, self.weights) + self.biases 

    layer_1 = Layer_Dense(4, 5) # any values
    layer_2 = Layer_Dense(5,2) # first must correspond to the last 
    layer_1.forward(X_training_dataset) 
    layer_2.forward(layer_1.output)
    print(layer_2.output)
    
 


def testing(): 
    # output_machine_1()
    # output_machine_2() 
    # output_machine_3()
    # output_machine_4() 
    # output_machine_5() 
    output_machine_6() 

testing()