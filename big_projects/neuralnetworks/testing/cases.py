

# Duplicated Cases 

# inputs = [1.2, 5.1, 2.1]
# weights = [3.1,2.1,8.7]
# bias = 3

# inputs = [1,2,3,2.5]
# weights = [0.2,0.8,-0.5,1.0]
# bias = 2
# output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + inputs[3]*weights[3] + bias 
# print(output)



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


