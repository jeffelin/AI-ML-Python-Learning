import tensorflow as tf 
import matplotlib.pyplot as plt 

def looping_dataset(x_train): 
    counts = len(x_train) 
    print("These are all the mnist, but it is capped at 5 to not crash kernel! There are a total of " + str(counts)) 
    for digits in range(counts): 
        if digits < 5: # capping at 5 
            plt.imshow(x_train[digits])
            plt.show() 