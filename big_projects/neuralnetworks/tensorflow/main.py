import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 

# data import 
mnist = tf.keras.datasets.mnist # 28 * 28 

# data viz 

(x_train, y_train) , (x_test, y_test) = mnist.load_data() 

# model 

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

model = tf.keras.models.Sequential() # selecting model 
model.add(tf.keras.layers.Flatten()) # input layers 
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) # hidden layers to the activation function 
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) # hidden layers 
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax)) # output layer 

# loss, accuracy, optimizer 

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']) # loss, optimizer 
model.fit(x_train, y_train, epochs = 3)
val_loss, val_acc = model.evaluate(x_test, y_test)

# my own functions to loop stuff

def looping_dataset(x_train): 
    counts = len(x_train) 
    print("These are all the mnist, but it is capped at 5 to not crash kernel! There are a total of " + str(counts)) 
    for digits in range(counts): 
        if digits < 5: # capping at 5 
            plt.imshow(x_train[digits])
            plt.show() 

def looping_testing_model(model): 
    predictions = model.predict([x_test])
    counts = len(x_test)
    for digits in range(counts): 
        if digits < 5: # capping at 5 
            print(np.argmax(predictions[0]))
            plt.imshow(x_train[digits])
            plt.show() 
            