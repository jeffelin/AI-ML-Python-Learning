

# importing and setting 
# credits for helping -> https://www.youtube.com/@ProjectDataScience

import os 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import datasets
from sklearn.model_selection import train_test_split


sns.set()


# loading data set 
data = datasets.load_iris()
df = pd.DataFrame(data['data'], columns = data['feature_names'])

sns.pairplot(df, hue = 'target_name')

# training 

X_train = df_train.drop(columns = ['target', 'target_name']).values 
y_train = df_train['target'].values 


# models --> simple manual model 

def single_feature_prediction(petal_length):
    if petal_length < 2.5: 
        return 0 
    elif petal_length < 4.8:
        return 1
    else: 
        return 2
    
first_case = X_train[:,2 ]

manual_y_predictions = np.array([single_feature_prediction(val) for val in first_case])

manual_y_predictions == y_train #seeing true or false 

manual_model_accuracy = np.mean(manual_y_predictions == y_train)

print(f"Manual Model Accuracy: {manual_model_accuracy * 100:.2f}%")

# models --> logistic regression 

from sklearn.linear_model import LogisticRegression

model = LogisticRegression() 

Xt, Xv, yt, yv = train_test_split(X_train, y_train, test_size= 0.25)

y_pred = model.predict(Xv)
np.mean(y_pred == yv)

model.score(Xv, yv) # short cut 

# models have not yet tested 

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier() 

accs = cross_val_score(model, X_train, y_train, cv = 5, scoring = 'accuracy')
np.mean(accs)

# plotting

def plot_incorrect_predictions(df_predictions, x_axis_feature, y_axis_feature_):
    fig, axs = plt.subplots(2,2, figsize = (10,10))
    axs = axs.flatten() 
    sns.scatterplot(x=x_axis_feature, y=y_axis_feature_, hue ='prediction_label', data = df_predictions, ax = axs[0])
    sns.scatterplot(x=x_axis_feature, y=y_axis_feature_, hue ='target_name', data = df_predictions, ax = axs[1])
    sns.scatterplot(x=x_axis_feature, y=y_axis_feature_, hue ='correct_prediction', data = df_predictions, ax = axs[2])
    axs[3].set_visible

    plt.show()

# manual model tuning 

for manual_try_param in (1, 1.3, 2, 5, 10, 100):
    print(manual_try_param)
    model = LogisticRegression(C = manual_try_param)

    accs = cross_val_score(model, X_train, y_train, cv = 5, scoring = 'accuracy')
    print(f"Model Accuracy: {np.mean(accs)*100:.2f}")
