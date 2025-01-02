import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


st.write("""

# Iris Flower Prediction Application 
         
The following below predicts the Iris Flower Type. All data points are in centimeters. 
         
         """)

st.sidebar.header("What item(s) would you like to select?")

def input_structure():
    sep_length = st.sidebar.slider('Slide Your Sepal Length', 0, 10, 5)
    sep_width = st.sidebar.slider('Slide Your Sepal Width', 0, 10, 5)
    pet_length = st.sidebar.slider('Slide Your Petal Length', 0, 10, 5)
    pet_width = st.sidebar.slider('Slide Your Petal Width', 0, 10, 5)

    data = {
        'Sepal Length': sep_length,
        'Sepal Width': sep_width,
        'Petal Length': pet_length,
        'Petal Width': pet_width
    }
    
    data_features = pd.DataFrame(data, index = [0])
    return data_features 

df = input_structure() 

st.subheader('Your Inputs')
st.write(df)

iris_dataset = datasets.load_iris() 
X = iris_dataset.data 
Y = iris_dataset.target

rfc = RandomForestClassifier() 
rfc.fit(X,Y) 

predictions = rfc.predict(df) 
predictions_probabilities = rfc.predict_proba(df)

st.subheader('Types of Flowers')
st.write(iris_dataset.target_names)

st.subheader('Prediction')
st.write(predictions)

st.subheader('Prediction Probabilities')
st.write(predictions_probabilities)