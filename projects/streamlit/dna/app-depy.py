import pandas as pd
import streamlit as st 
import altair as alt 

# credits to https://github.com/dataprofessor/streamlit_freecodecamp/tree/main/app_2_simple_bioinformatics_dna

st.write("""
         
# DNA Nucleotide Count Web App.
         
This app counts the nucleotide composition of query DNA... Scroll below to find out!
         
""")

st.header('Enter DNA Sequence')

sequence = st.text_area("Sequence Input here:", height = 100)


st.header('INPUT - DNA Query')
sequence

st.header('OUTPUT - Nucleotide Counts')

st.subheader('Congratulations...')

def counting(seq):
    d = dict([
        ('A', seq.count('A')),
        ('T', seq.count('T')),
        ('G', seq.count('G')),
        ('C', seq.count('C')), 

    ])
    return d 

X = counting(sequence)

X_label = list(X) 
x_values = list(X.values())

X

st.write("There are " + str(X['A']) + " Adenine")
st.write("There are " + str(X['T']) + " Thymine")
st.write("There are " + str(X['G']) + " Guanine")
st.write("There are " + str(X['C']) + " Cytosine")