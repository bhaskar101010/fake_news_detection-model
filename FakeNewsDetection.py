import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import nltk
import re 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
import pylab as pl
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import streamlit as st

st.title('Fake News Detector')

nav = st.sidebar.radio("Navigation",["HOME","Prediction"])

model = joblib.load('pipe.joblib')

if nav == 'HOME':
    st.header('Know the Truth')
    st.image('image.jpg', use_column_width=True)

else :
    st.subheader('Find The Correct News')
    news = st.text_area(label="News",placeholder="Enter the News here") 
    if st.button("Predict"):
        result = model.predict([news])
        if news=="":
            st.text('Enter the News first')
        elif result == [0] :
            st.text("Fake News")
        else :
            st.text('Real News')