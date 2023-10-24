#!/usr/bin/env python
# coding: utf-8

# # Streaming with Streamlit 
# 
# Streamlit lives up to its reputation of turning data scripts into shareable web apps in minutes, so let’s dive right into it
# 
# We will begin by initializing the binary file, then proceed to obtain user input from a list of movie names using a select box or in the form of text if the user prefers keyword-based recommendations. Based on the input, we will call the appropriate functions to generate the desired recommendations. To keep the app simple, we’ll set the top n parameter to a fixed value of 5.

# In[9]:


#!pip install streamlit


# In[20]:


import streamlit as st
import pandas as pd
import joblib
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# To generate picture thumbnails, we will leverage the TMDB API. To access this API, you will need to sign up on TMDB, which typically takes just a couple of minutes. Hers is the link go and signup to generate your free API Key : https://www.themoviedb.org/
# 
# 

# In[21]:


# set app config
st.set_page_config(page_title="Movie Recommendations For You", page_icon="🍿", layout="wide")    
st.markdown(f"""
            <style>
            .stApp {{background-image: url(""); 
                     background-attachment: fixed;
                     base: light;
                     background-size: cover}}
         </style>
         """, unsafe_allow_html=True)


# In[22]:


# Load models and MovieDB
df = joblib.load('models/movie_db.df')
tfidf_matrix = joblib.load('models/tfidf_mat.tf')
tfidf = joblib.load('models/vectorizer.tf')
cos_mat = joblib.load('models/cos_mat.mt')


# In[23]:


# define functions
def get_keywords_recommendations(keywords):
    
    keywords = keywords.split()
    keywords = " ".join(keywords) 
    
    # transform the string to vector representation
    key_tfidf = tfidf.transform([keywords]) 
    
    # compute cosine similarity    
    result = cosine_similarity(key_tfidf, tfidf_matrix)
    
    # sort top n similar movies   
    similar_key_movies = sorted(list(enumerate(result[0])), reverse=True, key=lambda x: x[1])
    
    # extract names from dataframe and return top 5 movie names
    recomm = []
    rating =[]
    for i in similar_key_movies[1:6]:
        recomm.append(df.iloc[i[0]].title)
        rating.append(df.iloc[i[0]].vote_average)
    
    return recomm, rating


# In[24]:


def get_recommendations(movie):
    
    # get index from dataframe
    index = df[df['title']== movie].index[0]   
    
    # sort top n similar movies     
    similar_movies = sorted(list(enumerate(cos_mat[index])), reverse=True, key=lambda x: x[1]) 
    
    # extract names from dataframe and return top 5 movie names
    recomm = []
    rating =[]
    for i in similar_movies[1:6]:
        recomm.append(df.iloc[i[0]].title)
        rating.append(df.iloc[i[0]].vote_average)
        
    return recomm, rating


# The process is fairly straightforward. By fetching the movie details using the movie ID from the DataFrame (which is why we included it in the DataFrame), we will receive a JSON response containing various key-value pairs. Our focus will be on the poster_path key, which provides the URL for the movie poster. We can utilize this URL to display the movie posters in our web application.

# In[25]:


def fetch_poster(movies):
    ids = []
    posters = []
    for i in movies:
        ids.append(df[df.title==i]['id'].values[0])
        
    for i in ids:    
        url = f"https://api.themoviedb.org/3/movie/{i}?api_key=13a6cbfe9aadf79953ea90c6452194e3"
        data = requests.get(url)
        data = data.json()
        poster_path = data['poster_path']
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        posters.append(full_path)
    return posters

#19995 /gC3tW9a45RGOzzSh6wv91pFnmFr.jpg


# In[28]:


# App Layout
st.image("images/applogo1.png")
st.title("🍿 Let me help you find a movie of your choice! 🍿 ")
posters = 0
movies = 0
ratings = 0


# In[29]:


with st.sidebar:
    st.image("images/app1.png", use_column_width=True)
    st.header("Get Recommendations by 👇")
    search_type = st.radio("", ('Movies you like', 'Keywords'))
    st.header("Code 📦")
    st.markdown("[GitHub Repository](https://github.com/avijay24/Movie-Recommender-Web-Application-using-Streamlit)")    
    st.subheader('Enjoy the movie! 🎬 🍿')
    st.markdown("Source: https://www.themoviedb.org/")


# In[30]:


# call functions based on selectbox
if search_type == 'Movies you like': 
    st.subheader("Select a Movie you like 🎬")   
    movie_name = st.selectbox('', df.title)
    if st.button('Recommend 🚀'):
        with st.spinner('Wait for it...'):
            movies, ratings = get_recommendations(movie_name)
            posters = fetch_poster(movies)        
else:
    st.subheader('Enter Cast / Crew / Tags / Genre  🌟')
    keyword = st.text_input('', 'Christopher Nolan')
    if st.button('Recommend 🚀'):
        with st.spinner('Wait for it...'):
            movies, ratings = get_keywords_recommendations(keyword)
            posters = fetch_poster(movies)


# In[19]:


# display posters       
if posters:
    col1, col2, col3, col4, col5 = st.columns(5, gap='medium')
    with col1:
        st.text(movies[0])
        st.image(posters[0])
        st.text('Rating: '+ str(ratings[0]))
    with col2:
        st.text(movies[1])
        st.image(posters[1])
        st.text('Rating: '+ str(ratings[1]))
    with col3:
        st.text(movies[2])
        st.image(posters[2])
        st.text('Rating: '+ str(ratings[2]))
    with col4:
        st.text(movies[3])
        st.image(posters[3])
        st.text('Rating: '+ str(ratings[3]))
    with col5:
        st.text(movies[4])
        st.image(posters[4])
        st.text('Rating: '+ str(ratings[4]))


# Create a project folder in your local pc. Save all the images and models in that directory:
# C:..\MovieDB\images
# C:..\MovieDB\models
# C:..\MovieDB\app.py
# C:..\MovieDB\images\app1.png
# C:..\MovieDB\images\applogo1.png
# C:..\MovieDB\models\cos_mat.mt
# C:..\MovieDB\models\movie_db.df
# C:..\MovieDB\models\tfidf_mat.tf
# C:..\MovieDB\models\vectorizer.tf
# 
# 
# and this particule file as app.py
# 
# Open Anaconda > Environments > Click on triangle > 'Open Terminal'
# 
# Navigate to the project folder
# to run the app:
# 
# streamlit run app.py

# In[ ]:




