#Import libraries
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

#Title
st.title('Movie Recommender')

#Load CSV
movies_data = pd.read_csv('movies.csv')

#Select relevant features
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

#Fill null data points with empty
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

#Combine features
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

#Vectorize features
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)


similarity = cosine_similarity(feature_vectors)

#Get user input
movie_name = st.text_input('Enter movie', 'Finding Nemo')

#Create list with movie titles
movie_names = movies_data['title'].tolist()

#Find list of movies similar to user input
find_close_matches = difflib.get_close_matches(movie_name, movie_names)

#Continue if there is a match
if find_close_matches:
    #Take closest match
    close_match = find_close_matches[0]

    #Find index of closest match
    index_of_movie = movies_data[movies_data.title == close_match]['index'].values[0]

    #Get the list of similarities to all movies in the list
    similarity_score = list(enumerate(similarity[index_of_movie]))

    #Put movies in descending roder
    sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)

    #Direct user to which movie was the closest
    st.write('If you liked', close_match, ', you would like:')

    #Generate similar movies list
    i = 1
    for movie in sorted_similar_movies[1:]:
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        if (i<6):
            st.write(i, '.', title_from_index)
            i+=1

#Stop if there is no match to user input
else:
    st.write("Couldn't find the movie you specified.")