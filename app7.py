import streamlit as st
import pickle
import requests
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os

# # Content-based movie recommendation system

st.set_page_config(layout='wide')

@st.cache_resource
def load_data():
    if os.path.exists('./content_movie_model'):
        model = SentenceTransformer('./content_movie_model')
        print('model load successfully')
    else:
        print('folder not found')

    movie_data = pickle.load(open('movie.pkl', 'rb'))
    movie_vector_database = pickle.load(open('movie_vector_database.pkl', 'rb'))
    trending_movie = pickle.load(open('trending_movie.pkl', 'rb'))
    high_rated_movie = pickle.load(open('high_rated_movie.pkl', 'rb'))
    movie_list = movie_data['original_title']

    return model, movie_data, trending_movie, high_rated_movie, movie_vector_database, movie_list


model, movie_data, trending_movie, high_rated_movie, movie_vector_database,  movie_list = load_data()


# movie recommend function
def get_recommend_movies(movie_name, num_of_recommend):
    # Check if movie exists
    if movie_name not in movie_data['original_title'].values:
        return f"Movie '{movie_name}' not found in database!"
  
    movie_desc = movie_data[movie_data['original_title'] == movie_name]['movie_text'].values[0]
    movie_emb = model.encode([movie_desc]).astype("float32")
    D, I = movie_vector_database.search(movie_emb, num_of_recommend + 1)  # +1 to exclude itself
    
    recommended_titles = []
    poster_image = []
    recommend_rate = []
    recommend_date = []
    for idx in I[0]:
        if movie_data['original_title'][idx] != movie_name:
            recommended_titles.append(movie_data['original_title'][idx])
            poster_image.append(movie_data['poster_url'][idx])
            recommend_rate.append(movie_data['rating'][idx])
            recommend_date.append(movie_data['date'][idx])
        if len(recommended_titles) >= num_of_recommend:
            break

    return recommended_titles, poster_image, recommend_rate, recommend_date
    
# movie recommend function by search
def movie_recommend_by_search(search_text, num_of_recommend):
    # Encode user input
    movie_emb = model.encode([search_text]).astype("float32")

    # Search top 3 similar movies
    D, I = movie_vector_database.search(movie_emb, num_of_recommend+1)

    recommended_titles = []
    poster_image = []
    recommend_rate = []
    recommend_date = []
    for idx in I[0]:
        recommended_titles.append(movie_data['original_title'][idx])
        poster_image.append(movie_data['poster_url'][idx])
        recommend_rate.append(movie_data['rating'][idx])
        recommend_date.append(movie_data['date'][idx])
        if len(recommended_titles) >= num_of_recommend:
            break
    
    return recommended_titles, poster_image, recommend_rate, recommend_date


# Trending movie's
def get_movie_trending():
    trend_movie = []
    trend_picture = []
    trend_rate = []
    trend_date = []

    for idx in trending_movie['original_title'].index:
        trend_picture.append(trending_movie['poster_url'][idx])
        trend_movie.append(trending_movie['original_title'][idx])
        trend_rate.append(trending_movie['vote_average'][idx])
        trend_date.append(trending_movie['date'][idx])

    return trend_movie, trend_picture, trend_rate, trend_date


# Top rated movie's
def get_top_rated_movie():
    rated_movie = []
    rated_picture = []
    rated_rate = []
    rated_date = []
    for idx in high_rated_movie['original_title'].index:
        # movie_index = movie_data[movie_data['original_title'] == i].index[0] 
        # fetch poster from api
        rated_picture.append(high_rated_movie['poster_url'][idx])
        rated_movie.append(high_rated_movie['original_title'][idx])
        rated_rate.append(high_rated_movie['vote_average'][idx])
        rated_date.append(high_rated_movie['date'][idx])
    
    return rated_movie, rated_picture, rated_rate, rated_date



# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []



# Header
st.title("Content Filtering Recommendation System ")

# Sidebar
st.sidebar.header("Quick Actions")
if st.sidebar.button("New Recommendation"):
    st.rerun()

st.sidebar.header("Filters")
no_of_recommend = st.sidebar.slider("Number of Recommendations", 1, 20, 10)


# Main Content with Tabs
tab1, tab2 = st.tabs(["Recommend", "History"])


st.markdown("""
        <style>
            .movie-title{
            font-size:13px;
            line-height:1.3;
            word-wrap:break-word;
            white-space:normal;
            text-align:left;
            height:20px;
            }
            .rate-date{
            margin-bottom:0px;
            font-size:13px;
            color:#fafafa99;
            }
        </style>
            """, unsafe_allow_html=True)


# Recommend tab
with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        selected_movie = st.selectbox('', placeholder='Select/search movies.. ', options=movie_list, index=None)
        recommend_button = st.button("Get Recommendations")
    with col2:
        search = st.text_input('', placeholder='Search movie by topics/genres.. ')
        search_button = st.button('search')
    
    # Recommend movie
    if recommend_button:
        recommend_movie, poster_image, recommend_rate, recommend_date = get_recommend_movies(selected_movie, no_of_recommend)
        st.subheader('Recommended Movies')
        with st.container(border=True, height=280):
            cols1 = st.columns(5, gap='small') 
            for i in range(len(recommend_movie)):
                with cols1[i % 5]:
                    st.image(poster_image[i])
                    st.markdown(f'<p class="movie-title"> {recommend_movie[i]} </p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="rate-date">⭐{recommend_rate[i]} </p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="rate-date"> {recommend_date[i]} </p>', unsafe_allow_html=True)
                
                
            # Add to history
            st.session_state.history.append({
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Movie": selected_movie,
                'Search' : '',
                "Recommendations" : [movie for movie in recommend_movie]
            })
       
       
        #Trending session    
        trend_movie, trend_pciture, trend_rate, trend_date = get_movie_trending()
        st.subheader('Trending Now')
        with st.container(border=True, height=280):
            cols2 = st.columns(5, gap='small')
            for i in range(len(trend_movie)):
                with cols2[i % 5]:
                    st.image(trend_pciture[i])
                    st.markdown(f'<p class="movie-title"> {trend_movie[i]} </p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="rate-date">⭐{trend_rate[i]} </p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="rate-date"> {trend_date[i]} </p>', unsafe_allow_html=True)
            

        # # Top rated session
        top_rated_movie, top_rated_picture, top_rated_rate, top_rated_date = get_top_rated_movie()
        st.subheader('Top 25 High Rated Movies')
        with st.container(border=True, height=400):
            cols3 = st.columns(5, gap='small')
            for i in range(len(top_rated_movie)):
                with cols3[i % 5]:
                    st.image(top_rated_picture[i])
                    st.markdown(f'<p class="movie-title"> {top_rated_movie[i]} </p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="rate-date">⭐{top_rated_rate[i]} </p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="rate-date">{top_rated_date[i]} </p>', unsafe_allow_html=True)

    # searching movie 
    if search_button:
        recommend_movie_by_search, poster_image_by_search, recommend_rate_by_search, recommend_date_by_search = movie_recommend_by_search(search, no_of_recommend)
        # Recommend session
        st.subheader('Recommended Movies')
        with st.container(border=True, height=280):         
            cols1 = st.columns(5, gap='small') 
            for i in range(len(recommend_movie_by_search)):
                with cols1[i % 5]:
                    st.image(poster_image_by_search[i])
                    st.markdown(f'<p class="movie-title"> {recommend_movie_by_search[i]} </p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="rate-date">⭐{recommend_rate_by_search[i]} </p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="rate-date"> {recommend_date_by_search[i]} </p>', unsafe_allow_html=True)
                
                
            # Add to history
            st.session_state.history.append({
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Movie": '',
                'Search' : search,
                "Recommendations": [movie for movie in recommend_movie_by_search]
            })

       
        #Trending session    
        trend_movie, trend_pciture, trend_rate, trend_date = get_movie_trending()
        st.subheader('Trending Now')
        with st.container(border=True, height=280):
            cols2 = st.columns(5, gap='small')
            for i in range(len(trend_movie)):
                with cols2[i % 5]:
                    st.image(trend_pciture[i])
                    st.markdown(f'<p class="movie-title"> {trend_movie[i]} </p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="rate-date">⭐{trend_rate[i]} </p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="rate-date"> {trend_date[i]} </p>', unsafe_allow_html=True)
            
    

        # # Top rated session
        top_rated_movie, top_rated_picture, top_rated_rate, top_rated_date = get_top_rated_movie()
        st.subheader('Top 25 High Rated Movies')
        with st.container(border=True, height=400):
            cols3 = st.columns(5, gap='small')
            for i in range(len(top_rated_movie)):
                with cols3[i % 5]:
                    st.image(top_rated_picture[i])
                    st.markdown(f'<p class="movie-title"> {top_rated_movie[i]} </p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="rate-date">⭐{top_rated_rate[i]} </p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="rate-date">{top_rated_date[i]} </p>', unsafe_allow_html=True)
          
            
# Histroy tab           
with tab2:
    st.header("Recommendation History")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)
        # Clear history button 
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.write("No history yet.")

