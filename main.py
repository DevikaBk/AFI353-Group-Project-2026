import streamlit as st
import pickle
import pandas as pd
import requests
import ast
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from style import apply_custom_styling, set_page_config, COLORS

# Apply styling and page config
set_page_config()
apply_custom_styling()

API_KEY = "8265bd1679663a7ea12ac168da84d2e8"

# Load the sentence transformer model for description-based recommendations
@st.cache_resource
def load_semantic_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load movie embeddings for semantic search
@st.cache_data
def load_movie_embeddings():
    return np.load('movie_embeddings.npy')

# Load the full movies dataset for semantic search
@st.cache_data
def load_full_movies():
    return pd.read_csv('data/movies_5000.csv')

# Initialize semantic search components
try:
    semantic_model = load_semantic_model()
    movie_embeddings = load_movie_embeddings()
    full_movies_df = load_full_movies()
    semantic_search_available = True
except Exception as e:
    semantic_search_available = False
    st.warning(f"Semantic search model not loaded: {str(e)}")

def fetch_poster(movie_id):
    response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US")
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']

def fetch_movie_details(movie_id):
    """Fetch overview from API"""
    response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US")
    data = response.json()
    return data.get('overview', 'No description available')

def clean_input(text):
    noise_words = ['movie', 'movies', 'film', 'films', 'cinema']
    text = text.lower()
    words = text.split()
    cleaned_words = [word for word in words if word not in noise_words]
    return " ".join(cleaned_words)


def recommend_by_description(description, top_n=5):
    """
    Get movie recommendations based on text description.
    First tries exact title match (after cleaning), then falls back to semantic search.
    Returns: (movie_titles, posters, movie_ids)
    """
    try:
        # Clean the input description
        cleaned_input = clean_input(description)

        # 1️⃣ Exact title match (case‑insensitive substring)
        exact_matches = movies[movies['title'].str.contains(cleaned_input, case=False, na=False)]
        if len(exact_matches) >= top_n:
            # Return the first top_n exact matches
            recommended_movies = []
            recommended_movies_posters = []
            recommended_movies_ids = []
            for _, row in exact_matches.head(top_n).iterrows():
                title = row['title']
                movie_id = int(row['movie_id'])
                recommended_movies.append(title)
                recommended_movies_ids.append(movie_id)
                poster = fetch_poster(movie_id)
                recommended_movies_posters.append(poster)
            return recommended_movies, recommended_movies_posters, recommended_movies_ids

        # 2️⃣ Fallback to semantic search if exact matches are insufficient
        if not semantic_search_available:
            st.error("Semantic search model is not available")
            return [], [], []

        # Encode the cleaned description
        user_embedding = semantic_model.encode([cleaned_input])
        similarity_scores = cosine_similarity(user_embedding, movie_embeddings)[0]
        top_indices = np.argsort(similarity_scores)[::-1][:top_n]

        recommended_movies = []
        recommended_movies_posters = []
        recommended_movies_ids = []

        for idx in top_indices:
            try:
                movie_title = full_movies_df.iloc[idx]['title']
                # Try to get movie_id from the main movies dataframe
                movie_data = movies[movies['title'] == movie_title]
                if len(movie_data) > 0:
                    movie_id = int(movie_data.iloc[0]['movie_id'])
                else:
                    original_movie = original_df[original_df['title'] == movie_title]
                    if len(original_movie) > 0:
                        movie_id = int(original_movie.iloc[0]['movie_id'])
                    else:
                        continue

                recommended_movies.append(movie_title)
                recommended_movies_ids.append(movie_id)
                poster = fetch_poster(movie_id)
                recommended_movies_posters.append(poster)

            except Exception:
                continue

        return recommended_movies, recommended_movies_posters, recommended_movies_ids

    except Exception as e:
        st.error(f"Error getting hybrid recommendations: {str(e)}")
        return [], [], []

def recommend(movie):
    try:
        movie_index = movies[movies['title'] == movie].index

        if len(movie_index) == 0:
            st.error(f"Movie '{movie}' not found in database")
            return [], [], []

        movie_index = movie_index[0]
        distances = similarity[movie_index]

        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        recommended_movies = []
        recommended_movies_posters = []
        recommended_movies_ids = []

        for i in movie_list:
            try:
                movie_id = int(movies.iloc[i[0]]['movie_id'])
                recommended_movies.append(movies.iloc[i[0]]['title'])
                recommended_movies_ids.append(movie_id)

                poster = fetch_poster(movie_id)
                recommended_movies_posters.append(poster)

            except Exception:
                continue

        return recommended_movies, recommended_movies_posters, recommended_movies_ids

    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return [], [], []

# Load original CSV for genres, cast, crew
original_movies = pd.read_csv('data/movies.csv')
original_credits = pd.read_csv('data/credits.csv')
original_df = original_movies.merge(original_credits, on='title')

# Load model data
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('similarity.pkl', 'rb'))

# Session state
if 'show_details' not in st.session_state:
    st.session_state.show_details = None

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

if 'recommendation_source' not in st.session_state:
    st.session_state.recommendation_source = None

# Title
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@600&display=swap" rel="stylesheet">
<h1 style='font-family: "Barlow Semi Condensed", sans-serif; font-size: 80px; color: #ffbe24; text-align: center;'>
MOVIEMEND
</h1>
""", unsafe_allow_html=True)

# MAIN PAGE
if st.session_state.show_details is None:

    st.markdown("""
    <style>
    .custom-selectbox-label {
        font-size: 24px;
        font-weight: bold;
    }
    .description-box {
        margin-top: 20px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

       #size of tabs 
    st.markdown("""
    <style>
        button[data-baseweb="tab"] p {
            font-size: 20px !important;
            font-weight: semi bold !important;
           }
        </style>
    """, unsafe_allow_html=True)

    # Create two tabs for different recommendation methods
    tab1, tab2 = st.tabs(["Search by Movie Title", "  Search by Description"])
    
    # TAB 1: Movie Title Search
    with tab1:
        st.markdown('<div class="custom-selectbox-label">Watch your next movie</div>', unsafe_allow_html=True)
        st.markdown(
          '<p style="font-size: 19px; font-weight:450; margin-bottom: 0;">Select a movie:</p>',
           unsafe_allow_html=True
        )

        selected_movie_name = st.selectbox(
          label="",
          options=movies['title'].values,
          key='movie_select',
          label_visibility="collapsed"  # Removes default label space
        )
        
        col_empty1, col_btn, col_empty2 = st.columns([1, 2, 1])
        
        with col_btn:
            if st.button('Recommend', key='recommend_main', use_container_width=True):
                names, posters, movie_ids = recommend(selected_movie_name)
                
                if names and len(names) > 0:
                    st.session_state.recommendations = list(zip(names, posters, movie_ids))
                    st.session_state.recommendation_source = 'title'
                    st.rerun()
                else:
                    st.error("Could not get recommendations")
    
    # TAB 2: Description Search
    with tab2:
        st.markdown("""
            <div style='font-size: 22px; font-weight: bold; margin-bottom: 5px;'>
              Describe the movie you want to watch
             </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="font-size: 18px; font-weight: 450;">Enter movie description (e.g., \'A thrilling action movie about spies\')</div>', 
            unsafe_allow_html=True)

        description_input = st.text_area(
          label="",
          height=60,
          key='description_input',
          placeholder="study motivation movie...",
          label_visibility="collapsed"
        )
        
        col_desc_empty1, col_desc_btn, col_desc_empty2 = st.columns([1, 2, 1])
        
        with col_desc_btn:
            if st.button('Find Movies by Description', key='recommend_description', use_container_width=True):
                if description_input and description_input.strip():
                    with st.spinner('Finding movies that match your description...'):
                        names, posters, movie_ids = recommend_by_description(description_input, top_n=5)
                        
                        if names and len(names) > 0:
                            st.session_state.recommendations = list(zip(names, posters, movie_ids))
                            st.session_state.recommendation_source = 'description'
                            st.rerun()
                        else:
                            st.error("Could not find movies matching your description. Please try a different description.")
                else:
                    st.warning("Please enter a movie description first.")
        
        if semantic_search_available:
            st.info("💡 Tip: Be descriptive! Include genres, mood, plot elements, or themes you're interested in.")
        else:
            st.error("⚠️ Description-based search is currently unavailable. Please use the Movie Title search instead.")

    # Display recommendations if they exist
    if st.session_state.recommendations:
        source_label = "Recommended Movies Based on Your Description" if st.session_state.recommendation_source == 'description' else "Recommended Movies"
        st.subheader(source_label)
        
        # Add a note for description-based recommendations
        if st.session_state.recommendation_source == 'description':
            st.caption("✨ These movies are recommended based on the description you provided")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        columns = [col1, col2, col3, col4, col5]
        
        for idx, (col, (name, poster, movie_id)) in enumerate(zip(columns, st.session_state.recommendations)):
            with col:
                st.image(poster, width=200)
                
                if st.button(name, key=f"movie_{idx}", use_container_width=True):
                    st.session_state.show_details = idx
                    st.rerun()

# DETAILS PAGE
else:
    col_back, col_title = st.columns([1, 4], gap="small")
    
    with col_back:
        if st.button("← Back"):
            st.session_state.show_details = None
            st.rerun()
    
    movie_name, poster, movie_id = st.session_state.recommendations[st.session_state.show_details]
    
    overview = fetch_movie_details(movie_id)
    
    try:
        movie_data = original_df[original_df['title'] == movie_name].iloc[0]
        
        genres = ast.literal_eval(movie_data.get('genres', '[]'))
        cast = ast.literal_eval(movie_data.get('cast', '[]'))
        crew = ast.literal_eval(movie_data.get('crew', '[]'))
        
        genres_list = [genre['name'] for genre in genres] if genres else []
        cast_list = [actor['name'] for actor in cast[:3]] if cast else []
        director_list = [person['name'] for person in crew if person['job'] == 'Director'] if crew else []
        
    except:
        genres_list = []
        cast_list = []
        director_list = []
    
    col_poster, col_details = st.columns([1, 2], gap="small")
    
    with col_poster:
        st.image(poster, width=280)
    
    with col_details:
        st.markdown('<div class="detail-section">', unsafe_allow_html=True)
        
        st.markdown(f"**<span style='font-size: 2.5em'>{movie_name}</span>**", unsafe_allow_html=True)
        
        st.markdown("**Overview:**")
        st.markdown(overview)
        
        if genres_list:
            st.markdown("**Genres:**")
            st.markdown(", ".join(genres_list))
        
        if cast_list:
            st.markdown("**Cast:**")
            st.markdown(", ".join(cast_list))
        
        if director_list:
            st.markdown("**Director:**")
            st.markdown(", ".join(director_list))
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    col_empty1, col_btn, col_empty2 = st.columns([1, 2, 1])
    
    with col_btn:
        if st.button("Get Similar Movie Recommendations", key='recommend_details', use_container_width=True):
            names, posters, movie_ids = recommend(movie_name)
            
            if names and len(names) > 0:
                st.session_state.recommendations = list(zip(names, posters, movie_ids))
                st.session_state.show_details = None
                st.rerun()