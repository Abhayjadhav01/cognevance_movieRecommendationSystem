import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# --------------------------------------------------
# Load Data
# --------------------------------------------------

@st.cache_data
def load_data():

    movies = pd.read_pickle(
        MODEL_DIR / "movies.pkl"
    )

    ratings = pd.read_pickle(
        MODEL_DIR / "ratings.pkl"
    )

    similarity = joblib.load(
        MODEL_DIR / "movie_similarity.pkl"
    )

    return movies, ratings, similarity


movies, ratings, movie_similarity_df = load_data()


# --------------------------------------------------
# Recommendation Function
# --------------------------------------------------

def recommend_movies(movie_title, n=10):

    movie_matches = movies[
        movies["title"].str.contains(
            movie_title,
            case=False,
            na=False
        )
    ]

    if movie_matches.empty:
        return pd.DataFrame()

    movie_id = movie_matches.iloc[0]["movieId"]

    if movie_id not in movie_similarity_df.index:
        return pd.DataFrame()

    similarity_scores = movie_similarity_df[movie_id]

    similar_movie_ids = (
        similarity_scores
        .sort_values(ascending=False)
        .iloc[1:]
        .index
    )

    recommendations = movies[
        movies["movieId"].isin(similar_movie_ids)
    ][["movieId", "title"]].copy()

    recommendations["similarity"] = (
        recommendations["movieId"]
        .map(similarity_scores)
    )

    recommendations = recommendations.sort_values(
        "similarity",
        ascending=False
    )

    return recommendations.head(n)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🎬 Movie Recommendation System")

st.write(
    "Discover movies similar to the ones you already enjoy "
    "using collaborative filtering."
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Recommendation Settings")

number_of_recommendations = st.sidebar.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10
)


# --------------------------------------------------
# Movie Selection
# --------------------------------------------------

st.subheader("Find Similar Movies")

movie_search = st.text_input(
    "Enter a movie title",
    placeholder="Example: Toy Story"
)


if st.button("Recommend Movies"):

    if movie_search.strip() == "":
        st.warning("Please enter a movie title.")

    else:

        recommendations = recommend_movies(
            movie_search,
            number_of_recommendations
        )

        if recommendations.empty:

            st.error(
                "Movie not found or insufficient rating data."
            )

        else:

            st.success(
                f"Recommendations based on '{movie_search}'"
            )

            display_data = recommendations[
                ["title", "similarity"]
            ].copy()

            display_data["similarity"] = (
                display_data["similarity"]
                .round(3)
            )

            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True
            )


# --------------------------------------------------
# Dataset Statistics
# --------------------------------------------------

st.divider()

st.subheader("📊 Dataset Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Users",
        ratings["userId"].nunique()
    )

with col2:
    st.metric(
        "Movies",
        movies["movieId"].nunique()
    )

with col3:
    st.metric(
        "Ratings",
        len(ratings)
    )