"""
Media Recommendation System - Streamlit UI
Main application page with welcome message and navigation.
"""

import streamlit as st
import requests

# Page configuration
st.set_page_config(
    page_title="Media Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {}

if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"

# Main page content
st.title("🎬 Cross-Media Recommendation System")

st.markdown("""
Welcome to the **Cross-Media Recommendation Engine**! This system provides personalized
recommendations across **movies, anime, and manga** using advanced machine learning algorithms.

### 🚀 How It Works

1. **Rate Movies** → Search and rate movies you've seen
2. **Get Recommendations** → Receive personalized suggestions across all media types
3. **Analytics** → Visualize your preferences and insights

### ✨ Key Features

- 🎯 **Hybrid Algorithm**: Combines collaborative filtering, content-based, and cross-domain signals
- 🆕 **Cold Start Support**: Works with just 3-5 ratings (adapts as you rate more)
- 🌐 **Cross-Media Transfer**: Your movie preferences → Anime & Manga recommendations
- 📊 **56,439 Items**: 32K movies, 14K anime, 10K manga
- ⚡ **Fast**: < 200ms response time

### 🧠 Technology Stack

- **Algorithms**: Item-based CF, TF-IDF embeddings, genre bridges, weighted hybrid
- **Backend**: FastAPI REST API
- **Frontend**: Streamlit
- **Data**: 78M+ ratings from MovieLens, MyAnimeList, and more
""")

st.divider()

# Getting started guide
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1️⃣ Rate Movies")
    st.write("Search for movies you've seen and rate them on a 0.5-5.0 scale.")
    st.caption("💡 Tip: Rate at least 3 movies to get started")

with col2:
    st.subheader("2️⃣ Get Recommendations")
    st.write("Generate personalized recommendations for movies, anime, or manga.")
    st.caption("💡 Tip: Try cross-domain transfer (movies → anime)")

with col3:
    st.subheader("3️⃣ View Analytics")
    st.write("Visualize your genre preferences and rating patterns.")
    st.caption("💡 Tip: Rate 10+ movies for hybrid algorithm")

st.divider()

# Sidebar with stats
with st.sidebar:
    st.header("📊 Your Profile")

    # Rating count
    num_ratings = len(st.session_state.user_ratings)
    st.metric("Movies Rated", num_ratings)

    # Strategy indicator
    if num_ratings == 0:
        st.info("Not started")
    elif num_ratings < 10:
        st.warning(f"Cold Start ({10 - num_ratings} more for hybrid)")
    else:
        st.success("Hybrid Strategy Active")

    # Recent ratings
    if st.session_state.user_ratings:
        st.write("**Recent Ratings:**")
        for item_id, rating in list(st.session_state.user_ratings.items())[-5:]:
            st.caption(f"⭐ {rating} - {item_id}")

    st.divider()

    # API status
    st.header("🔌 API Status")

    try:
        health = requests.get(f"{st.session_state.api_url}/health", timeout=2).json()

        if health['status'] == 'healthy':
            st.success("✅ Connected")
            st.caption(f"Movies: {health['items_count']['movie']:,}")
            st.caption(f"Anime: {health['items_count']['anime']:,}")
            st.caption(f"Manga: {health['items_count']['manga']:,}")
        else:
            st.warning(f"⚠️ API status: {health['status']}")

    except Exception as e:
        st.error("❌ API not reachable")
        st.caption("Make sure API is running: `python scripts/run_api.py`")

# Footer
st.divider()
st.caption("Built with ❤️ using FastAPI + Streamlit | Data: MovieLens 32M, MyAnimeList, Steam")
