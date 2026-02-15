"""
Personalized Recommendations Display
Show recommendations across movies, anime, and manga with tabs.
"""

import streamlit as st
import requests

st.title("🎯 Your Personalized Recommendations")

# Check if user has rated anything
if not st.session_state.user_ratings:
    st.warning("⚠️ You haven't rated any movies yet!")
    st.info("👈 Go to **Rate Movies** page to get started")
    st.stop()

# User profile metrics
num_ratings = len(st.session_state.user_ratings)
strategy = "Cold Start (Genre-based)" if num_ratings < 10 else "Hybrid (CF + Content + Cross-Domain)"

st.subheader("📊 Your Profile")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Ratings Provided", num_ratings)

with col2:
    st.metric("Strategy", strategy.split('(')[0].strip())

with col3:
    algorithm = "Popularity" if num_ratings < 3 else "Genre Match" if num_ratings < 10 else "Hybrid ML"
    st.metric("Algorithm", algorithm)

st.divider()

# Media type tabs
tab_movie, tab_anime, tab_manga = st.tabs(["🎬 Movies", "📺 Anime", "📚 Manga"])

# Helper function to display recommendations
def display_recommendations(recommendations, response_time, media_type_name):
    """Display recommendations in a consistent format."""
    if not recommendations:
        st.warning("No recommendations generated")
        return

    st.success(f"✅ Generated {len(recommendations)} recommendations in {response_time:.0f}ms")

    for i, rec in enumerate(recommendations, 1):
        with st.container():
            # Header
            col_title, col_score = st.columns([4, 1])

            with col_title:
                st.markdown(f"### {i}. {rec['title']}")
                st.caption(f"{rec.get('year', 'N/A')} • {rec['media_type']}")

            with col_score:
                st.metric("Score", f"{rec['score']:.3f}")

            # Details
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write(f"**Genres**: {', '.join(rec['genres'])}")

            with col2:
                st.write(f"**Avg Rating**: ⭐ {rec['avg_rating']:.2f}")

            with col3:
                if 'rating_count' in rec:
                    st.caption(f"{rec.get('rating_count', 0):,} ratings")

            # Progress bar showing recommendation strength
            st.progress(min(rec['score'] / max(recommendations[0]['score'], 1.0), 1.0))

            st.divider()

# MOVIES TAB
with tab_movie:
    st.subheader("🎬 Movie Recommendations")
    st.write("Personalized movie suggestions based on your ratings")

    if st.button("🎯 Get Movie Recommendations", key="btn_movie", type="primary"):
        with st.spinner("🤖 Generating recommendations..."):
            try:
                response = requests.post(
                    f"{st.session_state.api_url}/recommend",
                    json={
                        "user_ratings": [
                            {"item_id": item_id, "rating": rating}
                            for item_id, rating in st.session_state.user_ratings.items()
                        ],
                        "target_media": "movie",
                        "top_n": 10
                    },
                    timeout=10
                )

                if response.ok:
                    data = response.json()
                    display_recommendations(
                        data['recommendations'],
                        data['response_time_ms'],
                        "Movies"
                    )
                else:
                    st.error(f"API error: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"Failed to get recommendations: {str(e)}")

# ANIME TAB
with tab_anime:
    st.subheader("📺 Anime Recommendations")

    st.info("💡 **Cross-Domain Transfer**: Based on your **movie** preferences, here are anime you might enjoy!")

    if st.button("🎯 Get Anime Recommendations", key="btn_anime", type="primary"):
        with st.spinner("🌐 Transferring preferences to anime..."):
            try:
                response = requests.post(
                    f"{st.session_state.api_url}/recommend",
                    json={
                        "user_ratings": [
                            {"item_id": item_id, "rating": rating}
                            for item_id, rating in st.session_state.user_ratings.items()
                        ],
                        "target_media": "anime",
                        "top_n": 10
                    },
                    timeout=10
                )

                if response.ok:
                    data = response.json()

                    st.success("✅ Cross-domain transfer successful!")
                    st.caption("Using genre/theme embeddings to bridge movies → anime")

                    display_recommendations(
                        data['recommendations'],
                        data['response_time_ms'],
                        "Anime"
                    )
                else:
                    st.error(f"API error: {response.status_code}")

            except Exception as e:
                st.error(f"Failed to get recommendations: {str(e)}")

# MANGA TAB
with tab_manga:
    st.subheader("📚 Manga Recommendations")

    st.info("💡 **Cross-Domain Transfer**: Based on your **movie** preferences, here are manga you might enjoy!")

    if st.button("🎯 Get Manga Recommendations", key="btn_manga", type="primary"):
        with st.spinner("🌐 Transferring preferences to manga..."):
            try:
                response = requests.post(
                    f"{st.session_state.api_url}/recommend",
                    json={
                        "user_ratings": [
                            {"item_id": item_id, "rating": rating}
                            for item_id, rating in st.session_state.user_ratings.items()
                        ],
                        "target_media": "manga",
                        "top_n": 10
                    },
                    timeout=10
                )

                if response.ok:
                    data = response.json()

                    st.success("✅ Cross-domain transfer successful!")
                    st.caption("Using genre/theme embeddings to bridge movies → manga")

                    display_recommendations(
                        data['recommendations'],
                        data['response_time_ms'],
                        "Manga"
                    )
                else:
                    st.error(f"API error: {response.status_code}")

            except Exception as e:
                st.error(f"Failed to get recommendations: {str(e)}")

# Footer info
st.divider()
st.caption(f"Ratings in session: {len(st.session_state.user_ratings)} | Strategy: {strategy}")
