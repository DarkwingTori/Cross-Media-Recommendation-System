"""
Analytics Dashboard
Visualize user preferences, rating patterns, and system insights.
"""

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Analytics Dashboard")

# Check if user has ratings
if not st.session_state.user_ratings:
    st.info("📝 Rate some movies to see analytics and insights about your preferences!")
    st.stop()

num_ratings = len(st.session_state.user_ratings)

st.write(f"Analyzing your **{num_ratings} ratings**...")

st.divider()

# Load rated movie details
@st.cache_data
def load_rated_items(ratings_dict, api_url):
    """Load details for all rated items."""
    items = []

    for item_id, rating in ratings_dict.items():
        try:
            resp = requests.get(f"{api_url}/items/{item_id}", timeout=2)
            if resp.ok:
                item = resp.json()
                item['user_rating'] = rating
                items.append(item)
        except:
            pass

    return items


rated_items = load_rated_items(dict(st.session_state.user_ratings), st.session_state.api_url)

if not rated_items:
    st.error("Could not load item details. Make sure the API is running.")
    st.stop()

# Convert to DataFrame for analysis
df_items = pd.DataFrame(rated_items)

# 1. GENRE PREFERENCES
st.subheader("🎭 Your Genre Preferences")

# Extract all genres with ratings
genre_ratings = []
for item in rated_items:
    for genre in item['genres']:
        genre_ratings.append({
            'genre': genre,
            'rating': item['user_rating']
        })

df_genres = pd.DataFrame(genre_ratings)

if not df_genres.empty:
    # Calculate average rating per genre
    genre_avg = df_genres.groupby('genre')['rating'].agg(['mean', 'count']).sort_values('mean', ascending=False)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    genre_avg['mean'].plot(kind='barh', ax=ax, color='skyblue', edgecolor='navy')
    ax.set_xlabel("Average Rating", fontsize=12)
    ax.set_ylabel("Genre", fontsize=12)
    ax.set_title("Your Top Genres", fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    st.pyplot(fig)

    # Top genres table
    col1, col2 = st.columns(2)

    with col1:
        st.metric("🏆 Favorite Genre", genre_avg.index[0])
        st.caption(f"Average rating: {genre_avg['mean'].iloc[0]:.2f}")

    with col2:
        st.metric("📈 Most Rated Genre", genre_avg['count'].idxmax())
        st.caption(f"Count: {genre_avg['count'].max()} movies")

    # Genre distribution table
    with st.expander("📋 All Genres"):
        genre_table = genre_avg.reset_index()
        genre_table.columns = ['Genre', 'Avg Rating', 'Count']
        genre_table['Avg Rating'] = genre_table['Avg Rating'].round(2)
        st.dataframe(genre_table, use_container_width=True, hide_index=True)

else:
    st.info("Rate more movies with different genres to see preferences!")

st.divider()

# 2. RATING DISTRIBUTION
st.subheader("⭐ Your Rating Distribution")

ratings_list = [item['user_rating'] for item in rated_items]

col1, col2 = st.columns(2)

with col1:
    # Histogram
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(ratings_list, bins=10, range=(0.5, 5.0), color='coral', edgecolor='darkred', alpha=0.7)
    ax2.set_xlabel("Rating", fontsize=12)
    ax2.set_ylabel("Number of Movies", fontsize=12)
    ax2.set_title("Rating Distribution", fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    st.pyplot(fig2)

with col2:
    # Statistics
    st.metric("Mean Rating", f"{pd.Series(ratings_list).mean():.2f}")
    st.metric("Median Rating", f"{pd.Series(ratings_list).median():.2f}")
    st.metric("Rating Std Dev", f"{pd.Series(ratings_list).std():.2f}")

    # Rating breakdown
    st.write("**Rating Breakdown:**")
    rating_counts = pd.Series(ratings_list).value_counts().sort_index(ascending=False)
    for rating, count in rating_counts.items():
        st.write(f"⭐ {rating}: {count} movies")

st.divider()

# 3. SYSTEM STATISTICS
st.subheader("🔧 System Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Items", "56,439")
    st.caption("Movies, Anime, Manga")

with col2:
    coverage_pct = num_ratings / 31961 * 100
    st.metric("Catalog Coverage", f"{coverage_pct:.3f}%")
    st.caption(f"{num_ratings:,} of 31,961 movies")

with col3:
    strategy_type = "Cold Start" if num_ratings < 10 else "Hybrid"
    st.metric("Active Strategy", strategy_type)
    st.caption("Adapts to rating count")

with col4:
    recs_available = num_ratings > 0
    st.metric("Recommendations", "Ready" if recs_available else "N/A")
    st.caption("All 3 media types")

st.divider()

# 4. RATING TIMELINE
st.subheader("📅 Your Rating Journey")

if len(rated_items) >= 3:
    # Sort by rating (as proxy for time if no timestamps)
    items_sorted = sorted(rated_items, key=lambda x: x['user_rating'], reverse=True)

    st.write(f"**Top 5 Highest Rated:**")
    for i, item in enumerate(items_sorted[:5], 1):
        st.write(f"{i}. ⭐ {item['user_rating']} - {item['title']} ({', '.join(item['genres'][:2])})")

    st.write(f"\n**Bottom 3 Lowest Rated:**")
    for i, item in enumerate(items_sorted[-3:], 1):
        st.write(f"{i}. ⭐ {item['user_rating']} - {item['title']} ({', '.join(item['genres'][:2])})")

else:
    st.info("Rate more movies to see rating patterns!")

st.divider()

# 5. RECOMMENDATIONS INSIGHTS
st.subheader("💡 Recommendation Insights")

if num_ratings < 3:
    st.warning("⚠️ Need at least 3 ratings to generate meaningful recommendations")
    st.info("Current strategy: Pure popularity")

elif num_ratings < 10:
    st.info("🌱 **Cold Start Mode**")
    st.write(f"- Using genre-based popularity")
    st.write(f"- Rate {10 - num_ratings} more movies to unlock hybrid algorithm")
    st.write(f"- Current genres: {', '.join(df_genres['genre'].unique()[:5])}")

else:
    st.success("🚀 **Hybrid Mode Active**")
    st.write(f"- Combining collaborative filtering + content-based + cross-domain")
    st.write(f"- Personalized recommendations based on {num_ratings} ratings")
    st.write(f"- Cross-media transfer available (movie → anime/manga)")

# Footer
st.divider()
st.caption("Analytics update in real-time as you rate more movies")
