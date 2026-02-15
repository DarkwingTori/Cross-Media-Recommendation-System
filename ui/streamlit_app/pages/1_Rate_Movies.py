"""
Movie Rating Interface
Search for movies and rate them to build your profile.
"""

import streamlit as st
import requests
import pandas as pd

st.title("🎬 Rate Movies")

# Get current number of ratings
num_ratings = len(st.session_state.user_ratings)

st.markdown("""
Search for movies you've seen and rate them. Your ratings will be used to generate
personalized recommendations.

💡 **Tip**: Rate at least 3 movies to get started. Rate 10+ for the best hybrid recommendations!
""")

st.divider()

# Search interface
st.subheader("🔍 Search for Movies")

col1, col2 = st.columns([3, 1])

with col1:
    search_query = st.text_input(
        "Search by title",
        placeholder="e.g., Matrix, Star Wars, Toy Story, Lord of the Rings",
        label_visibility="collapsed"
    )

with col2:
    limit = st.selectbox("Results", [5, 10, 20], index=1)

# Perform search
if search_query:
    try:
        response = requests.get(
            f"{st.session_state.api_url}/items/search",
            params={"query": search_query, "media_type": "movie", "limit": limit},
            timeout=5
        )

        if response.ok:
            data = response.json()
            results = data['results']
            total = data['total_results']

            st.success(f"Found {total} movies (showing top {len(results)})")

            # Display search results
            for item in results:
                with st.expander(f"⭐ {item['title']} ({item.get('year', 'N/A')})"):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.write(f"**Genres**: {', '.join(item['genres'])}")
                        st.write(f"**Average Rating**: ⭐ {item['avg_rating']:.2f}")
                        st.caption(f"{item['rating_count']:,} ratings")

                        if item.get('themes'):
                            st.caption(f"Themes: {', '.join(item['themes'][:3])}")

                    with col2:
                        # Get current rating if exists
                        current_rating = st.session_state.user_ratings.get(item['item_id'], 3.0)

                        # Rating slider
                        rating = st.slider(
                            "Your rating",
                            min_value=0.5,
                            max_value=5.0,
                            value=current_rating,
                            step=0.5,
                            key=f"rating_{item['item_id']}",
                            help="Rate from 0.5 (awful) to 5.0 (masterpiece)"
                        )

                        # Save button
                        if st.button("💾 Save Rating", key=f"save_{item['item_id']}"):
                            st.session_state.user_ratings[item['item_id']] = rating
                            st.success(f"Saved: ⭐ {rating}")
                            st.rerun()

        else:
            st.error(f"Search failed: {response.status_code}")

    except requests.exceptions.Timeout:
        st.error("⏱️ Search timed out. Please try again.")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure it's running: `python scripts/run_api.py`")
    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    st.info("👆 Enter a movie title to search")

st.divider()

# Display current ratings
st.subheader("📝 Your Ratings")

if st.session_state.user_ratings:
    st.write(f"You've rated **{len(st.session_state.user_ratings)} movies**")

    # Build ratings table
    ratings_data = []

    with st.spinner("Loading rated movies..."):
        for item_id, rating in st.session_state.user_ratings.items():
            try:
                item_resp = requests.get(
                    f"{st.session_state.api_url}/items/{item_id}",
                    timeout=2
                )

                if item_resp.ok:
                    item = item_resp.json()
                    ratings_data.append({
                        'Title': item['title'],
                        'Rating': f"⭐ {rating}",
                        'Genres': ', '.join(item['genres'][:3]),
                        'Year': item.get('year', 'N/A'),
                        'Item ID': item_id
                    })
            except:
                # If API call fails, show basic info
                ratings_data.append({
                    'Title': 'Unknown',
                    'Rating': f"⭐ {rating}",
                    'Genres': '',
                    'Year': '',
                    'Item ID': item_id
                })

    if ratings_data:
        # Display as dataframe
        df = pd.DataFrame(ratings_data)
        st.dataframe(df[['Title', 'Rating', 'Genres', 'Year']], use_container_width=True, hide_index=True)

        # Action buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🗑️ Clear All Ratings", type="secondary"):
                st.session_state.user_ratings = {}
                st.success("All ratings cleared!")
                st.rerun()

        with col2:
            # Export ratings
            if st.button("💾 Export Ratings", type="secondary"):
                ratings_json = {
                    "user_ratings": [
                        {"item_id": item_id, "rating": rating}
                        for item_id, rating in st.session_state.user_ratings.items()
                    ]
                }
                st.download_button(
                    "Download JSON",
                    data=str(ratings_json),
                    file_name="my_ratings.json",
                    mime="application/json"
                )

    # Progress indicator
    if num_ratings < 10:
        st.info(f"💡 Rate {10 - num_ratings} more movies to unlock the full hybrid algorithm!")
    else:
        st.success("✅ Hybrid algorithm unlocked! You have enough ratings for personalized CF recommendations.")

else:
    st.info("🎬 You haven't rated any movies yet. Search for movies above to get started!")
    st.caption("Try searching for popular movies like 'Star Wars', 'Matrix', or 'Toy Story'")
