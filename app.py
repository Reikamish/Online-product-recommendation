import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random

# Sample data creation
def create_sample_data():
    products = [
        'Laptop', 'Smartphone', 'Headphones', 'Tablet', 'Smartwatch',
        'Camera', 'Speaker', 'Monitor', 'Keyboard', 'Mouse'
    ]
    
    users = [f'User_{i}' for i in range(1, 101)]
    
    # Create random user-product ratings
    ratings = []
    for user in users:
        for product in products:
            if random.random() < 0.3:  # 30% chance of rating
                rating = random.randint(1, 5)
                ratings.append([user, product, rating])
    
    return pd.DataFrame(ratings, columns=['user', 'product', 'rating']), products

# Recommendation engine
def get_recommendations(user_id, df, products, n=5):
    # Create user-product matrix
    user_product_matrix = df.pivot_table(index='user', columns='product', values='rating')
    
    # Fill NA with 0
    user_product_matrix = user_product_matrix.fillna(0)
    
    # Calculate cosine similarity between users
    user_similarity = cosine_similarity(user_product_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, 
                                   index=user_product_matrix.index,
                                   columns=user_product_matrix.index)
    
    # Get similar users
    if user_id not in user_product_matrix.index:
        return random.sample(products, n)  # Return random products for new users
    
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:11]
    
    # Get products rated by similar users
    similar_users_ratings = user_product_matrix.loc[similar_users.index]
    product_scores = similar_users_ratings.mean().sort_values(ascending=False)
    
    # Remove already rated products
    user_rated = user_product_matrix.loc[user_id]
    user_rated = user_rated[user_rated > 0].index
    recommendations = [p for p in product_scores.index if p not in user_rated]
    
    return recommendations[:n]

# Streamlit app
def main():
    st.set_page_config(page_title="Online Shop Recommendation System", page_icon="🛒")
    
    st.title("🛒 Online Shop Product Recommendation System")
    
    # Create or load data
    @st.cache_data
    def load_data():
        return create_sample_data()
    
    df, products = load_data()
    
    # Sidebar
    st.sidebar.header("User Settings")
    user_id = st.sidebar.selectbox("Select User", sorted(df['user'].unique()))
    
    # Main content
    st.header("Welcome to Our Online Shop!")
    
    # Show recommendations
    st.subheader("Recommended Products for You")
    recommendations = get_recommendations(user_id, df, products)
    
    cols = st.columns(3)
    for i, product in enumerate(recommendations):
        with cols[i % 3]:
            st.image("https://via.placeholder.com/150", caption=product)
            st.write(f"**{product}**")
            rating = st.slider(f"Rate {product}", 1, 5, 3, key=f"rate_{product}")
            if st.button(f"Add to Cart - {product}", key=f"cart_{product}"):
                st.success(f"{product} added to cart!")
    
    # Show user's previous ratings
    st.subheader("Your Previous Ratings")
    user_ratings = df[df['user'] == user_id][['product', 'rating']]
    if not user_ratings.empty:
        st.dataframe(user_ratings)
    else:
        st.write("No ratings yet. Start rating products to get better recommendations!")
    
    # About section
    with st.expander("About This System"):
        st.write("""
        This is a product recommendation system for an online shop built using:
        - Streamlit for the web interface
        - Pandas for data handling
        - Scikit-learn for collaborative filtering
        - Cosine similarity for finding similar users
        
        The system recommends products based on other users' ratings and your own rating history.
        """)

if __name__ == "__main__":
    main()