import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_pickle("amazon.pkl")

df['product_name'] = df['product_name'].fillna('')
df['about_product'] = df['about_product'].fillna('')

combined_text = df['product_name'] + " " + df['about_product']

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(combined_text)

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

st.title("🛍️ Product Recommender System")

user_input = st.text_input("Enter a product name")

top_n = st.selectbox("Select number of recommendations:", [5, 10, 15])

def recommend_products(product_name, top_n):
    
    matching_products = df[df['product_name'].str.contains(product_name, case=False, na=False) |
                           df['about_product'].str.contains(product_name, case=False, na=False)]

    if matching_products.empty:
        st.warning("No products found matching your search term!")
        return pd.DataFrame()

    matching_indices = matching_products.index.tolist()

    recommendations = []
    for idx in matching_indices:
        similarity_scores = list(enumerate(cosine_sim[idx]))
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        top_similar = [i for i in sorted_scores if i[0] != idx][:top_n]

        for i in top_similar:
            recommendations.append({
                "Product Name": df.iloc[i[0]]['product_name'],
                "Similarity Score": round(i[1], 3)
            })

    return pd.DataFrame(recommendations).drop_duplicates().head(top_n)

if st.button("Recommend"):
    if user_input:
        results = recommend_products(user_input, top_n)
        if not results.empty:
            st.subheader("Top Recommendations:")
            st.dataframe(results)
        else:
            st.info("No similar products found.")
    else:
        st.warning("Please enter a product name.")
