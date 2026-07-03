import pickle
import difflib

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Shopper Spectrum", layout="wide")

SEGMENT_COLORS = {
    "High-Value": "#e74c3c",
    "Regular": "#3498db",
    "Occasional": "#2ecc71",
    "At-Risk": "#f39c12",
}

SEGMENT_DESCRIPTIONS = {
    "High-Value": "Recent, frequent, and big spenders.",
    "Regular": "Steady purchasers with moderate frequency and spend.",
    "Occasional": "Infrequent, low-spend, occasional buyers.",
    "At-Risk": "Haven't purchased in a long time.",
}

CLUSTER_LABELS = {
    0: "Occasional",
    1: "At-Risk",
    2: "High-Value",
    3: "Regular",
}


@st.cache_resource
def load_cluster_model():
    with open("kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return kmeans, scaler


@st.cache_resource
def build_recommendation_engine():
    try:
        with open("product_similarity.pkl", "rb") as f:
            similarity_df = pickle.load(f)
        product_names = sorted(similarity_df.columns.tolist())
        return similarity_df, product_names
    except FileNotFoundError:
        pass

    df = pd.read_csv("online_retail.csv")
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df = df.dropna(subset=["Description"])
    df["Description"] = df["Description"].astype(str).str.strip()

    customer_product = df.pivot_table(
        index="CustomerID",
        columns="Description",
        values="Quantity",
        aggfunc="sum",
        fill_value=0,
    )

    similarity_matrix = cosine_similarity(customer_product.T)
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=customer_product.columns,
        columns=customer_product.columns,
    )

    product_names = sorted(customer_product.columns.tolist())
    return similarity_df, product_names


def get_recommendations(product_name, similarity_df, top_n=5):
    if product_name in similarity_df.columns:
        matched_name = product_name
    else:
        upper_map = {p.upper(): p for p in similarity_df.columns}
        if product_name.upper() in upper_map:
            matched_name = upper_map[product_name.upper()]
        else:
            return None, None

    scores = similarity_df[matched_name].drop(labels=[matched_name])
    top_matches = scores.sort_values(ascending=False).head(top_n)
    return matched_name, top_matches


st.sidebar.title("Shopper Spectrum")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Clustering", "Recommendation"],
    label_visibility="collapsed",
)

if page == "Home":
    st.title("Shopper Spectrum")
    st.subheader("Customer Segmentation and Product Recommendations")

    st.write(
        "This app segments customers based on Recency, Frequency, and "
        "Monetary (RFM) behavior using KMeans clustering, and recommends "
        "products using item-based collaborative filtering."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Customer Segmentation")
        st.write(
            "Enter a customer's Recency, Frequency, and Monetary values to "
            "predict which segment they belong to: High-Value, Regular, "
            "Occasional, or At-Risk."
        )
    with col2:
        st.markdown("#### Product Recommendation")
        st.write(
            "Enter a product name to get the top 5 most similar products, "
            "based on customer purchase patterns."
        )

elif page == "Clustering":
    st.title("Customer Segmentation")
    st.caption("Predict a customer's segment from their RFM values.")

    try:
        kmeans, scaler = load_cluster_model()
    except FileNotFoundError:
        st.error("kmeans_model.pkl / scaler.pkl not found in the app folder.")
        st.stop()

    recency = st.number_input(
        "Recency (days since last purchase)",
        min_value=0, max_value=1000, value=30, step=1,
    )
    frequency = st.number_input(
        "Frequency (number of purchases)",
        min_value=1, max_value=1000, value=5, step=1,
    )
    monetary = st.number_input(
        "Monetary (total spend)",
        min_value=0.0, max_value=1_000_000.0, value=500.0, step=10.0, format="%.2f",
    )

    if st.button("Predict Segment", type="primary"):
        input_df = pd.DataFrame(
            [[recency, frequency, monetary]],
            columns=["Recency", "Frequency", "Monetary"],
        )
        scaled_array = scaler.transform(input_df)
        scaled_input = pd.DataFrame(scaled_array, columns=["Recency", "Frequency", "Monetary"])
        cluster = int(kmeans.predict(scaled_input)[0])
        segment = CLUSTER_LABELS.get(cluster, f"Cluster {cluster}")
        color = SEGMENT_COLORS.get(segment, "#7f8c8d")

        st.markdown(
            f"""
            <div style="padding:18px 24px; border-radius:8px; font-size:20px;
            font-weight:600; text-align:center; margin-top:12px;
            background-color:{color}22; border:1px solid {color}; color:{color};">
                This customer belongs to: {segment}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(SEGMENT_DESCRIPTIONS.get(segment, ""))

elif page == "Recommendation":
    st.title("Product Recommender")
    st.caption("Enter a product name to get 5 similar products.")

    try:
        similarity_df, product_names = build_recommendation_engine()
    except FileNotFoundError:
        st.error("online_retail.csv not found in the app folder.")
        st.stop()

    product_input = st.text_input("Enter Product Name")

    if st.button("Recommend", type="primary"):
        if not product_input.strip():
            st.warning("Please enter a product name.")
        else:
            matched_name, recommendations = get_recommendations(
                product_input.strip(), similarity_df, top_n=5
            )

            if matched_name is None:
                st.error(f"'{product_input}' not found in the catalog.")
                close = difflib.get_close_matches(
                    product_input.upper(), product_names, n=5, cutoff=0.4
                )
                if close:
                    st.write("Did you mean:")
                    for name in close:
                        st.markdown(f"- {name}")
            else:
                if matched_name.upper() != product_input.strip().upper():
                    st.caption(f"Showing results for closest match: {matched_name}")

                st.markdown("**Recommended Products:**")
                for name, score in recommendations.items():
                    st.markdown(
                        f'<div style="background-color:#f5f5f8; border-left:4px solid #ff4b4b; '
                        f'padding:10px 16px; border-radius:6px; margin-bottom:8px; '
                        f'font-weight:500; color:#262730;">{name}</div>',
                        unsafe_allow_html=True,
                    )
