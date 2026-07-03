# Shopper Spectrum: Customer Segmentation and Product Recommendations in E-Commerce

A customer analytics project for the e-commerce and retail domain. It segments customers using RFM (Recency, Frequency, Monetary) analysis with KMeans clustering, and recommends products using item-based collaborative filtering, deployed as an interactive Streamlit app.

## Live App

[Add your deployed Streamlit Cloud link here once deployed]

## Problem Statement

E-commerce platforms generate large volumes of transaction data daily. This project analyzes that data to:

- Segment customers into meaningful groups (High-Value, Regular, Occasional, At-Risk) based on purchasing behavior
- Recommend similar products to customers based on purchase patterns across the customer base

## Dataset

[Online Retail Dataset](https://drive.google.com/file/d/1rzRwxm_CJxcRzfoo9Ix37A2JTlMummY-/view?usp=sharing)

Transaction-level data (2022-2023) from an online retail business, with columns for InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, and Country.

## Project Structure

```
├── app.py                    # Streamlit application
├── Shopper_Spectrum.ipynb    # Data cleaning, EDA, clustering, and recommendation analysis
├── kmeans_model.pkl          # Trained KMeans clustering model
├── scaler.pkl                # StandardScaler fitted on RFM values
├── product_similarity.pkl    # Precomputed item-item cosine similarity matrix
├── requirements.txt          # Python dependencies
└── .gitignore
```

## Approach

**Data Preprocessing**

- Removed rows with missing CustomerID
- Excluded cancelled invoices (InvoiceNo starting with 'C')
- Removed non-positive Quantity and UnitPrice values

**Customer Segmentation**

- Calculated Recency, Frequency, and Monetary values per customer
- Standardized RFM values using StandardScaler
- Used the Elbow Method and Silhouette Score to select the number of clusters
- Trained a KMeans model (k=4) and labeled clusters based on their average RFM profile:
  - **High-Value** — recent, frequent, big spenders
  - **Regular** — steady purchasers with moderate frequency and spend
  - **Occasional** — infrequent, low-spend buyers
  - **At-Risk** — haven't purchased in a long time

**Product Recommendation**

- Built a Customer x Product matrix from purchase quantities
- Computed item-item cosine similarity between products
- Given a product name, returns the top 5 most similar products

## Streamlit App

Two interactive modules:

- **Clustering** — enter a customer's Recency, Frequency, and Monetary values to predict their segment
- **Recommendation** — enter a product name to get 5 similar product recommendations

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit
