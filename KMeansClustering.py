import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# load data
df = pd.read_csv("kiva_loans.csv")

# create dataframe
df = df[['loan_amount', 'sector', 'term_in_months', 'lender_count', 'borrower_genders', 'repayment_interval']]

# encode categorical variables
df['sector'] = pd.Categorical(df['sector']).codes
df['borrower_genders'] = pd.Categorical(df['borrower_genders']).codes
df['repayment_interval'] = pd.Categorical(df['repayment_interval']).codes

# scale 
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)

# apply pca for dimensionality reduction
pca = PCA(n_components=2)
pca_df = pca.fit_transform(scaled_df)

# determine the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(pca_df)
    wcss.append(kmeans.inertia_)
    
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# fit the K-means model with the optimal number of clusters
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42)
pred_clusters = kmeans.fit_predict(pca_df)

# visualize the clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(x=pca_df[:, 0], y=pca_df[:, 1], hue=pred_clusters, palette='viridis', legend='full')
plt.title("K-means Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
