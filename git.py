# 🔷 Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 🔷 Step 2: Load Dataset
data = pd.read_csv("customers.csv.csv")   # make sure file is in same folder

print("First 5 rows:")
print(data.head())

# 🔷 Step 3: Data Preprocessing
print("\nMissing values:")
print(data.isnull().sum())

# Remove missing values
data = data.dropna()

# 🔷 Step 4: Select Features
# ⚠️ Ensure column names match your dataset
X = data[['Annual Income (k$)', 'Spending Score (1-100)']].values

# 🔷 Step 5: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔷 Step 6: Elbow Method (Find Best K)
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(range(1,11), wcss, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# 🔷 Step 7: Apply K-Means
kmeans = KMeans(n_clusters=5, random_state=0)
y_kmeans = kmeans.fit_predict(X_scaled)

# Add cluster column
data['Cluster'] = y_kmeans

# 🔷 Step 8: Visualization
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=data['Annual Income (k$)'],
    y=data['Spending Score (1-100)'],
    hue=data['Cluster'],
    palette='Set1'
)

plt.title("Customer Segments")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()

# 🔷 Step 9: Analyze Clusters
print("\nCluster Analysis:")
print(data.groupby('Cluster').mean())

# 🔷 Step 10: Business Insights
print("\nBusiness Insights:")
print("Cluster 0: High spenders (VIP customers)")
print("Cluster 1: Low spenders (Need offers)")
print("Cluster 2: High income but low spending (Target customers)")
print("Cluster 3: Medium group")
print("Cluster 4: Budget customers")