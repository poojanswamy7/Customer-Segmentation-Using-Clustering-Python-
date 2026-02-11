import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# STEP 1 - LOAD DATASET
print("===== STEP 1: LOAD DATASET ")
data = pd.read_csv("D:/Infotach Pvt.Lt Projects/Project 2/archive (2)/Shopping_data.csv")
print("Dataset Loaded Successfully!")
print("FIRST 5 ROWS:", data.head())

# STEP 2 - CLEAN DATA
print("===== STEP 2: CLEANING DATA ")
print("Dataset Info ")
print(data.info())
print("Dataset Shape:", data.shape)
print("Missing Values Before Cleaning")
print(data.isnull().sum())

# Handle missing values
num_cols = data.select_dtypes(include=['int64', 'float64']).columns
cat_cols = data.select_dtypes(include=['object']).columns
data[num_cols]=data[num_cols].fillna(data[num_cols].mean())
data[cat_cols]=data[cat_cols].fillna(data[cat_cols].mode().iloc[0])
print("Missing Values After Cleaning")
print(data.isnull().sum())

# Remove duplicates
duplicate_count = data.duplicated().sum()
print(f"Duplicates Found: {duplicate_count}")
data = data.drop_duplicates()
print("Duplicates are Removed!")
print("New Shape:",data.shape)

# Fix data types
if 'Age' in data.columns:
    data['Age'] = data['Age'].astype(int)
if 'Purchase Amount (USD)' in data.columns:
    data['Purchase Amount (USD)'] = data['Purchase Amount (USD)'].astype(float)
print("Updated Data Types")
print(data.dtypes)

# STEP 3 - EDA (Exploratory Data Analysis)
print(" ===== STEP 3: EDA - SUMMARY STATISTICS ")
print("Basic Summary Statistics")
print(data.describe())
print("Column Names:\n", data.columns)

# ======== HISTOGRAM: AGE
plt.figure(figsize=(6,4))
sns.histplot(data['Age'], kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ======== HISTOGRAM: PURCHASE AMOUNT 
plt.figure(figsize=(6,4))
sns.histplot(data['Purchase Amount (USD)'], kde=True)
plt.title("Purchase Amount Distribution")
plt.xlabel("Purchase Amount (USD)")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ======== BOXPLOT: AGE
plt.figure(figsize=(6,4))
sns.boxplot(x=data['Age'])
plt.title("Boxplot of Age")
plt.xlabel("Age")
plt.tight_layout()
plt.show()

# ====== BOXPLOT: PURCHASE AMOUNT 
plt.figure(figsize=(6,4))
sns.boxplot(x=data['Purchase Amount (USD)'])
plt.title("Boxplot of Purchase Amount (USD)")
plt.xlabel("Purchase Amount (USD)")
plt.tight_layout()
plt.show()

# =======PAIRPLOT 
numeric_features = ['Age', 'Purchase Amount (USD)', 'Review Rating']
sns.pairplot(data[numeric_features])
plt.suptitle("Pairplot: Age, Purchase Amount, Review Rating", y=1.02)
plt.show()

# ======= CORRELATION HEATMAP 
plt.figure(figsize=(6,4))
corr = data[numeric_features].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.xlabel("Features")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# STEP 4 - SCALING
print("===== STEP 4: SCALING DATA ")

features = ['Age', 'Purchase Amount (USD)', 'Review Rating']
print("\nNumeric Columns Selected for Scaling:", features)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])

# STEP 5: ELBOW METHOD
print("=====STEP 5: Running Elbow Method")
inertia = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
plt.figure(figsize=(7, 4))
plt.plot(K_range, inertia, '-o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid()
elbow_img = r"D:\Infotach Pvt.Lt Projects\Project 2\archive (2)\Elbow_Method.png"
plt.savefig(elbow_img)
plt.close()
print("Elbow Method graph saved at:", elbow_img)

# STEP 6: SILHOUETTE SCORE
print("=====STEP 6: Running Silhouette Score")
silhouette_scores = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    silhouette_scores.append(sil)
    print(f"k={k} → silhouette={sil:.3f}")
plt.figure(figsize=(7, 4))
plt.plot(K_range, silhouette_scores, '-o')
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores for Different k")
plt.grid()
silhouette_img = r"D:\Infotach Pvt.Lt Projects\Project 2\archive (2)\Silhouette_Scores.png"
plt.savefig(silhouette_img)
plt.close()
print("Silhouette graph saved at:", silhouette_img)

# STEP 7: FINAL MODEL FITTING
print("=====STEP 7:FINAL MODEL FITTING (kmeans)")
k_final = 4
print(f"Applying Final KMeans (k={k_final})")
kmeans = KMeans(n_clusters=k_final, random_state=42, n_init=50)
labels = kmeans.fit_predict(X_scaled)
data['cluster'] = labels
print("\nCluster Counts:")
print(data['cluster'].value_counts())

# STEP 8: CLUSTER CENTROIDS
print("=====STEP 8: Cluster Centroids (Scaled)")
centroids_scaled = pd.DataFrame(kmeans.cluster_centers_, columns=features)
print(centroids_scaled)
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
centroids_orig_df = pd.DataFrame(centroids_original, columns=features)
print("Cluster Centroids (Original Units)")
print(centroids_orig_df.round(2))

# STEP 9: PCA VISUALIZATION
print("=====STEP 9: PCA Visualization")
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)
plt.figure(figsize=(7, 6))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=labels, palette="tab10")
plt.title("K-Means Clusters (PCA Projection)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
pca_img = r"D:\Infotach Pvt.Lt Projects\Project 2\archive (2)\PCA_Clusters.png"
plt.savefig(pca_img)
plt.close()
print("PCA Cluster plot saved at:", pca_img)

# STEP 10: CLUSTER PROFILE SUMMARY
cluster_profile = data.groupby('cluster')[features].agg(['mean', 'median', 'count'])
cluster_csv = r"D:\Infotach Pvt.Lt Projects\Project 2\archive (2)\Cluster_Profiles.csv"
cluster_profile.to_csv(cluster_csv)
print("Cluster Profiles saved to:", cluster_csv)

# STEP 11: SAVE FINAL OUTPUT
output_csv = r"D:\Infotach Pvt.Lt Projects\Project 2\archive (2)\ShoppingData_With_Clusters.csv"
data.to_csv(output_csv, index=False)
print(f"FINAL OUTPUT SAVED → {output_csv}")