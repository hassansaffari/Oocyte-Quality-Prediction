# Required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

# Read data from Excel file
file_path = r'C:\Users\shsaf\OneDrive\Desktop\PHD\AI\dataf2.xlsx'
df = pd.read_excel(file_path)

# Select features and tags
X = df[['Cell size', 'CT', 'Q', 'DI']].values
y = df['Category'].values

# Data standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# List of models
models = {
    'KMeans': KMeans(n_clusters=2, random_state=42),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=2),
    'Agglomerative Clustering': AgglomerativeClustering(n_clusters=2),
    'Gaussian Mixture': GaussianMixture(n_components=2, random_state=42)
}

# Save results
results = {}

# Running algorithms and calculating accuracy
for name, model in models.items():
    model.fit(X_scaled)

    # labels Prediction
    labels = model.predict(X_scaled) if hasattr(model, 'predict') else model.labels_
    
    # Calculate different score
    silhouette_avg = silhouette_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)
    calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
    ari = adjusted_rand_score(y, labels)
    
    # Save results
    results[name] = {
        'labels': labels,
        'silhouette_score': silhouette_avg,
        'davies_bouldin_score': davies_bouldin,
        'calinski_harabasz_score': calinski_harabasz,
        'ari_score': ari
    }

# Draw a scatter plot with solid and hollow circles
plt.figure(figsize=(12, 10))

for name, result in results.items():
    plt.subplot(2, 2, list(results.keys()).index(name) + 1)  # ایجاد زیرنمودار برای هر الگوریتم
    plt.title(f'{name} Clustering Result')

    # Right and wrong points of classification
    for i in np.unique(result['labels']):
        correct_mask = (result['labels'] == i) & (y-1 == i)  
        incorrect_mask = (result['labels'] == i) & (y-1 != i)  
        
        # Correct points with solid circles
        plt.scatter(X_scaled[correct_mask, 1], X_scaled[correct_mask, 0], 
                    color='blue', marker='o', label='Correct' if i == 0 else "", alpha=0.7)
        
        # Wrong points with hollow circle
        plt.scatter(X_scaled[incorrect_mask, 1], X_scaled[incorrect_mask, 0], 
                    color='red', marker='o', edgecolor='black', facecolor='none', label='Incorrect' if i == 0 else "", alpha=0.7)

    plt.xlabel('CT (Feature 2)')
    plt.ylabel('Cell size (Feature 1)')
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()

# Plotting Silhouette, Davies-Bouldin, Calinski-Harabasz and ARI scores
# Silhouette Score
plt.figure(figsize=(10, 6))
model_names = list(results.keys())
silhouette_scores = [result['silhouette_score'] for result in results.values()]

plt.bar(model_names, silhouette_scores, color='skyblue')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores of Clustering Algorithms')
plt.ylim(0, 1)  
for index, value in enumerate(silhouette_scores):
    plt.text(index, value + 0.01, f'{value:.2f}', ha='center', va='bottom')

plt.grid(axis='y')
plt.show()

# Davies-Bouldin Score
plt.figure(figsize=(10, 6))
davies_bouldin_scores = [result['davies_bouldin_score'] for result in results.values()]

plt.bar(model_names, davies_bouldin_scores, color='lightgreen')
plt.ylabel('Davies-Bouldin Score')
plt.title('Davies-Bouldin Scores of Clustering Algorithms')
for index, value in enumerate(davies_bouldin_scores):
    plt.text(index, value + 0.01, f'{value:.2f}', ha='center', va='bottom')

plt.grid(axis='y')
plt.show()

# Calinski-Harabasz Score
plt.figure(figsize=(10, 6))
calinski_harabasz_scores = [result['calinski_harabasz_score'] for result in results.values()]
plt.bar(model_names, calinski_harabasz_scores, color='coral')
plt.ylabel('Calinski-Harabasz Score')
plt.title('Calinski-Harabasz Scores of Clustering Algorithms')
for index, value in enumerate(calinski_harabasz_scores):
    plt.text(index, value + 0.01, f'{value:.2f}', ha='center', va='bottom')

plt.grid(axis='y')
plt.show()

# Adjusted Rand Index (ARI) Score
plt.figure(figsize=(10, 6))
ari_scores = [result['ari_score'] for result in results.values()]

plt.bar(model_names, ari_scores, color='purple')
plt.ylabel('ARI Score')
plt.title('Adjusted Rand Index (ARI) Scores of Clustering Algorithms')
plt.ylim(0, 1)
for index, value in enumerate(ari_scores):
    plt.text(index, value + 0.01, f'{value:.2f}', ha='center', va='bottom')

plt.grid(axis='y')
plt.show()

# Display the best algorithm based on each criterion
best_silhouette_model = max(results, key=lambda x: results[x]['silhouette_score'])
best_davies_bouldin_model = min(results, key=lambda x: results[x]['davies_bouldin_score'])
best_calinski_harabasz_model = max(results, key=lambda x: results[x]['calinski_harabasz_score'])
best_ari_model = max(results, key=lambda x: results[x]['ari_score'])

print(f"The best clustering method by Silhouette Score is: {best_silhouette_model} with a score of {results[best_silhouette_model]['silhouette_score']:.2f}")
print(f"The best clustering method by Davies-Bouldin Index is: {best_davies_bouldin_model} with a score of {results[best_davies_bouldin_model]['davies_bouldin_score']:.2f}")
print(f"The best clustering method by Calinski-Harabasz Index is: {best_calinski_harabasz_model} with a score of {results[best_calinski_harabasz_model]['calinski_harabasz_score']:.2f}")
print(f"The best clustering method by ARI is: {best_ari_model} with a score of {results[best_ari_model]['ari_score']:.2f}")