# Required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from itertools import combinations

# Read data from Excel file
file_path = r'C:\Users\shsaf\OneDrive\Desktop\PHD\AI\dataf2.xlsx'
df = pd.read_excel(file_path)

# Select features and lables
X = df[['Cell size', 'CT', 'Q', 'DI']].values
y = df['Category'].values

# Data standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature list
features = ['Cell size', 'CT', 'Q', 'DI']

# List to store results
results_kfold = []
results_loo = []

# Function to evaluate KNN on feature combinations
def evaluate_knn(X_subset, y, n_neighbors=5):
    X_subset_scaled = scaler.fit_transform(X_subset)
    
    # KFold Cross Validation
    kfold = KFold(n_splits=4, shuffle=True, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    kfold_scores = cross_val_score(knn, X_subset_scaled, y, cv=kfold, scoring='accuracy')
    kfold_mean_score = np.mean(kfold_scores)

    # Leave-One-Out Cross Validation
    loo = LeaveOneOut()
    loo_scores = cross_val_score(knn, X_subset_scaled, y, cv=loo, scoring='accuracy')
    loo_mean_score = np.mean(loo_scores)

    return kfold_mean_score, loo_mean_score

# Creating different combinations of features
for r in range(1, len(features)+1):
    for combo in combinations(features, r):
        # Select features
        X_subset = df[list(combo)].values

        # KNN evaluation
        kfold_score, loo_score = evaluate_knn(X_subset, y)
        
        # Save results
        combo_name = ', '.join(combo)
        results_kfold.append((combo_name, kfold_score))
        results_loo.append((combo_name, loo_score))

# Convert results to DataFrame for display
results_kfold_df = pd.DataFrame(results_kfold, columns=['Feature Combination', 'KFold Accuracy'])
results_loo_df = pd.DataFrame(results_loo, columns=['Feature Combination', 'LOO Accuracy'])

# Draw a comparison chart
plt.figure(figsize=(16, 8))

# K-Fold chart
plt.subplot(1, 2, 1)
bars_kfold = plt.barh(results_kfold_df['Feature Combination'], results_kfold_df['KFold Accuracy'], color='blue')
plt.title('K-Fold Cross Validation Accuracy for Different Feature Combinations')
plt.xlabel('Accuracy')
plt.ylabel('Feature Combinations')
plt.xlim(0, 1)

# Display a numeric value next to K-Fold bars
for bar in bars_kfold:
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.4f}', va='center')

# LOO chart
plt.subplot(1, 2, 2)
bars_loo = plt.barh(results_loo_df['Feature Combination'], results_loo_df['LOO Accuracy'], color='green')
plt.title('Leave-One-Out Cross Validation Accuracy for Different Feature Combinations')
plt.xlabel('Accuracy')
plt.ylabel('Feature Combinations')
plt.xlim(0, 1)

# Display a numeric value next to the LOO bars
for bar in bars_loo:
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.4f}', va='center')

plt.tight_layout()
plt.show()

# Finding the best results
best_kfold = results_kfold_df.loc[results_kfold_df['KFold Accuracy'].idxmax()]
best_loo = results_loo_df.loc[results_loo_df['LOO Accuracy'].idxmax()]

print(f"Best Feature Combination (K-Fold): {best_kfold['Feature Combination']} with Accuracy: {best_kfold['KFold Accuracy']:.4f}")
print(f"Best Feature Combination (LOO): {best_loo['Feature Combination']} with Accuracy: {best_loo['LOO Accuracy']:.4f}")