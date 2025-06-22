# Required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

# Read data from Excel file
file_path = r'C:\Users\shsaf\OneDrive\Desktop\PHD\AI\dataf2.xlsx'
df = pd.read_excel(file_path)

# Select features and lables
X = df[['Cell size', 'CT', 'Q', 'DI']].values
y = df['Category'].values

# Data standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find the best value of K
k_values = list(range(1, 31))  # Range of k we want to check
kfold = KFold(n_splits=4, shuffle=True, random_state=42)  

mean_scores = []  # A list to hold the average accuracies for each value of K

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    # Calculate the average accuracy with Cross-Validation
    scores = cross_val_score(knn, X_scaled, y, cv=kfold, scoring='accuracy')
    mean_scores.append(np.mean(scores))

# Find the best value of K
best_k = k_values[np.argmax(mean_scores)]
best_score = max(mean_scores)

# Show result
print(f"Best K: {best_k}")
print(f"Best Cross-Validation Accuracy: {best_score * 100:.2f}%")

# Plotting K vs. Accuracy
plt.figure(figsize=(10, 6))
plt.plot(k_values, mean_scores, marker='o', linestyle='-', color='b')
plt.xlabel('K Value')
plt.ylabel('Cross-Validation Accuracy')
plt.title('K Value vs. Cross-Validation Accuracy')
plt.grid(True)

# Display the accuracy value next to each point
for i, score in enumerate(mean_scores):
    plt.text(k_values[i], score, f'{score:.2f}', ha='center', va='bottom')

plt.show()