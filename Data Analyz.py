import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.manifold import TSNE
from scipy.stats import f_oneway

# Read data from Excel file
file_path = r'C:\Users\shsaf\OneDrive\Desktop\PHD\AI\dataf2.xlsx'
df = pd.read_excel(file_path)

# انتخاب ویژگی‌ها و برچسب‌ها
# Select features and Lable
X = df[['Cell size', 'CT', 'Q', 'DI']].values
y = df['Category'].values

# Data standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

### 1. Data Distribution Analysis (Histogram)
df[['Cell size', 'CT', 'Q', 'DI']].hist(bins=20, figsize=(10, 8))
plt.suptitle('Distribution of Features')
plt.show()

### 2. Correlation Analysis (Heatmap)

plt.figure(figsize=(8, 6))
corr_matrix = df[['Cell size', 'CT', 'Q', 'DI']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


