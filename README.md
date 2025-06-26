# Machine Learning and Microfluidic Integration for Oocyte Quality Prediction

This repository contains the complete set of Python scripts used for data analysis and model development for the paper:  
**"Machine Learning and Microfluidic Integration for Oocyte Quality Prediction"**

## 📂 Project Structure

| File Name | Description |
|-----------|-------------|
| `Data Analyz.py` | Exploratory Data Analysis (EDA), including distribution histograms and correlation heatmap. |
| `Comparison of clustering results using four different algorithms.py` | Clustering using KMeans, DBSCAN, Agglomerative, and GMM + evaluation (Silhouette, ARI, etc). |
| `Comparison of K-fold and LOO Cross-Validation Results for Classification Algorithms.py` | Evaluates feature combinations using KNN with both K-Fold and Leave-One-Out CV. |
| `Find the best k for KNN method.py` | Determines the optimal K value for KNN classification via K-Fold cross-validation. |

## 📊 Dataset

The dataset used in these scripts is included in this repository as `dataf2.xlsx`.

It includes:
- **Features:** `Cell size`, `CT`, `Q`, `DI`  
- **Label:** `Category` (used for classification/clustering)

## 📦 Requirements

You can install the required libraries via:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn openpyxl
```

Python version: **3.8 or higher**

## 🚀 How to Run

Clone this repository:

```bash
git clone https://github.com/hassansaffari/Oocyte-Quality-Prediction.git
cd Oocyte-Quality-Prediction
```

Then, run any of the scripts using:

```bash
python "Comparison of clustering results using four different algorithms.py"
```

Make sure the file `dataf2.xlsx` is present in the same directory as the script or update the file path accordingly.

## 📈 Output Examples

- Clustering results visualized with scatter plots (correct vs incorrect clusters)
- Evaluation metrics: Silhouette Score, ARI, Davies-Bouldin, Calinski-Harabasz
- KNN classification accuracy over different K values and feature sets

## 🧠 Related Publication

This code is part of the supplementary material for the manuscript:  
*"Machine Learning and Microfluidic Integration for Oocyte Quality Prediction"*  
(Under review at **Scientific Reports**)

## 📝 License

Feel free to use, modify, and cite this work with attribution. License: **MIT**
