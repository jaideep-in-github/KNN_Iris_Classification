# KNN Classification on Iris Dataset
# Elevale AI & ML Internship, Task 6
# A unique implementation with feature selection and custom visualizations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Load and explore the Iris dataset
data = pd.read_csv('Iris.csv')
X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y = pd.Categorical(data['Species']).codes  # Convert species to 0, 1, 2
class_names = pd.Categorical(data['Species']).categories  # ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# Step 2: Feature selection (use petal features for 2D visualization)
X_petal = X[:, [2, 3]]  # PetalLengthCm, PetalWidthCm

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_petal, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Experiment with K values and distance metrics
k_values = [3, 5, 7, 9]
metrics = ['euclidean', 'minkowski']  # Minkowski with p=1.5 for uniqueness
results = []

for metric in metrics:
    for k in k_values:
        # Configure KNN (use p=1.5 for Minkowski)
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric, p=1.5 if metric == 'minkowski' else 2)
        knn.fit(X_train_scaled, y_train)
        
        # Predict and evaluate
        y_pred = knn.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results.append({
            'Metric': metric,
            'K': k,
            'Accuracy': accuracy,
            'Confusion_Matrix': cm
        })
        
        # Plot confusion matrix as heatmap
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix (K={k}, {metric.capitalize()})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'cm_k{k}_{metric}.png')
        plt.close()

# Step 4: Visualize decision boundaries (K=5, Euclidean)
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train_scaled, y_train)

# Create mesh grid
h = 0.02
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict on mesh
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']), alpha=0.3)
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF']), edgecolors='k', label='Train')
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF']), marker='^', edgecolors='k', label='Test')
plt.xlabel('Petal Length (Scaled)')
plt.ylabel('Petal Width (Scaled)')
plt.title('Decision Boundary (K=5, Euclidean, Petal Features)')
plt.legend()
plt.savefig('decision_boundary.png')
plt.close()

# Step 5: Plot accuracy vs. K
results_df = pd.DataFrame(results)
plt.figure(figsize=(8, 5))
for metric in metrics:
    metric_data = results_df[results_df['Metric'] == metric]
    plt.plot(metric_data['K'], metric_data['Accuracy'], marker='o', label=metric.capitalize())
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. K for Different Metrics')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_vs_k.png')
plt.close()

# Step 6: Feature importance (creative: compare performance with all features)
X_all = X  # All 4 features
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y, test_size=0.2, random_state=42)
X_train_all_scaled = scaler.fit_transform(X_train_all)
X_test_all_scaled = scaler.transform(X_test_all)

knn_all = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_all.fit(X_train_all_scaled, y_train_all)
accuracy_all = accuracy_score(y_test_all, knn_all.predict(X_test_all_scaled))

# Plot feature importance (based on accuracy difference)
plt.figure(figsize=(6, 4))
plt.bar(['Petal Features (2)', 'All Features (4)'], [results_df[(results_df['Metric'] == 'euclidean') & (results_df['K'] == 5)]['Accuracy'].values[0], accuracy_all], color=['#1f77b4', '#ff7f0e'])
plt.ylabel('Accuracy')
plt.title('Feature Selection Impact on Accuracy (K=5, Euclidean)')
plt.savefig('feature_importance.png')
plt.close()

# Print results
print("Results Summary (Petal Features):")
for result in results:
    print(f"Metric: {result['Metric'].capitalize()}, K: {result['K']}, Accuracy: {result['Accuracy']:.4f}")
print(f"Accuracy with All Features (K=5, Euclidean): {accuracy_all:.4f}")