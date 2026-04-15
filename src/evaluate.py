import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

def evaluate_clustering(X, true_labels, pred_labels):
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    sil = silhouette_score(X, pred_labels)
    
    print(f"Clustering Results:")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
    print(f"Silhouette Score: {sil:.4f}")
    
    return {"ARI": ari, "NMI": nmi, "Silhouette": sil}

def evaluate_classification(y_true, y_pred, classes, title="Model"):
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"{title} Classification Results:")
    print(f"Macro F1 Score: {f1:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    
    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig(f"results/confusion_matrix_{title.lower().replace(' ', '_')}.png")
    plt.close()
    
    return {"F1": f1, "Precision": precision, "Recall": recall}

def plot_pca(X_pca, labels, title="PCA Projection"):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='viridis', style=labels)
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(title='Cancer Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"results/pca_projection.png")
    plt.close()
