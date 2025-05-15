import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.Grayscale(num_output_channels=3), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  
                         [0.229, 0.224, 0.225])
])

batch_size = 64
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = models.vgg16(pretrained=True).features.to(device)
vgg.eval()


print("Extracting features from VGG...")
embeddings = []
labels = []

with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(device)
        output = vgg(images)  
        output = torch.flatten(output, start_dim=1)  
        embeddings.append(output.cpu().numpy())
        labels.append(targets.cpu().numpy())

embeddings = np.vstack(embeddings)
labels = np.hstack(labels)


pca = PCA(n_components=50)
reduced_embeddings = pca.fit_transform(embeddings)


kmeans = KMeans(n_clusters=10, random_state=42)
kmeans_labels = kmeans.fit_predict(reduced_embeddings)

dbscan = DBSCAN(eps=2.0, min_samples=5)
dbscan_labels = dbscan.fit_predict(reduced_embeddings)


print("\n--- Clustering Evaluation Results (KMeans) ---")
sil_kmeans = silhouette_score(reduced_embeddings, kmeans_labels)
dbi_kmeans = davies_bouldin_score(reduced_embeddings, kmeans_labels)
ch_kmeans = calinski_harabasz_score(reduced_embeddings, kmeans_labels)

print(f"Silhouette Score          : {sil_kmeans:.4f}  (higher is better)")
print(f"Davies-Bouldin Index      : {dbi_kmeans:.4f}       (lower is better)")
print(f"Calinski-Harabasz Index   : {ch_kmeans:.4f}        (higher is better)")


print("\nVisualizing clusters with t-SNE...")
tsne = TSNE(n_components=2, random_state=42)
tsne_proj = tsne.fit_transform(reduced_embeddings)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(tsne_proj[:, 0], tsne_proj[:, 1], c=kmeans_labels, cmap='tab10', s=20, alpha=0.7, edgecolors='k')
plt.title("t-SNE Visualization of MNIST Clusters (VGG16 + KMeans)")
plt.colorbar(scatter, label='Cluster ID')
plt.grid(True)
plt.tight_layout()
plt.show()
