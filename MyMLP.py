import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from datasets import load_dataset


class MyMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)


input_size = 784          
hidden_size = 128
output_size = 10          
learning_rate = 0.001
batch_size = 64
epochs = 5


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyMLP(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Training the MLP model on MNIST...")
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.view(data.size(0), -1).to(device)
        targets = targets.to(device)

        scores = model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


print("\nEvaluating model and extracting latent embeddings...")
model.eval()
correct = 0
total = 0
embeddings = []
labels = []

with torch.no_grad():
    for data, targets in test_loader:
        data = data.view(data.size(0), -1).to(device)
        targets = targets.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)

        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        latent = model.net[0](data)  
        embeddings.append(latent.cpu().numpy())
        labels.append(targets.cpu().numpy())

print(f"Test Accuracy: {100 * correct / total:.2f}%")

embeddings = np.vstack(embeddings)
labels = np.hstack(labels)


pca = PCA(n_components=32)
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
plt.title("t-SNE Visualization of MNIST Clusters (KMeans)")
plt.colorbar(scatter, label='Cluster ID')
plt.grid(True)
plt.tight_layout()
plt.show()
