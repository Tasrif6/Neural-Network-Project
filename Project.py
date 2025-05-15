import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import BertTokenizer, BertModel
from datasets import load_dataset

ds = load_dataset("Salesforce/APIGen-MT-5k")
texts = ds["train"]["system"]
print("Sample text:", texts[0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased").to(device)
model.eval()

def extract_cls_embeddings(texts, batch_size=32):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=128)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            output = model(**encoded)
            cls_embed = output.last_hidden_state[:, 0, :]
            embeddings.append(cls_embed.cpu().numpy())
    return np.vstack(embeddings)

embeddings = extract_cls_embeddings(texts)

pca = PCA(n_components=50)
reduced_embeddings = pca.fit_transform(embeddings)

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(reduced_embeddings)

dbscan = DBSCAN(eps=3, min_samples=5)
dbscan_labels = dbscan.fit_predict(reduced_embeddings)

silhouette = silhouette_score(reduced_embeddings, dbscan_labels)
dbi = davies_bouldin_score(reduced_embeddings, dbscan_labels)
ch = calinski_harabasz_score(reduced_embeddings, dbscan_labels)

print("\n--- BERT Clustering Evaluation Results (DBSCAN) ---")
print(f"Silhouette Score: {silhouette:.4f}")
print(f"Davies-Bouldin Index: {dbi:.4f}")
print(f"Calinski-Harabasz Index: {ch:.4f}")

tsne = TSNE(n_components=2, random_state=42)
tsne_proj = tsne.fit_transform(reduced_embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(tsne_proj[:, 0], tsne_proj[:, 1], c=dbscan_labels, cmap='tab10', s=10)
plt.title("t-SNE Visualization of Clusters (BERT + DBSCAN)")
plt.colorbar()
plt.show()

total_params = sum(p.numel() for p in model.parameters())
print(f"BERT Model Parameters: {total_params:,}")
