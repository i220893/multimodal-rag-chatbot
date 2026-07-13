import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import chromadb
import os

# Ensure report_plots directory exists
OUTPUT_DIR = "report_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_embeddings(collection, limit=2000):
    print("Fetching embeddings from ChromaDB...")
    results = collection.get(include=["embeddings", "metadatas"])
    
    if results["embeddings"] is None or len(results["embeddings"]) == 0:
        print("No embeddings found in the collection!")
        return None, None
        
    embeddings = np.array(results["embeddings"][:limit])
    types = [m["type"] for m in results["metadatas"][:limit]]
    
    print(f"Loaded {len(embeddings)} embeddings.")
    return embeddings, types


def plot_pca(embeddings, types):
    print("Running PCA...")
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    
    # Map types to colors
    unique_types = list(set(types))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_types)))
    type_color_map = dict(zip(unique_types, colors))
    
    for t in unique_types:
        indices = [i for i, x in enumerate(types) if x == t]
        plt.scatter(reduced[indices, 0], reduced[indices, 1], 
                   label=t, alpha=0.6, s=20, c=[type_color_map[t]])
                   
    plt.title("PCA Projection of RAG Embeddings")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = f"{OUTPUT_DIR}/embeddings_pca.png"
    plt.savefig(output_path)
    print(f"Saved PCA plot to {output_path}")


def plot_tsne(embeddings, types):
    print("Running t-SNE (this may take a moment)...")
    # Adjust perplexity based on dataset size
    perplexity = min(30, len(embeddings) - 1) if len(embeddings) > 1 else 1
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    
    # Map types to colors
    unique_types = list(set(types))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_types)))
    type_color_map = dict(zip(unique_types, colors))
    
    for t in unique_types:
        indices = [i for i, x in enumerate(types) if x == t]
        plt.scatter(reduced[indices, 0], reduced[indices, 1], 
                   label=t, alpha=0.6, s=20, c=[type_color_map[t]])
                   
    plt.title("t-SNE Projection of RAG Embeddings")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = f"{OUTPUT_DIR}/embeddings_tsne.png"
    plt.savefig(output_path)
    print(f"Saved t-SNE plot to {output_path}")

if __name__ == "__main__":
    # Connect to ChromaDB
    persist_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_db")
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection("rag_multimodal")
    
    embeddings, types = load_embeddings(collection)
    
    if embeddings is not None and len(embeddings) > 0:
        plot_pca(embeddings, types)
        plot_tsne(embeddings, types)
