import chromadb
from src.retrieval.query_embedder import embed_text_query, embed_image_query


class Retriever:

    def __init__(self, persist_dir="chroma_db", collection_name="rag_multimodal"):
        print("Connecting to Chroma vector database...")
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_collection(collection_name)

    def search_text(self, query, k=5):
        query_emb = embed_text_query(query)

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=k
        )
        return results

    def search_image(self, image_path, k=5):
        query_emb = embed_image_query(image_path)

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=k
        )
        return results
