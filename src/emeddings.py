from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import chromadb

class EmbeddingManager:
    def __init__(self):
        # Use TF-IDF vectorizer instead of sentence-transformers
        self.vectorizer = TfidfVectorizer()
        # Initialize ChromaDB with new configuration
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name="attention_paper"
        )

    def generate_embeddings(self, texts: List[str]) -> None:
        """Generate embeddings for text chunks and store in ChromaDB."""
        if not texts:
            return

        # Generate TF-IDF embeddings
        embeddings = self.vectorizer.fit_transform(texts).toarray()

        # Add documents to the collection with unique IDs
        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            ids=[f"chunk_{i}" for i in range(len(texts))]
        )

    def search_similar(self, query: str, n_results: int = 3) -> List[str]:
        """Search for similar chunks based on query."""
        # Transform query using the same vectorizer
        query_embedding = self.vectorizer.transform([query]).toarray()
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        return results['documents'][0]