"""Embeddings management module for RAG system."""

import time
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI


class EmbeddingsManager:
    def __init__(self, config):
        """Initialize the embeddings manager.

        Args:
            config (dict): Configuration dictionary containing API keys and model settings
        """
        self.config = config
        self.client = OpenAI(api_key=config["openai_api_key"])
        self.embedding_model = config["embedding_model"]
        self.dimension = None  # Will be set after first embedding creation

    def create_embedding(self, text):
        """Create embedding for a single text.

        Args:
            text (str): Text to create embedding for

        Returns:
            list: Embedding vector
        """
        response = self.client.embeddings.create(model=self.embedding_model, input=text)
        embedding = response.data[0].embedding

        if self.dimension is None:
            self.dimension = len(embedding)

        return embedding

    def create_embeddings(self, texts, show_progress=True):
        """Create embeddings for multiple texts.

        Args:
            texts (list): List of texts to create embeddings for
            show_progress (bool): Whether to show progress information

        Returns:
            list: List of embedding vectors
        """
        embeddings = []
        start_time = time.time()

        for i, text in enumerate(texts):
            if show_progress and i % 5 == 0:
                print(f"Embedding作成中: {i + 1}/{len(texts)}")

            embedding = self.create_embedding(text)
            embeddings.append(embedding)

        if show_progress:
            processing_time = time.time() - start_time
            print(f"\n埋め込み完了: {len(embeddings)}件のテキストを処理しました")
            print(f"処理時間: {processing_time:.2f}秒")

        return embeddings

    def create_index(self, embeddings):
        """Create FAISS index from embeddings.

        Args:
            embeddings (list): List of embedding vectors

        Returns:
            faiss.Index: FAISS index containing the embeddings
        """
        # Convert to numpy array
        embeddings_np = np.array(embeddings).astype("float32")

        # Create FAISS index
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)

        # Add vectors to index
        index.add(embeddings_np)
        print(f"FAISSインデックスにベクトルを追加しました: {index.ntotal}件")

        return index

    def search(self, index, query_embedding, top_k=3):
        """Search for similar vectors in the index.

        Args:
            index (faiss.Index): FAISS index to search in
            query_embedding (list): Query embedding vector
            top_k (int): Number of results to return

        Returns:
            tuple: (distances, indices) of the search results
        """
        query_embedding_np = np.array([query_embedding]).astype("float32")
        return index.search(query_embedding_np, top_k)
