"""Core RAG (Retrieval-Augmented Generation) implementation."""

import glob
import re
from pathlib import Path

from openai import OpenAI

from .cache import CacheManager
from .embeddings import EmbeddingsManager


class RAG:
    def __init__(self, config):
        """Initialize the RAG system.

        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.client = OpenAI(api_key=config["openai_api_key"])
        self.embeddings_manager = EmbeddingsManager(config)
        self.cache_manager = (
            CacheManager(config) if config.get("use_cache", True) else None
        )

        # Initialize state
        self.chunks = []
        self.chunk_metadata = []
        self.index = None

    def split_into_chunks(self, text, chunk_size=500, overlap=50):
        """Split text into chunks.

        Args:
            text (str): Text to split
            chunk_size (int): Maximum size of each chunk
            overlap (int): Number of characters to overlap between chunks

        Returns:
            list: List of text chunks
        """
        # Use headings as split points
        headings = list(re.finditer(r"^#{1,6}\s+.*$", text, re.MULTILINE))
        sections = []

        if not headings:
            # Simple character-based splitting if no headings
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                if end < len(text) and not text[end].isspace():
                    end = text.rfind(" ", start, end)
                    if end == -1:
                        end = min(start + chunk_size, len(text))
                chunks.append(text[start:end])
                start = end - overlap if end > overlap else 0
            return chunks

        # Split by headings
        for i in range(len(headings)):
            start = headings[i].start()
            if i < len(headings) - 1:
                end = headings[i + 1].start()
            else:
                end = len(text)
            sections.append(text[start:end])

        # Split sections into chunks
        chunks = []
        for section in sections:
            if len(section) <= chunk_size:
                chunks.append(section)
            else:
                section_chunks = []
                start = 0
                while start < len(section):
                    end = min(start + chunk_size, len(section))
                    if end < len(section) and not section[end].isspace():
                        end = section.rfind(" ", start, end)
                        if end == -1:
                            end = min(start + chunk_size, len(section))
                    section_chunks.append(section[start:end])
                    start = end - overlap if end > overlap else 0
                chunks.extend(section_chunks)

        return chunks

    def load_documents(self):
        """Load and process documents.

        Returns:
            tuple: (chunks, chunk_metadata, num_docs)
        """
        path = Path(self.config["docs_dir"])
        doc_files = glob.glob(str(path / "*.md"))

        chunks = []
        chunk_metadata = []

        for file_path in doc_files:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                doc_name = Path(file_path).name

                # Split into chunks
                doc_chunks = self.split_into_chunks(
                    content, self.config["chunk_size"], self.config["chunk_overlap"]
                )

                # Store chunks and metadata
                for i, chunk in enumerate(doc_chunks):
                    chunks.append(chunk)
                    chunk_metadata.append(
                        {
                            "doc_name": doc_name,
                            "chunk_index": i,
                            "total_chunks": len(doc_chunks),
                        }
                    )

        print(f"読み込んだドキュメント: {len(doc_files)}件")
        print(f"生成されたチャンク数: {len(chunks)}件")

        self.chunks = chunks
        self.chunk_metadata = chunk_metadata

        return chunks, chunk_metadata, len(doc_files)

    def create_index(self):
        """Create search index from documents.

        Returns:
            faiss.Index: FAISS index
        """
        if not self.chunks:
            raise ValueError("No documents loaded. Call load_documents() first.")

        # Create embeddings
        embeddings = []
        for i, chunk in enumerate(self.chunks):
            if i % 5 == 0:
                print(f"Embedding作成中: {i + 1}/{len(self.chunks)}")

            # Check cache first
            if self.cache_manager:
                cached_embedding = self.cache_manager.get_embedding(chunk)
                if cached_embedding:
                    embeddings.append(cached_embedding)
                    continue

            # Create new embedding
            embedding = self.embeddings_manager.create_embedding(chunk)
            embeddings.append(embedding)

            # Save to cache
            if self.cache_manager:
                self.cache_manager.save_embedding(chunk, embedding)

        # Create index
        self.index = self.embeddings_manager.create_index(embeddings)
        return self.index

    def retrieve_relevant_chunks(self, query, top_k=3):
        """Retrieve relevant chunks for a query.

        Args:
            query (str): Query text
            top_k (int): Number of chunks to retrieve

        Returns:
            list: List of relevant chunks with metadata
        """
        if not self.index:
            raise ValueError("No index created. Call create_index() first.")

        # Get query embedding
        if self.cache_manager:
            query_embedding = self.cache_manager.get_embedding(query)

        if not query_embedding:
            query_embedding = self.embeddings_manager.create_embedding(query)
            if self.cache_manager:
                self.cache_manager.save_embedding(query, query_embedding)

        # Search
        distances, indices = self.embeddings_manager.search(
            self.index, query_embedding, top_k
        )

        # Get relevant chunks
        relevant_chunks = []
        for i in range(top_k):
            if i < len(indices[0]):
                idx = indices[0][i]
                if idx < len(self.chunks):
                    relevant_chunks.append(
                        {
                            "content": self.chunks[idx],
                            "metadata": self.chunk_metadata[idx],
                            "distance": distances[0][i],
                        }
                    )

        return relevant_chunks

    def generate_response(self, query, chunks):
        """Generate response for a query using retrieved chunks.

        Args:
            query (str): Query text
            chunks (list): List of relevant chunks

        Returns:
            str: Generated response
        """
        # Check cache
        if self.cache_manager:
            chunks_info = [
                f"{chunk['metadata']['doc_name']}:{chunk['distance']:.4f}"
                for chunk in chunks
            ]
            cached_response = self.cache_manager.get_response(query, chunks_info)
            if cached_response:
                return cached_response

        # Build prompt
        prompt = f"""
質問: {query}

以下の情報をもとに質問に回答してください:
"""

        for i, chunk in enumerate(chunks):
            doc_name = chunk["metadata"]["doc_name"]
            chunk_idx = chunk["metadata"]["chunk_index"]
            total_chunks = chunk["metadata"]["total_chunks"]
            prompt += f"\n情報源 {i + 1}: {doc_name} (チャンク {chunk_idx + 1}/{total_chunks})\n"
            prompt += f"{chunk['content']}\n"

        prompt += "\n回答:"

        # Generate response
        response = self.client.chat.completions.create(
            model=self.config["completion_model"],
            messages=[
                {
                    "role": "system",
                    "content": "あなたは提供された情報に基づいて質問に回答するアシスタントです。提供された情報から回答できる場合のみ回答し、情報がない場合は「提供された情報からは回答できません」と伝えてください。",
                },
                {"role": "user", "content": prompt},
            ],
        )

        answer = response.choices[0].message.content

        # Save to cache
        if self.cache_manager:
            self.cache_manager.save_response(query, chunks_info, answer)

        return answer
