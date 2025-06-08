"""Core RAG (Retrieval-Augmented Generation) implementation."""

import glob
import re
from pathlib import Path

from openai import OpenAI

from .cache import CacheManager
from .embeddings import EmbeddingsManager
from .token_counter import TokenCounter


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

        # Initialize token counters
        self.embedding_counter = TokenCounter("text-embedding-3-small")
        self.completion_counter = TokenCounter(config["completion_model"])

        # Initialize state
        self.chunks = []
        self.chunk_metadata = []
        self.index = None

    def split_into_chunks(self, text, target_tokens=384, overlap_tokens=50):
        """Split text into chunks based on token count.

        Args:
            text (str): Text to split
            target_tokens (int): Target number of tokens per chunk
            overlap_tokens (int): Number of tokens to overlap between chunks

        Returns:
            list: List of text chunks
        """
        # Use headings as primary split points
        headings = list(re.finditer(r"^#{1,6}\s+.*$", text, re.MULTILINE))
        sections = []

        if headings:
            # Split by headings
            for i in range(len(headings)):
                start = headings[i].start()
                if i < len(headings) - 1:
                    end = headings[i + 1].start()
                else:
                    end = len(text)
                sections.append(text[start:end])
        else:
            # If no headings found, treat entire text as one section
            sections = [text]

        # Split sections into token-based chunks
        chunks = []
        for section in sections:
            section_tokens = self.completion_counter.count_tokens(section)

            if section_tokens <= target_tokens:
                # If section is small enough, keep it as is
                chunks.append(section)
                continue

            # Split section into chunks
            current_pos = 0
            current_chunk = ""
            current_tokens = 0

            # Split on sentence boundaries when possible
            sentences = re.split(r"([.!?。]\s+)", section)

            for sentence in sentences:
                sentence_tokens = self.completion_counter.count_tokens(sentence)

                if current_tokens + sentence_tokens > target_tokens and current_chunk:
                    # Current chunk is full, save it
                    chunks.append(current_chunk)

                    # Start new chunk with overlap
                    if current_chunk and overlap_tokens > 0:
                        # Calculate overlap text
                        overlap_text = current_chunk[-overlap_tokens:]
                        current_chunk = overlap_text + sentence
                        current_tokens = self.completion_counter.count_tokens(
                            current_chunk
                        )
                    else:
                        current_chunk = sentence
                        current_tokens = sentence_tokens
                else:
                    # Add sentence to current chunk
                    current_chunk += sentence
                    current_tokens += sentence_tokens

            # Add the last chunk if it's not empty
            if current_chunk:
                chunks.append(current_chunk)

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
            dict: Generated response, token statistics, and source references
        """
        # Reset token counters for this generation
        self.completion_counter.reset_totals()

        # Check cache
        if self.cache_manager:
            chunks_info = [
                f"{chunk['metadata']['doc_name']}:{chunk['distance']:.4f}"
                for chunk in chunks
            ]
            cached_response = self.cache_manager.get_response(query, chunks_info)
            if cached_response:
                return {
                    "response": cached_response,
                    "token_stats": {"cached": True},
                    "sources": [],  # キャッシュからの応答は現状ソース情報を持っていない
                }

        # Build system prompt
        system_prompt = """あなたは提供された情報に基づいて質問に回答するアシスタントです。以下のガイドラインに従ってください：

1. 提供された情報のみを使用して回答を生成してください
2. 情報が不十分な場合は、その旨を正直に伝えてください
3. 回答の中で情報を引用する際は、[情報源X]の形式で引用してください
4. 確信が持てない場合は、その不確実性を明確に伝えてください
5. 回答は簡潔かつ正確を心がけてください
6. 回答の最後に、使用した情報源の一覧を箇条書きで記載してください"""

        self.completion_counter.add_to_total(system_prompt)

        # Build context prompt
        context_prompt = "以下の情報源を参考に回答を生成してください：\n\n"

        # Track sources for reference
        sources = []

        for i, chunk in enumerate(chunks, 1):
            doc_name = chunk["metadata"]["doc_name"]
            chunk_idx = chunk["metadata"]["chunk_index"]
            total_chunks = chunk["metadata"]["total_chunks"]
            similarity = 1 - chunk["distance"]  # Convert distance to similarity score

            # Store source information
            sources.append(
                {
                    "id": i,
                    "file": doc_name,
                    "chunk": chunk_idx + 1,
                    "total_chunks": total_chunks,
                    "similarity": similarity,
                    "content": chunk["content"].strip(),
                }
            )

            context_section = f"[情報源{i}] {doc_name} (チャンク {chunk_idx + 1}/{total_chunks}, 関連度: {similarity:.2%})\n"
            context_section += f"{chunk['content'].strip()}\n\n"
            context_prompt += context_section

            self.completion_counter.add_to_total(context_section)

        # Build user prompt
        user_prompt = f"質問: {query}\n\n{context_prompt}\n回答を生成してください。"
        self.completion_counter.add_to_total(user_prompt)

        # Generate response
        response = self.client.chat.completions.create(
            model=self.config.get("completion_model", "gpt-4"),  # デフォルトはgpt-4
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,  # Lower temperature for more focused responses
        )

        answer = response.choices[0].message.content
        self.completion_counter.add_to_total(answer, is_output=True)

        # Get token statistics
        token_stats = self.completion_counter.get_total_stats()
        token_stats["cached"] = False

        # Save to cache
        if self.cache_manager:
            self.cache_manager.save_response(query, chunks_info, answer)

        return {"response": answer, "token_stats": token_stats, "sources": sources}
