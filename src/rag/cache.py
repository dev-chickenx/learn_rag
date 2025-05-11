"""Cache management module for RAG system."""

import hashlib
import json
from pathlib import Path


class CacheManager:
    def __init__(self, config):
        """Initialize the cache manager.

        Args:
            config (dict): Configuration dictionary containing cache settings
        """
        self.config = config
        self.cache_dir = Path(config["cache_dir"])
        self.embedding_cache = {}
        self.response_cache = {}
        self._init_cache()

    def _init_cache(self):
        """Initialize cache directory and load existing caches."""
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(exist_ok=True)
            print(f"キャッシュディレクトリを作成しました: {self.cache_dir}")

        # Initialize cache files
        self.embeddings_cache_file = self.cache_dir / "embeddings_cache.json"
        self.response_cache_file = self.cache_dir / "response_cache.json"

        # Load existing caches
        self._load_embedding_cache()
        self._load_response_cache()

    def _load_embedding_cache(self):
        """Load embedding cache from file."""
        if self.embeddings_cache_file.exists():
            try:
                with open(self.embeddings_cache_file, "r", encoding="utf-8") as f:
                    self.embedding_cache = json.load(f)
                print(
                    f"エンベディングキャッシュを読み込みました: {len(self.embedding_cache)}件"
                )
            except json.JSONDecodeError:
                print(
                    "キャッシュファイルが破損しています。新しいキャッシュを作成します。"
                )
                self.embedding_cache = {}

    def _load_response_cache(self):
        """Load response cache from file."""
        if self.response_cache_file.exists():
            try:
                with open(self.response_cache_file, "r", encoding="utf-8") as f:
                    self.response_cache = json.load(f)
                print(f"応答キャッシュを読み込みました: {len(self.response_cache)}件")
            except json.JSONDecodeError:
                print(
                    "キャッシュファイルが破損しています。新しいキャッシュを作成します。"
                )
                self.response_cache = {}

    def get_hash(self, text):
        """Calculate hash value for text.

        Args:
            text (str): Text to calculate hash for

        Returns:
            str: MD5 hash of the text
        """
        return hashlib.md5(text.encode()).hexdigest()

    def get_embedding(self, text):
        """Get embedding from cache if available.

        Args:
            text (str): Text to get embedding for

        Returns:
            list or None: Cached embedding if available, None otherwise
        """
        text_hash = self.get_hash(text)
        return self.embedding_cache.get(text_hash)

    def save_embedding(self, text, embedding):
        """Save embedding to cache.

        Args:
            text (str): Text the embedding was created for
            embedding (list): Embedding vector to cache
        """
        text_hash = self.get_hash(text)
        self.embedding_cache[text_hash] = embedding
        self._save_embedding_cache()

    def _save_embedding_cache(self):
        """Save embedding cache to file."""
        with open(self.embeddings_cache_file, "w", encoding="utf-8") as f:
            json.dump(self.embedding_cache, f)

    def get_response(self, query, chunks_info):
        """Get response from cache if available.

        Args:
            query (str): Query text
            chunks_info (list): List of chunk information used for response

        Returns:
            str or None: Cached response if available, None otherwise
        """
        # Create cache key from query and chunks info
        cache_key = self.get_hash(query + "".join(chunks_info))
        return self.response_cache.get(cache_key)

    def save_response(self, query, chunks_info, response):
        """Save response to cache.

        Args:
            query (str): Query text
            chunks_info (list): List of chunk information used for response
            response (str): Response to cache
        """
        cache_key = self.get_hash(query + "".join(chunks_info))
        self.response_cache[cache_key] = response
        self._save_response_cache()

    def _save_response_cache(self):
        """Save response cache to file."""
        with open(self.response_cache_file, "w", encoding="utf-8") as f:
            json.dump(self.response_cache, f)
