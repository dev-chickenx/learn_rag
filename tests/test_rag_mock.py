"""RAG実装のモックテスト（APIコールをモック化）。

注意: このテストを実行するには、以下のいずれかの方法で環境変数を設定する必要があります：

1. 実行時に環境変数を設定:
   ```bash
   OPENAI_API_KEY=your-api-key-here pytest tests/test_rag_mock.py
   ```

2. .envファイルを使用:
   ```bash
   # .envファイルに以下を追加
   OPENAI_API_KEY=your-api-key-here
   ```

3. 環境変数を直接設定:
   ```bash
   export OPENAI_API_KEY=your-api-key-here
   pytest tests/test_rag_mock.py
   ```

環境変数が設定されていない場合、テストはスキップされます。
"""

import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml
from dotenv import load_dotenv
from src.rag.rag import RAG


@pytest.fixture
def sample_config():
    """基本的な設定を提供するフィクスチャ"""
    # 環境変数からAPIキーを取得
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEYが設定されていません。環境変数を設定してください。")

    return {
        "openai_api_key": api_key,
        "completion_model": "gpt-3.5-turbo",
        "embedding_model": "text-embedding-3-small",
        "chunk_size": 384,
        "chunk_overlap": 50,
        "top_k": 3,
        "use_cache": True,
        "cache_dir": ".cache",
        "docs_dir": "docs",
    }


@pytest.fixture
def mock_openai():
    """OpenAI APIのモックを提供するフィクスチャ"""
    with patch("openai.OpenAI") as mock:
        # Embedding responseのモック
        mock.return_value.embeddings.create.return_value.data = [
            {"embedding": [0.1] * 1536}
        ]

        # Chat completion responseのモック
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="モックされた回答"))]
        mock.return_value.chat.completions.create.return_value = mock_response

        yield mock


def test_generate_response_structure(sample_config, mock_openai):
    """レスポンス生成の構造テスト（APIコールをモック化）"""
    rag = RAG(sample_config)

    # テスト用のチャンクとメタデータを準備
    chunks = [
        {
            "content": "テストコンテンツ",
            "metadata": {"doc_name": "test.md", "chunk_index": 0, "total_chunks": 1},
            "distance": 0.1,
        }
    ]

    result = rag.generate_response("テストクエリ", chunks)

    # レスポンスの構造を検証
    assert "response" in result
    assert "token_stats" in result
    assert "sources" in result
    assert isinstance(result["token_stats"], dict)
    assert isinstance(result["sources"], list)


def test_cache_hit(sample_config):
    """キャッシュヒット時の動作テスト（APIコールなし）"""
    rag = RAG(sample_config)

    # キャッシュマネージャーをモック化
    rag.cache_manager.get_response = Mock(return_value="キャッシュされた回答")

    result = rag.generate_response("テストクエリ", [])

    assert result["response"] == "キャッシュされた回答"
    assert result["token_stats"]["cached"] is True


def test_embedding_generation(sample_config, mock_openai):
    """埋め込みベクトル生成のテスト（APIコールをモック化）"""
    rag = RAG(sample_config)
    text = "テストテキスト"

    embedding = rag.generate_embedding(text)

    assert isinstance(embedding, list)
    assert len(embedding) == 1536  # text-embedding-3-smallの次元数
    assert all(isinstance(x, float) for x in embedding)


def test_retrieve_relevant_chunks(sample_config, mock_openai):
    """関連チャンクの取得テスト（APIコールをモック化）"""
    rag = RAG(sample_config)

    # テスト用のチャンクを準備
    rag.chunks = ["テストチャンク1", "テストチャンク2"]
    rag.chunk_metadata = [
        {"doc_name": "test1.md", "chunk_index": 0},
        {"doc_name": "test2.md", "chunk_index": 0},
    ]

    # インデックスをモック化
    rag.index = Mock()
    rag.index.search.return_value = [{"id": 0, "score": 0.9}, {"id": 1, "score": 0.7}]

    chunks = rag.retrieve_relevant_chunks("テストクエリ")

    assert len(chunks) == 2
    assert all(isinstance(chunk, dict) for chunk in chunks)
    assert all("content" in chunk for chunk in chunks)
    assert all("metadata" in chunk for chunk in chunks)
    assert all("distance" in chunk for chunk in chunks)
