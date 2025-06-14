"""RAG実装の単体テスト（APIコールなし）。"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from dotenv import load_dotenv
from src.rag.rag import RAG
from src.rag.token_counter import TokenCounter


@pytest.fixture
def sample_config():
    """基本的な設定を提供するフィクスチャ"""
    return {
        "openai_api_key": "dummy-key",
        "completion_model": "gpt-3.5-turbo",
        "embedding_model": "text-embedding-3-small",
        "chunk_size": 384,
        "chunk_overlap": 50,
        "top_k": 3,
        "use_cache": True,
        "cache_dir": ".cache",
        "docs_dir": "docs",
    }


def test_init(sample_config):
    """RAGクラスの初期化テスト"""
    rag = RAG(sample_config)
    assert rag.config == sample_config
    assert rag.chunks == []
    assert rag.chunk_metadata == []
    assert rag.index is None


def test_split_into_chunks_no_headings(sample_config):
    """見出しのないテキストのチャンク分割テスト"""
    rag = RAG(sample_config)
    text = "これはテストテキストです。" * 50  # 適度な長さのテキスト

    chunks = rag.split_into_chunks(text)

    assert len(chunks) > 0
    for chunk in chunks:
        tokens = rag.completion_counter.count_tokens(chunk)
        assert 256 <= tokens <= 512, f"チャンクのトークン数が範囲外です: {tokens}"


def test_split_into_chunks_with_headings(sample_config):
    """見出しを含むテキストのチャンク分割テスト"""
    rag = RAG(sample_config)
    text = """# 見出し1
これは1つ目のセクションです。

## 見出し2
これは2つ目のセクションです。

### 見出し3
これは3つ目のセクションです。
"""
    chunks = rag.split_into_chunks(text)

    assert len(chunks) == 3
    assert chunks[0].startswith("# 見出し1")
    assert chunks[1].startswith("## 見出し2")
    assert chunks[2].startswith("### 見出し3")


@patch("glob.glob")
def test_load_documents(mock_glob, sample_config, tmp_path):
    """ドキュメント読み込みのテスト"""
    # テスト用のファイルを作成
    doc_path = tmp_path / "test.md"
    doc_path.write_text("# テスト\nこれはテストドキュメントです。")

    mock_glob.return_value = [str(doc_path)]

    rag = RAG(sample_config)
    chunks, metadata, num_docs = rag.load_documents()

    assert num_docs == 1
    assert len(chunks) > 0
    assert len(metadata) == len(chunks)
    assert all(isinstance(m, dict) for m in metadata)


@pytest.mark.parametrize(
    "text,expected_tokens",
    [
        ("短いテキスト", 7),
        ("これは少し長めのテキストです。", 13),
    ],
)
def test_token_counter(sample_config, text, expected_tokens):
    """トークンカウンターの動作テスト"""
    rag = RAG(sample_config)
    tokens = rag.completion_counter.count_tokens(text)
    assert tokens == expected_tokens
