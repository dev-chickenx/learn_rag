"""Test suite for RAG pipeline."""

import os
from pathlib import Path

import pytest
import yaml
from dotenv import load_dotenv
from src.rag.rag import RAG


@pytest.fixture(scope="session")
def config():
    """Load configuration and environment variables."""
    load_dotenv()

    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Add API key from environment
    config["openai_api_key"] = os.getenv("OPENAI_API_KEY")
    if not config["openai_api_key"]:
        pytest.skip("OPENAI_API_KEY not set")

    return config


@pytest.fixture(scope="session")
def rag_instance(config):
    """Create and initialize RAG instance."""
    rag = RAG(config)

    # Load documents and create index
    chunks, metadata, num_docs = rag.load_documents()
    assert num_docs > 0, "No documents loaded"

    rag.create_index()
    assert rag.index is not None, "Index creation failed"

    return rag


@pytest.mark.parametrize(
    "query,expected_sources",
    [
        ("uvとpoetryの違いは何ですか？", ["tools_comparison.md", "uv_guide.md"]),
        (
            "pyenvのインストール方法を教えてください",
            ["pyenv_guide.md", "setup_guide.md"],
        ),
        ("パッケージの追加方法を教えてください", ["uv_guide.md", "poetry_guide.md"]),
    ],
)
def test_rag_pipeline(rag_instance, query, expected_sources):
    """Test the complete RAG pipeline."""
    # Retrieve relevant chunks
    chunks = rag_instance.retrieve_relevant_chunks(query)
    assert chunks, "No chunks retrieved"

    # Generate response
    result = rag_instance.generate_response(query, chunks)

    # Basic structure checks
    assert "response" in result
    assert "token_stats" in result
    assert "sources" in result

    # Response content checks
    assert len(result["response"]) > 0

    # Token statistics checks
    assert result["token_stats"]["total_tokens"] > 0
    assert result["token_stats"]["total_cost_usd"] > 0

    # Source checks
    assert len(result["sources"]) > 0
    found_sources = {source["file"] for source in result["sources"]}
    assert any(expected in found_sources for expected in expected_sources), (
        f"Expected to find at least one of {expected_sources} in {found_sources}"
    )

    # Print results for manual inspection
    print(f"\n=== Test Results for: {query} ===")
    print("\nResponse:")
    print(result["response"])
    print("\nToken Statistics:")
    print(f"Total Tokens: {result['token_stats']['total_tokens']}")
    print(f"Total Cost: ${result['token_stats']['total_cost_usd']:.6f}")
    print("\nSources Used:")
    for source in result["sources"]:
        print(
            f"- {source['file']} (Chunk {source['chunk']}/{source['total_chunks']}, "
            f"Similarity: {source['similarity']:.2%})"
        )


def test_token_counting(rag_instance):
    """Test token counting and cost estimation."""
    query = "短いテストクエリです"
    chunks = rag_instance.retrieve_relevant_chunks(query)
    result = rag_instance.generate_response(query, chunks)

    # Token statistics validation
    assert result["token_stats"]["total_tokens"] > 0
    assert result["token_stats"]["total_cost_usd"] > 0
    assert not result["token_stats"]["cached"]


def test_chunk_size(rag_instance):
    """Test chunk size constraints."""
    for chunk in rag_instance.chunks:
        tokens = rag_instance.completion_counter.count_tokens(chunk)
        assert 256 <= tokens <= 512, (
            f"Chunk size {tokens} is outside target range (256-512 tokens)"
        )


def test_source_metadata(rag_instance):
    """Test source metadata completeness."""
    query = "テストクエリ"
    chunks = rag_instance.retrieve_relevant_chunks(query)
    result = rag_instance.generate_response(query, chunks)

    for source in result["sources"]:
        assert "id" in source
        assert "file" in source
        assert "chunk" in source
        assert "total_chunks" in source
        assert "similarity" in source
        assert 0 <= source["similarity"] <= 1
