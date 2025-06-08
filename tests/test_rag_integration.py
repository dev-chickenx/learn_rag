"""Integration tests for RAG implementation.

Note: これらのテストはOpenAI APIを実際に呼び出すため、コストが発生します。
実行する場合は、環境変数 RUN_INTEGRATION_TESTS=1 を設定してください。
"""

import os
from pathlib import Path

import pytest
import yaml
from dotenv import load_dotenv
from src.rag.rag import RAG

# 統合テストを実行するかどうかをチェック
RUN_INTEGRATION_TESTS = os.getenv("RUN_INTEGRATION_TESTS") == "1"
if not RUN_INTEGRATION_TESTS:
    pytest.skip(
        "統合テストをスキップします。実行する場合は RUN_INTEGRATION_TESTS=1 を設定してください。",
        allow_module_level=True,
    )


@pytest.fixture(scope="session")
def config():
    """設定を読み込むフィクスチャ"""
    load_dotenv()

    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["openai_api_key"] = os.getenv("OPENAI_API_KEY")
    if not config["openai_api_key"]:
        pytest.skip("OPENAI_API_KEY not set")

    return config


@pytest.fixture(scope="session")
def rag_instance(config):
    """RAGインスタンスを準備するフィクスチャ"""
    rag = RAG(config)

    # ドキュメントを読み込み、インデックスを作成
    print("\n=== ドキュメントの読み込みとインデックス作成 ===")
    chunks, metadata, num_docs = rag.load_documents()
    assert num_docs > 0, "ドキュメントが読み込めません"

    rag.create_index()
    assert rag.index is not None, "インデックスの作成に失敗しました"

    return rag


@pytest.mark.integration
@pytest.mark.parametrize(
    "query,expected_keywords",
    [
        ("uvとpoetryの違いは何ですか？", ["uv", "poetry", "パッケージ管理"]),
        ("pyenvのインストール方法を教えてください", ["pyenv", "インストール", "設定"]),
    ],
)
def test_end_to_end(rag_instance, query, expected_keywords, capsys):
    """エンドツーエンドのテスト

    Note: このテストは実際にAPIを呼び出すため、コストが発生します。
    """
    print(f"\n=== テスト実行: {query} ===")

    # 関連チャンクの取得
    chunks = rag_instance.retrieve_relevant_chunks(query)
    assert chunks, "関連チャンクが見つかりません"

    # 回答の生成
    result = rag_instance.generate_response(query, chunks)

    # 基本的な構造の確認
    assert "response" in result
    assert "token_stats" in result
    assert "sources" in result

    # 回答の内容確認
    response = result["response"]
    assert any(keyword in response for keyword in expected_keywords), (
        f"回答に期待されるキーワードが含まれていません: {expected_keywords}"
    )

    # トークン統計の表示
    print("\n=== コスト統計 ===")
    print(f"総トークン数: {result['token_stats']['total_tokens']}")
    print(f"総コスト: ${result['token_stats']['total_cost_usd']:.6f}")

    # 使用された情報源の表示
    print("\n=== 使用された情報源 ===")
    for source in result["sources"]:
        print(
            f"- {source['file']} (チャンク {source['chunk']}/{source['total_chunks']}, "
            f"関連度: {source['similarity']:.2%})"
        )


@pytest.mark.integration
def test_token_limits(rag_instance):
    """トークン数の制限とコスト計測のテスト"""
    for chunk in rag_instance.chunks:
        tokens = rag_instance.completion_counter.count_tokens(chunk)
        assert 256 <= tokens <= 512, (
            f"チャンクのトークン数が目標範囲外です: {tokens} tokens"
        )
