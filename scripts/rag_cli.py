"""Command-line interface for the RAG system."""

import argparse
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from src.rag.rag import RAG


def load_config(config_path):
    """Load configuration from YAML file.

    Args:
        config_path (str): Path to config file

    Returns:
        dict: Configuration dictionary
    """
    # Load environment variables
    load_dotenv()

    # API key check (required)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEYが設定されていません。.envファイルを確認してください。"
        )

    # Default configuration
    default_config = {
        "embedding_model": "text-embedding-3-small",
        "completion_model": "gpt-3.5-turbo",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "top_k": 3,
        "docs_dir": "docs",
        "cache_dir": "cache",
        "use_cache": True,
    }

    # Load YAML config if exists
    config_path = Path(config_path)
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config and isinstance(yaml_config, dict):
                    # Update default config with YAML settings
                    default_config.update(yaml_config)
                    print(f"設定を{config_path}から読み込みました")
        except Exception as e:
            print(f"警告: 設定ファイルの読み込みに失敗しました: {e}")
            print("デフォルト設定を使用します")
    else:
        print(f"設定ファイル{config_path}が見つかりません。デフォルト設定を使用します")
        # Create sample config file (guide for first run)
        try:
            with open(f"{config_path}.sample", "w", encoding="utf-8") as f:
                yaml.dump(default_config, f, default_flow_style=False)
            print(f"サンプル設定ファイル{config_path}.sampleを作成しました")
        except Exception as e:
            print(f"サンプル設定ファイルの作成に失敗しました: {e}")

    # Final configuration (add API key)
    config = default_config.copy()
    config["openai_api_key"] = api_key

    # Display current settings
    print("現在の設定:")
    for key, value in config.items():
        if key != "openai_api_key":  # Don't display API key
            print(f"  {key}: {value}")
    print()

    return config


def main():
    """Main entry point for the RAG CLI."""
    parser = argparse.ArgumentParser(description="RAG system CLI")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Initialize RAG
    rag = RAG(config)

    # Load documents
    print("ドキュメントを読み込んでいます...")
    rag.load_documents()

    # Create index
    print("インデックスを作成しています...")
    rag.create_index()

    # Interactive loop
    print("\n質問を入力してください（終了するには 'q' または 'quit' と入力）:")
    while True:
        query = input("\n質問 > ").strip()

        if query.lower() in ["q", "quit", "exit"]:
            break

        if not query:
            continue

        # Retrieve relevant chunks
        chunks = rag.retrieve_relevant_chunks(query)

        # Generate response
        response = rag.generate_response(query, chunks)

        # Print response
        print("\n回答:")
        print(response)

        # Print sources
        print("\n情報源:")
        for chunk in chunks:
            doc_name = chunk["metadata"]["doc_name"]
            chunk_idx = chunk["metadata"]["chunk_index"]
            total_chunks = chunk["metadata"]["total_chunks"]
            distance = chunk["distance"]
            print(
                f"- {doc_name} (チャンク {chunk_idx + 1}/{total_chunks}, 類似度: {1 - distance:.4f})"
            )


if __name__ == "__main__":
    main()
