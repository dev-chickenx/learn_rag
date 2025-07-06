#!/usr/bin/env python3
"""Command-line interface for the RAG system."""

import argparse
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from src.rag.cost_manager import CostManager
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
        "cache_dir": ".cache",
        "use_cache": True,
        # Cost management defaults
        "cost_limit_per_query": 0.10,
        "cost_limit_session": 1.00,
        "cost_warning_threshold": 0.05,
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

    # Initialize cost manager
    cost_manager = CostManager(config)

    # Load documents
    print("ドキュメントを読み込んでいます...")
    rag.load_documents()

    # Create index
    print("インデックスを作成しています...")
    rag.create_index()

    # Interactive loop
    print("\n質問を入力してください:")
    print("  - 終了: 'q' または 'quit'")
    print("  - セッション情報: 'cost' または 'summary'")
    print("  - コスト確認なし: 'skip:' で始まる質問")

    while True:
        query = input("\n質問 > ").strip()

        if query.lower() in ["q", "quit", "exit"]:
            # Show session summary before exit
            cost_manager.print_session_summary()
            break

        if query.lower() in ["cost", "summary", "コスト", "サマリー"]:
            cost_manager.print_session_summary()
            continue

        if not query:
            continue

        # Check if cost check should be skipped
        skip_cost_check = False
        if query.lower().startswith("skip:"):
            skip_cost_check = True
            query = query[5:].strip()  # Remove "skip:" prefix
            if not query:
                print("質問が空です。")
                continue

        # Retrieve relevant chunks
        chunks = rag.retrieve_relevant_chunks(query)

        # Cost estimation and confirmation (unless skipped)
        if not skip_cost_check:
            cost_breakdown = cost_manager.estimate_query_cost(query, chunks)
            estimated_cost = cost_breakdown["total_cost"]

            # Check cost limits
            should_proceed, warning_message = cost_manager.check_cost_limits(
                estimated_cost
            )

            if not should_proceed:
                print(warning_message)
                if not cost_manager.prompt_user_confirmation(
                    estimated_cost, cost_breakdown
                ):
                    print("クエリがキャンセルされました。")
                    continue
            elif warning_message:  # Warning threshold exceeded
                print(f"\n{warning_message}")

        # Generate response
        response = rag.generate_response(query, chunks)

        # Print response
        print("\n回答:")
        print(response["response"])

        # Print cost information
        token_stats = response.get("token_stats", {})
        if "total_cost_usd" in token_stats and not skip_cost_check:
            actual_cost = token_stats["total_cost_usd"]
            cost_manager.record_actual_cost(actual_cost)
            print(f"\n💰 実際のコスト: ${actual_cost:.6f}")
            print(f"📊 セッション累計: ${cost_manager.session_cost:.6f}")

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
