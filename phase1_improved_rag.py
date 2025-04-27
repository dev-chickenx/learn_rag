import glob
import os
import re
from pathlib import Path

import faiss
import numpy as np
import yaml
from dotenv import load_dotenv
from openai import OpenAI


def load_config():
    """設定を読み込む（.envとconfig.yamlから）"""
    # .envからシークレットを読み込む
    load_dotenv()

    # APIキーの確認（必須）
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEYが設定されていません。.envファイルを確認してください。"
        )

    # デフォルト設定
    default_config = {
        "embedding_model": "text-embedding-3-small",
        "completion_model": "gpt-3.5-turbo",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "top_k": 3,
        "docs_dir": "docs",
    }

    # config.yamlファイルがあれば読み込む
    config_path = Path("config.yaml")
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config and isinstance(yaml_config, dict):
                    # デフォルト設定をYAML設定で上書き
                    default_config.update(yaml_config)
                    print(f"設定を{config_path}から読み込みました")
        except Exception as e:
            print(f"警告: 設定ファイルの読み込みに失敗しました: {e}")
            print("デフォルト設定を使用します")
    else:
        print(f"設定ファイル{config_path}が見つかりません。デフォルト設定を使用します")
        # サンプル設定ファイルの作成（初回実行時のガイド）
        try:
            with open(f"{config_path}.sample", "w", encoding="utf-8") as f:
                yaml.dump(default_config, f, default_flow_style=False)
            print(f"サンプル設定ファイル{config_path}.sampleを作成しました")
        except Exception as e:
            print(f"サンプル設定ファイルの作成に失敗しました: {e}")

    # 最終的な設定（APIキーを追加）
    config = default_config.copy()
    config["openai_api_key"] = api_key

    # 設定内容の表示
    print("現在の設定:")
    for key, value in config.items():
        if key != "openai_api_key":  # APIキーは表示しない
            print(f"  {key}: {value}")
    print()

    return config


def init_openai_client(api_key):
    """OpenAI APIクライアントを初期化する"""
    return OpenAI(api_key=api_key)


def split_into_chunks(text, chunk_size=500, overlap=50):
    """テキストをチャンクに分割する"""
    # 見出し（#で始まる行）を分割点として使用
    headings = list(re.finditer(r"^#{1,6}\s+.*$", text, re.MULTILINE))
    sections = []

    if not headings:
        # 見出しがない場合は単純に文字数で分割
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            # チャンクの終わりが単語の途中である場合、最後の空白まで戻る
            if end < len(text) and not text[end].isspace():
                end = text.rfind(" ", start, end)
                if end == -1:  # 空白が見つからない場合
                    end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - overlap if end > overlap else 0
        return chunks

    # 見出しごとにセクションを分割
    for i in range(len(headings)):
        start = headings[i].start()
        if i < len(headings) - 1:
            end = headings[i + 1].start()
        else:
            end = len(text)

        section_text = text[start:end]
        sections.append(section_text)

    # 各セクションを適切なサイズのチャンクに分割
    chunks = []
    for section in sections:
        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            # 長いセクションはさらに分割
            section_chunks = []
            start = 0
            while start < len(section):
                end = min(start + chunk_size, len(section))
                # チャンクの終わりが単語の途中である場合、最後の空白まで戻る
                if end < len(section) and not section[end].isspace():
                    end = section.rfind(" ", start, end)
                    if end == -1:  # 空白が見つからない場合
                        end = min(start + chunk_size, len(section))
                section_chunks.append(section[start:end])
                start = end - overlap if end > overlap else 0
            chunks.extend(section_chunks)

    return chunks


def load_documents(docs_dir, chunk_size, chunk_overlap):
    """ドキュメントを読み込み、チャンクに分割する"""
    path = Path(docs_dir)
    doc_files = glob.glob(str(path / "*.md"))

    chunks = []
    chunk_metadata = []  # 各チャンクのメタデータ（出典、位置など）

    # 各ファイルをチャンクに分割
    for file_path in doc_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            doc_name = Path(file_path).name

            # テキストをチャンクに分割
            doc_chunks = split_into_chunks(content, chunk_size, chunk_overlap)

            # チャンクとメタデータを保存
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

    return chunks, chunk_metadata, len(doc_files)


def create_embeddings(chunks, client, embedding_model):
    """テキストチャンクの埋め込みを生成する"""
    embeddings = []

    for i, chunk in enumerate(chunks):
        if i % 5 == 0:  # 進捗表示（5チャンクごと）
            print(f"Embedding作成中: {i + 1}/{len(chunks)}")

        response = client.embeddings.create(model=embedding_model, input=chunk)
        embedding = response.data[0].embedding
        embeddings.append(embedding)

    print(f"\n埋め込み完了: {len(embeddings)}件のチャンクを処理しました")

    return embeddings


def create_faiss_index(embeddings):
    """FAISSインデックスを作成する"""
    # numpy配列に変換
    embeddings_np = np.array(embeddings).astype("float32")

    # FAISSインデックスの作成
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)

    # ベクトルをインデックスに追加
    index.add(embeddings_np)
    print(f"FAISSインデックスにベクトルを追加しました: {index.ntotal}件")

    return index


def retrieve_relevant_chunks(
    query, client, index, chunks, chunk_metadata, embedding_model, top_k=3
):
    """クエリに関連するチャンクを検索する"""
    # クエリのEmbedding作成
    query_embedding_response = client.embeddings.create(
        model=embedding_model, input=query
    )
    query_embedding = query_embedding_response.data[0].embedding
    query_embedding_np = np.array([query_embedding]).astype("float32")

    # 検索実行
    distances, indices = index.search(query_embedding_np, top_k)

    # 関連チャンクの取得
    relevant_chunks = []
    for i in range(top_k):
        if i < len(indices[0]):
            idx = indices[0][i]
            if idx < len(chunks):
                relevant_chunks.append(
                    {
                        "content": chunks[idx],
                        "metadata": chunk_metadata[idx],
                        "distance": distances[0][i],
                    }
                )

    return relevant_chunks


def generate_rag_response(query, retrieved_chunks, client, completion_model):
    """検索結果を使用して回答を生成する"""
    # プロンプトの構築
    prompt = f"""
質問: {query}

以下の情報をもとに質問に回答してください:
"""

    for i, chunk in enumerate(retrieved_chunks):
        doc_name = chunk["metadata"]["doc_name"]
        chunk_idx = chunk["metadata"]["chunk_index"]
        total_chunks = chunk["metadata"]["total_chunks"]
        prompt += (
            f"\n情報源 {i + 1}: {doc_name} (チャンク {chunk_idx + 1}/{total_chunks})\n"
        )
        prompt += f"{chunk['content']}\n"

    prompt += "\n回答:"

    # Chat APIを使用して回答生成
    response = client.chat.completions.create(
        model=completion_model,
        messages=[
            {
                "role": "system",
                "content": "あなたは提供された情報に基づいて質問に回答するアシスタントです。提供された情報から回答できる場合のみ回答し、情報がない場合は「提供された情報からは回答できません」と伝えてください。",
            },
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content


def display_chunks_preview(relevant_chunks):
    """検索されたチャンクのプレビューを表示する"""
    print(f"{len(relevant_chunks)}件の関連チャンクが見つかりました")
    for i, chunk in enumerate(relevant_chunks):
        doc_name = chunk["metadata"]["doc_name"]
        chunk_idx = chunk["metadata"]["chunk_index"]
        print(
            f"  チャンク {i + 1}: {doc_name} (チャンク {chunk_idx + 1}, 距離: {chunk['distance']:.2f})"
        )
        # チャンクの内容を一部表示（最初の100文字）
        preview = chunk["content"][:100].replace("\n", " ") + "..."
        print(f"    プレビュー: {preview}")


def interactive_loop(client, index, chunks, chunk_metadata, config):
    """対話ループを実行する"""
    print("\nRAGシステムに質問してください（終了するには 'exit' と入力）")
    while True:
        query = input("\n質問: ")
        if query.lower() == "exit":
            break

        print("関連文書検索中...")
        relevant_chunks = retrieve_relevant_chunks(
            query,
            client,
            index,
            chunks,
            chunk_metadata,
            config["embedding_model"],
            config["top_k"],
        )

        display_chunks_preview(relevant_chunks)

        print("\n回答生成中...")
        answer = generate_rag_response(
            query, relevant_chunks, client, config["completion_model"]
        )

        print(f"\n回答:\n{answer}")


def main():
    """メイン処理"""
    # 設定の読み込み
    config = load_config()

    # OpenAIクライアントの初期化
    client = init_openai_client(config["openai_api_key"])

    # ドキュメントの読み込みとチャンク分割
    chunks, chunk_metadata, _ = load_documents(
        config["docs_dir"], config["chunk_size"], config["chunk_overlap"]
    )

    # 埋め込みの生成
    embeddings = create_embeddings(chunks, client, config["embedding_model"])

    # FAISSインデックスの作成
    index = create_faiss_index(embeddings)

    # 対話ループの実行
    interactive_loop(client, index, chunks, chunk_metadata, config)


if __name__ == "__main__":
    main()
