import glob
import hashlib
import json
import os
import re
import time
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
        "cache_dir": "cache",
        "use_cache": True,
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


def init_cache(cache_dir):
    """キャッシュの初期化"""
    cache_dir_path = Path(cache_dir)
    if not cache_dir_path.exists():
        cache_dir_path.mkdir(exist_ok=True)
        print(f"キャッシュディレクトリを作成しました: {cache_dir}")

    # キャッシュファイルのパス
    embeddings_cache_file = cache_dir_path / "embeddings_cache.json"
    response_cache_file = cache_dir_path / "response_cache.json"

    # キャッシュの読み込み
    embedding_cache = {}
    if embeddings_cache_file.exists():
        try:
            with open(embeddings_cache_file, "r", encoding="utf-8") as f:
                embedding_cache = json.load(f)
            print(f"エンベディングキャッシュを読み込みました: {len(embedding_cache)}件")
        except json.JSONDecodeError:
            print("キャッシュファイルが破損しています。新しいキャッシュを作成します。")
            embedding_cache = {}

    response_cache = {}
    if response_cache_file.exists():
        try:
            with open(response_cache_file, "r", encoding="utf-8") as f:
                response_cache = json.load(f)
            print(f"応答キャッシュを読み込みました: {len(response_cache)}件")
        except json.JSONDecodeError:
            print("キャッシュファイルが破損しています。新しいキャッシュを作成します。")
            response_cache = {}

    cache = {
        "embedding": embedding_cache,
        "response": response_cache,
        "files": {
            "embedding": embeddings_cache_file,
            "response": response_cache_file,
        },
    }

    return cache


def get_hash(text):
    """テキストのハッシュ値を計算"""
    return hashlib.md5(text.encode()).hexdigest()


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
    chunk_hashes = []  # 各チャンクのハッシュ値

    # 各ファイルをチャンクに分割
    for file_path in doc_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            doc_name = Path(file_path).name

            # テキストをチャンクに分割
            doc_chunks = split_into_chunks(content, chunk_size, chunk_overlap)

            # チャンクとメタデータを保存
            for i, chunk in enumerate(doc_chunks):
                chunk_hash = get_hash(chunk)
                chunks.append(chunk)
                chunk_hashes.append(chunk_hash)
                chunk_metadata.append(
                    {
                        "doc_name": doc_name,
                        "chunk_index": i,
                        "total_chunks": len(doc_chunks),
                        "hash": chunk_hash,
                    }
                )

    print(f"読み込んだドキュメント: {len(doc_files)}件")
    print(f"生成されたチャンク数: {len(chunks)}件")

    return chunks, chunk_metadata, chunk_hashes, len(doc_files)


def create_embeddings(
    chunks, chunk_hashes, client, embedding_model, cache=None, use_cache=True
):
    """テキストチャンクの埋め込みを生成する（キャッシュ対応）"""
    embeddings = []
    cache_hits = 0
    cache_misses = 0

    start_time = time.time()

    for i, (chunk, chunk_hash) in enumerate(zip(chunks, chunk_hashes)):
        if use_cache and cache and chunk_hash in cache["embedding"]:
            # キャッシュヒット
            embedding = cache["embedding"][chunk_hash]
            embeddings.append(embedding)
            cache_hits += 1
            if i % 10 == 0:  # 進捗表示（10チャンクごと）
                print(f"Embedding キャッシュヒット: {i + 1}/{len(chunks)}")
        else:
            # キャッシュミス - APIで生成
            if i % 5 == 0:  # 進捗表示（5チャンクごと）
                print(f"Embedding 作成中: {i + 1}/{len(chunks)}")

            response = client.embeddings.create(model=embedding_model, input=chunk)
            embedding = response.data[0].embedding
            embeddings.append(embedding)
            cache_misses += 1

            # キャッシュに保存
            if use_cache and cache:
                cache["embedding"][chunk_hash] = embedding

    # キャッシュの保存
    if use_cache and cache and cache_misses > 0:
        with open(cache["files"]["embedding"], "w", encoding="utf-8") as f:
            json.dump(cache["embedding"], f)
            print(
                f"エンベディングキャッシュを更新しました: {len(cache['embedding'])}件"
            )

    processing_time = time.time() - start_time
    print(f"\n埋め込み完了: {len(embeddings)}件のチャンクを処理しました")
    print(f"キャッシュヒット: {cache_hits}件, キャッシュミス: {cache_misses}件")
    print(f"処理時間: {processing_time:.2f}秒")

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
    query,
    client,
    index,
    chunks,
    chunk_metadata,
    embedding_model,
    cache=None,
    use_cache=True,
    top_k=3,
):
    """クエリに関連するチャンクを検索する（キャッシュ対応）"""
    start_time = time.time()
    query_hash = get_hash(query)

    # クエリのEmbedding作成（キャッシュ対応）
    if use_cache and cache and query_hash in cache["embedding"]:
        query_embedding = cache["embedding"][query_hash]
        print(f"クエリ埋め込みをキャッシュから取得 ({time.time() - start_time:.2f}秒)")
    else:
        query_embedding_response = client.embeddings.create(
            model=embedding_model, input=query
        )
        query_embedding = query_embedding_response.data[0].embedding

        # キャッシュに保存
        if use_cache and cache:
            cache["embedding"][query_hash] = query_embedding
            with open(cache["files"]["embedding"], "w", encoding="utf-8") as f:
                json.dump(cache["embedding"], f)
            print(
                f"クエリ埋め込みを生成してキャッシュに保存 ({time.time() - start_time:.2f}秒)"
            )

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

    search_time = time.time() - start_time
    print(f"検索時間: {search_time:.2f}秒")
    return relevant_chunks


def generate_rag_response(
    query, retrieved_chunks, client, completion_model, cache=None, use_cache=True
):
    """検索結果を使用して回答を生成する（キャッシュ対応）"""
    # キャッシュキーの生成（クエリと検索結果の組み合わせに基づく）
    cache_key = None
    if use_cache and cache:
        # 検索結果のハッシュを計算
        chunks_info = [
            f"{chunk['metadata']['hash']}:{chunk['distance']:.4f}"
            for chunk in retrieved_chunks
        ]
        cache_key = get_hash(query + "".join(chunks_info))

        # キャッシュチェック
        if cache_key in cache["response"]:
            print("応答キャッシュヒット！")
            return cache["response"][cache_key]

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
    start_time = time.time()
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

    answer = response.choices[0].message.content
    print(f"回答生成時間: {time.time() - start_time:.2f}秒")

    # キャッシュに保存
    if use_cache and cache and cache_key:
        cache["response"][cache_key] = answer
        with open(cache["files"]["response"], "w", encoding="utf-8") as f:
            json.dump(cache["response"], f)

    return answer


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


def interactive_loop(client, index, chunks, chunk_metadata, config, cache=None):
    """対話ループを実行する"""
    print("\nRAGシステムに質問してください（終了するには 'exit' と入力）")
    while True:
        query = input("\n質問: ")
        if query.lower() == "exit":
            break

        # 完全一致のクエリに対するキャッシュチェック
        query_hash = get_hash(query)
        if config["use_cache"] and query_hash in cache["response"]:
            print("完全一致のクエリキャッシュヒット！")
            print(f"\n回答:\n{cache['response'][query_hash]}")
            continue

        print("関連文書検索中...")
        relevant_chunks = retrieve_relevant_chunks(
            query,
            client,
            index,
            chunks,
            chunk_metadata,
            config["embedding_model"],
            cache,
            config["use_cache"],
            config["top_k"],
        )

        display_chunks_preview(relevant_chunks)

        print("\n回答生成中...")
        answer = generate_rag_response(
            query,
            relevant_chunks,
            client,
            config["completion_model"],
            cache,
            config["use_cache"],
        )

        print(f"\n回答:\n{answer}")


def main():
    """メイン処理"""
    start_time = time.time()

    # 設定の読み込み
    config = load_config()

    # キャッシュの初期化
    cache = init_cache(config["cache_dir"]) if config["use_cache"] else None

    # OpenAIクライアントの初期化
    client = init_openai_client(config["openai_api_key"])

    # ドキュメントの読み込みとチャンク分割
    chunks, chunk_metadata, chunk_hashes, _ = load_documents(
        config["docs_dir"], config["chunk_size"], config["chunk_overlap"]
    )

    # 埋め込みの生成（キャッシュ対応）
    embeddings = create_embeddings(
        chunks,
        chunk_hashes,
        client,
        config["embedding_model"],
        cache,
        config["use_cache"],
    )

    # FAISSインデックスの作成
    index = create_faiss_index(embeddings)

    setup_time = time.time() - start_time
    print(f"\n初期化完了! 設定時間: {setup_time:.2f}秒")

    # 対話ループの実行
    interactive_loop(client, index, chunks, chunk_metadata, config, cache)


if __name__ == "__main__":
    main()
