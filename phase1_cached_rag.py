import glob
import hashlib
import json
import os
import time
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# .envファイルから環境変数を読み込む
load_dotenv()

# OpenAI APIクライアントの初期化
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# キャッシュディレクトリの設定
cache_dir = Path("cache")
if not cache_dir.exists():
    cache_dir.mkdir(exist_ok=True)

# エンベディングのキャッシュファイル
embeddings_cache_file = cache_dir / "embeddings_cache.json"
# クエリ応答のキャッシュファイル
response_cache_file = cache_dir / "response_cache.json"

# エンベディングキャッシュの読み込み
embedding_cache = {}
if embeddings_cache_file.exists():
    try:
        with open(embeddings_cache_file, "r", encoding="utf-8") as f:
            embedding_cache = json.load(f)
        print(f"エンベディングキャッシュを読み込みました: {len(embedding_cache)}件")
    except json.JSONDecodeError:
        print("キャッシュファイルが破損しています。新しいキャッシュを作成します。")
        embedding_cache = {}

# 応答キャッシュの読み込み
response_cache = {}
if response_cache_file.exists():
    try:
        with open(response_cache_file, "r", encoding="utf-8") as f:
            response_cache = json.load(f)
        print(f"応答キャッシュを読み込みました: {len(response_cache)}件")
    except json.JSONDecodeError:
        print("キャッシュファイルが破損しています。新しいキャッシュを作成します。")
        response_cache = {}


# テキストのハッシュ値を計算
def get_hash(text):
    return hashlib.md5(text.encode()).hexdigest()


# ドキュメントの読み込み
docs_path = Path("docs")
doc_files = glob.glob(str(docs_path / "*.md"))

docs = []
doc_contents = []
doc_hashes = []

# 各ファイルの内容を読み込む
for file_path in doc_files:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        content_hash = get_hash(content)
        doc_contents.append(content)
        docs.append(Path(file_path).name)
        doc_hashes.append(content_hash)

print(f"読み込んだドキュメント: {docs}")

# テキスト埋め込みの生成（キャッシュ利用）
embeddings = []
embedding_model = "text-embedding-3-small"
cache_hits = 0
cache_misses = 0

start_time = time.time()
for i, (content, doc_hash) in enumerate(zip(doc_contents, doc_hashes)):
    if doc_hash in embedding_cache:
        # キャッシュヒット
        embedding = embedding_cache[doc_hash]
        embeddings.append(embedding)
        cache_hits += 1
        print(f"Embedding キャッシュヒット: {docs[i]}")
    else:
        # キャッシュミス - APIで生成
        print(f"Embedding 作成中: {docs[i]}")
        response = client.embeddings.create(model=embedding_model, input=content)
        embedding = response.data[0].embedding
        embeddings.append(embedding)
        # キャッシュに保存
        embedding_cache[doc_hash] = embedding
        cache_misses += 1

# キャッシュの保存
with open(embeddings_cache_file, "w", encoding="utf-8") as f:
    json.dump(embedding_cache, f)

print(f"\n埋め込み完了: {len(embeddings)}件のドキュメントを処理しました")
print(f"キャッシュヒット: {cache_hits}件, キャッシュミス: {cache_misses}件")
print(f"処理時間: {time.time() - start_time:.2f}秒")

# numpy配列に変換
embeddings_np = np.array(embeddings).astype("float32")

# FAISSインデックスの作成
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)

# ベクトルをインデックスに追加
index.add(embeddings_np)
print(f"FAISSインデックスにベクトルを追加しました: {index.ntotal}件")


def retrieve_relevant_docs(query, top_k=2):
    """クエリに関連する文書を検索する"""
    # クエリのEmbedding作成
    start_time = time.time()
    query_hash = get_hash(query)

    if query_hash in embedding_cache:
        query_embedding = embedding_cache[query_hash]
        print(f"クエリ埋め込みをキャッシュから取得 ({time.time() - start_time:.2f}秒)")
    else:
        query_embedding_response = client.embeddings.create(
            model=embedding_model, input=query
        )
        query_embedding = query_embedding_response.data[0].embedding
        embedding_cache[query_hash] = query_embedding
        print(f"クエリ埋め込みを生成 ({time.time() - start_time:.2f}秒)")

        # キャッシュの保存
        with open(embeddings_cache_file, "w", encoding="utf-8") as f:
            json.dump(embedding_cache, f)

    query_embedding_np = np.array([query_embedding]).astype("float32")

    # 検索実行
    distances, indices = index.search(query_embedding_np, top_k)

    # 関連文書の取得
    relevant_docs = []
    for i in range(top_k):
        if i < len(indices[0]):
            idx = indices[0][i]
            if idx < len(docs):
                relevant_docs.append(
                    {
                        "name": docs[idx],
                        "content": doc_contents[idx],
                        "distance": distances[0][i],
                    }
                )

    return relevant_docs


def generate_rag_response(query, retrieved_docs):
    """検索結果を使用して回答を生成する（キャッシュ利用）"""
    # キャッシュキーの生成（クエリと検索結果の組み合わせに基づく）
    cache_key = get_hash(query + "".join([doc["name"] for doc in retrieved_docs]))

    # キャッシュチェック
    if cache_key in response_cache:
        print("応答キャッシュヒット！")
        return response_cache[cache_key]

    # プロンプトの構築
    prompt = f"""
質問: {query}

以下の情報をもとに質問に回答してください:
"""

    for i, doc in enumerate(retrieved_docs):
        prompt += f"\n情報源 {i + 1}: {doc['name']}\n"
        prompt += (
            f"{doc['content'][:1000]}...\n"  # 長いドキュメントは先頭1000文字のみ使用
        )

    prompt += "\n回答:"

    # Chat APIを使用して回答生成
    start_time = time.time()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
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
    response_cache[cache_key] = answer
    with open(response_cache_file, "w", encoding="utf-8") as f:
        json.dump(response_cache, f)

    return answer


# 対話ループ
print("\nRAGシステムに質問してください（終了するには 'exit' と入力）")
while True:
    query = input("\n質問: ")
    if query.lower() == "exit":
        break

    # キャッシュされた完全一致の応答をチェック
    query_hash = get_hash(query)
    if query_hash in response_cache:
        print("完全一致のクエリキャッシュヒット！")
        print(f"\n回答:\n{response_cache[query_hash]}")
        continue

    print("関連文書検索中...")
    start_time = time.time()
    relevant_docs = retrieve_relevant_docs(query)
    print(f"検索時間: {time.time() - start_time:.2f}秒")

    print(f"{len(relevant_docs)}件の関連文書が見つかりました")
    for i, doc in enumerate(relevant_docs):
        print(f"  文書 {i + 1}: {doc['name']} (距離: {doc['distance']:.2f})")

    print("\n回答生成中...")
    answer = generate_rag_response(query, relevant_docs)

    print(f"\n回答:\n{answer}")
