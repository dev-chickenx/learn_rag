import glob
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# .envファイルから環境変数を読み込む
load_dotenv()

# OpenAI APIクライアントの初期化
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ドキュメントの読み込み
docs_path = Path("docs")
doc_files = glob.glob(str(docs_path / "*.md"))

docs = []
doc_contents = []

# 各ファイルの内容を読み込む
for file_path in doc_files:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        doc_contents.append(content)
        docs.append(Path(file_path).name)

print(f"読み込んだドキュメント: {docs}")

# テキスト埋め込みの生成
embeddings = []
embedding_model = "text-embedding-3-small"

for i, content in enumerate(doc_contents):
    print(f"Embedding作成中: {docs[i]}")
    response = client.embeddings.create(model=embedding_model, input=content)
    embedding = response.data[0].embedding
    embeddings.append(embedding)
    print(f"  次元数: {len(embedding)}")

print(f"\n埋め込み完了: {len(embeddings)}件のドキュメントを処理しました")

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
    query_embedding_response = client.embeddings.create(
        model=embedding_model, input=query
    )
    query_embedding = query_embedding_response.data[0].embedding
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
    """検索結果を使用して回答を生成する"""
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

    return response.choices[0].message.content


# 対話ループ
print("\nRAGシステムに質問してください（終了するには 'exit' と入力）")
while True:
    query = input("\n質問: ")
    if query.lower() == "exit":
        break

    print("関連文書検索中...")
    relevant_docs = retrieve_relevant_docs(query)

    print(f"{len(relevant_docs)}件の関連文書が見つかりました")
    for i, doc in enumerate(relevant_docs):
        print(f"  文書 {i + 1}: {doc['name']} (距離: {doc['distance']:.2f})")

    print("\n回答生成中...")
    answer = generate_rag_response(query, relevant_docs)

    print(f"\n回答:\n{answer}")
