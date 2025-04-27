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

# サンプルクエリで検索
query = "パスワード長は？"
print(f"\nクエリ: '{query}'")

# クエリのEmbedding作成
query_embedding_response = client.embeddings.create(model=embedding_model, input=query)
query_embedding = query_embedding_response.data[0].embedding
query_embedding_np = np.array([query_embedding]).astype("float32")

# 検索実行 (top-k=3)
k = 3
distances, indices = index.search(query_embedding_np, k)

print("\n検索結果:")
for i in range(k):
    if i < len(indices[0]):
        idx = indices[0][i]
        if idx < len(docs):
            print(f"結果 {i + 1}: {docs[idx]} (距離: {distances[0][i]:.2f})")
            # ドキュメントの内容を一部表示（最初の150文字）
            content_preview = doc_contents[idx][:150] + "..."
            print(f"  プレビュー: {content_preview}")
