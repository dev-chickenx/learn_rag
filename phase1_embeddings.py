import glob
import os
from pathlib import Path

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
