# RAG Sandbox

RAG（Retrieval-Augmented Generation）学習用のサンドボックスプロジェクト。

## 環境構築

```bash
# 1. クローン後、プロジェクトディレクトリに移動
cd rag-sandbox

# 2. Poetry環境を有効化
poetry env use python3.11
source $(poetry env info --path)/bin/activate

# 3. .envファイルを編集してAPIキーを設定
cp .env.example .env
# .envファイルを編集してOpenAI APIキーを設定

# 4. 依存関係の確認
python -c "import openai, faiss, tiktoken, dotenv; print('環境構築完了')"
```

## ディレクトリ構成

- `docs/`: 埋め込み対象のドキュメント
- `logs/`: 実験結果のログ
