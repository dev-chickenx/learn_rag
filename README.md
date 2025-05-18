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

## 実装済み機能

### Phase 1: 埋め込み & Vector DB ハンズオン

#### テキスト埋め込み

- Markdownドキュメントの読み込みとチャンク分割
- OpenAI API（text-embedding-3-small）によるベクトル化
- キャッシュ機能による高速な再実行

#### FAISS 検索

- FAISSインデックスの作成と管理
- 類似度検索（L2距離）による関連文書の抽出
- キャッシュを活用した効率的な検索

#### 使用方法

```bash
# RAGシステムの起動
poetry run python scripts/rag_cli.py

# 対話モードで質問
質問 > パスワード長は？
```

## ディレクトリ構成

- `docs/`: 埋め込み対象のドキュメント
- `logs/`: 実験結果のログ
- `src/rag/`: コア実装
  - `rag.py`: RAGシステムのメインクラス
  - `embeddings.py`: テキスト埋め込み管理
  - `cache.py`: キャッシュ管理
- `scripts/`: 実行スクリプト
  - `rag_cli.py`: コマンドラインインターフェース
- `cache/`: キャッシュファイル
  - `embeddings_cache.json`: 埋め込みベクトルのキャッシュ
  - `response_cache.json`: 応答のキャッシュ
