# uv 導入・利用ガイド

高速なPythonパッケージインストーラー「uv」の導入と基本的な利用方法について説明します。

## 概要

uvは、Astral（Rust製ツールチェーン開発企業）によって開発された、高速なPythonパッケージインストーラーおよび仮想環境マネージャーです。pipやその他のインストーラーよりも大幅に高速で、依存関係解決も効率的に行います。

## インストール手順

### Linuxおよび macOS

```bash
curl -sSf https://astral.sh/uv/install.sh | sh
```

### Windows

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Cargoを使用する場合（Rustユーザー向け）

```bash
cargo install uv
```

## 環境変数の設定

インストール後、以下を `~/.bashrc` または `~/.zshrc` に追加します：

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

変更を反映：

```bash
source ~/.bashrc  # または source ~/.zshrc
```

## 基本的な使い方

### バージョン確認

```bash
uv --version
```

### パッケージのインストール

```bash
# 単一パッケージ
uv pip install パッケージ名

# 複数パッケージ
uv pip install パッケージ名1 パッケージ名2

# バージョン指定
uv pip install パッケージ名==1.2.3

# editable mode
uv pip install -e .

# requirements.txtからインストール
uv pip install -r requirements.txt
```

### pipの更新

```bash
uv pip install -U pip
```

### 仮想環境の作成

```bash
uv venv
```

指定したパスに作成：

```bash
uv venv /path/to/venv
```

Pythonバージョンを指定して作成：

```bash
uv venv --python=3.11
```

### 仮想環境の有効化

```bash
source /path/to/venv/bin/activate
```

### システム全体へのインストール（グローバルインストール）

```bash
uv pip install --system パッケージ名
```

## Poetry との組み合わせ

Poetry環境内でuvを使用することで、パッケージインストールを高速化できます：

```bash
# Poetry環境を有効化
source $(poetry env info --path)/bin/activate

# uvを使用してインストール
uv pip install パッケージ名
```

## パフォーマンス比較

uvは従来のpipと比較して大幅に高速です。一般的な状況では：

- 大規模プロジェクト: 5〜10倍高速
- 依存関係解決: 最大30倍高速
- キャッシュ利用時: ほぼ瞬時

## トラブルシューティング

### uvコマンドが見つからない場合

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

### 特定のパッケージでエラーが発生する場合

```bash
# 詳細なログ表示
uv pip install -v パッケージ名
```

## まとめ

uvを使用することで、Pythonプロジェクトのセットアップ時間を大幅に短縮できます。特に大規模プロジェクトや複雑な依存関係を持つプロジェクトで効果を発揮します。PoetryやPyenvと組み合わせることで、さらに効率的な開発環境を構築できます。
