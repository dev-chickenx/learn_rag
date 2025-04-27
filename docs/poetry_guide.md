# Poetry 導入・利用ガイド

Pythonプロジェクト管理ツール「Poetry」の導入と基本的な利用方法について説明します。

## 概要

Poetryは、Pythonプロジェクトの依存関係管理、パッケージング、環境管理を一括して行うためのツールです。`requirements.txt`や`setup.py`の代わりに、標準的なプロジェクト構成と依存関係管理を提供します。

## インストール手順

### 公式インストーラー（推奨）

```bash
curl -sSf https://install.python-poetry.org | python3 -
```

### pipを使用する場合

```bash
pip install poetry
```

## 環境変数の設定

インストール後、以下を `~/.bashrc` または `~/.zshrc` に追加します：

```bash
export PATH="$HOME/.local/bin:$PATH"
```

変更を反映：

```bash
source ~/.bashrc  # または source ~/.zshrc
```

## Poetryの設定

### バージョン確認

```bash
poetry --version
```

### 設定の確認

```bash
poetry config --list
```

### 仮想環境の作成場所の設定

デフォルトでは `~/.cache/pypoetry/virtualenvs/` に作成されますが、プロジェクトディレクトリ内に作成することもできます：

```bash
# プロジェクト内に.venvディレクトリを作成
poetry config virtualenvs.in-project true
```

### pyenvを使用している場合の設定

```bash
# pyenvで設定されたPythonを優先的に使用
poetry config virtualenvs.prefer-active-python true
```

## 基本的な使い方

### 新規プロジェクトの作成

```bash
poetry new プロジェクト名
```

または既存ディレクトリで初期化：

```bash
cd プロジェクトディレクトリ
poetry init
```

非対話式で初期化する場合：

```bash
poetry init --name プロジェクト名 --description "説明" --author "著者名" --python ">=3.11,<3.12" -n
```

### パッケージの追加

```bash
# 通常の依存関係
poetry add パッケージ名

# バージョン指定
poetry add パッケージ名==1.2.3

# 開発用依存関係
poetry add --group dev pytest
```

### 依存関係のインストール

```bash
poetry install
```

開発用依存関係なしでインストール：

```bash
poetry install --without dev
```

### 仮想環境の利用

Python実行環境の指定：

```bash
poetry env use python3.11
```

仮想環境の情報確認：

```bash
poetry env info
```

仮想環境の有効化（Poetry 2.0以降）：

```bash
poetry env activate
```

以前の方法（すべてのPoetryバージョンで動作）：

```bash
source $(poetry env info --path)/bin/activate
```

### パッケージの実行

```bash
poetry run python スクリプト.py
```

### シェルの起動

```bash
poetry shell
```

## パッケージング

### ビルド

```bash
poetry build
```

### 公開

```bash
poetry publish
```

## トラブルシューティング

### 依存関係の解決に失敗する場合

```bash
# lockファイルを再生成
poetry lock --no-update
```

### パッケージのキャッシュをクリアする

```bash
poetry cache clear --all .
```

## まとめ

Poetryを使用することで、再現可能な環境構築、依存関係の適切な管理、簡単なパッケージング・公開が可能になります。pyenvと組み合わせることで、より柔軟なPython環境管理ができます。
