# pyenv 導入・利用ガイド

Pythonバージョン管理ツール「pyenv」の導入と基本的な利用方法について説明します。

## 概要

pyenvは複数のPythonバージョンを管理し、プロジェクトごとに異なるバージョンを簡単に切り替えることができるツールです。

## インストール手順

### Linux (Ubuntu/Debian)

```bash
# 依存パッケージのインストール
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev \
liblzma-dev python3-openssl git

# pyenvのインストール
curl https://pyenv.run | bash
```

### macOS

```bash
# Homebrewを使用
brew install pyenv
```

### Windows (WSL推奨)

Windows環境では WSL (Windows Subsystem for Linux) を使用し、Linux向けの手順でインストールすることを推奨します。

## 環境変数の設定

インストール後、以下を `~/.bashrc` または `~/.zshrc` に追加します：

```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

変更を反映：

```bash
source ~/.bashrc  # または source ~/.zshrc
```

## 基本的な使い方

### 利用可能なPythonバージョンの確認

```bash
pyenv install --list
```

### 特定のPythonバージョンのインストール

```bash
pyenv install 3.11.9
```

### グローバルPythonバージョンの設定

```bash
pyenv global 3.11.9
```

### プロジェクト固有のPythonバージョン設定

```bash
cd プロジェクトディレクトリ
pyenv local 3.11.9  # .python-versionファイルが作成される
```

### 現在のPythonバージョンの確認

```bash
pyenv version
```

### インストール済みのすべてのバージョンの確認

```bash
pyenv versions
```

## トラブルシューティング

### pyenvでインストールしたPythonが見つからない場合

```bash
pyenv rehash
```

### インストール中にエラーが発生する場合

依存関係が不足している可能性があります。各OSの依存パッケージをインストールしてから再試行してください。

## まとめ

pyenvを使うことで、システム全体に影響を与えることなく、複数のPythonバージョンを簡単に管理できます。Poetryと組み合わせることで、プロジェクトごとに独立した環境を構築できます。
