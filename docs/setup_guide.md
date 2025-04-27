# Python プロジェクト環境構築ガイド

pyenv, Poetry, uv を組み合わせた最新のPython開発環境セットアップガイド（2025年版）です。

## 概要

この環境構築では以下のツールを組み合わせます：

- **pyenv**: 複数のPythonバージョンを管理
- **Poetry**: プロジェクトの依存関係と仮想環境を管理
- **uv**: 高速なパッケージインストーラー

これらを組み合わせることで、再現性の高い、分離された、高速な開発環境を構築できます。

## 前提条件の確認

現在の環境を確認します：

```bash
# Python が既にインストールされているか確認
python --version || python3 --version

# pyenv が既にインストールされているか確認
command -v pyenv || echo "pyenv がインストールされていません"

# Poetry が既にインストールされているか確認
command -v poetry || echo "Poetry がインストールされていません"

# uv が既にインストールされているか確認
command -v uv || echo "uv がインストールされていません"
```

## 1. pyenv のインストール

### Linux / macOS

```bash
# 依存パッケージのインストール（Ubuntu/Debian の場合）
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev \
liblzma-dev python3-openssl git

# pyenv のインストール
curl https://pyenv.run | bash

# シェル設定ファイルに追加
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

# 設定を反映
source ~/.bashrc
```

### Windows (WSL)

WSL環境でLinux向けの手順を実行してください。

## 2. 適切なPythonバージョンのインストール

```bash
# 利用可能なバージョンの確認
pyenv install --list | grep "3.11"

# 最新の安定バージョンをインストール
pyenv install 3.11.9

# デフォルトとして設定（オプション）
pyenv global 3.11.9

# インストールの確認
python --version
```

## 3. Poetry のインストール

```bash
# Poetry のインストール
curl -sSf https://install.python-poetry.org | python3 -

# PATH設定（必要な場合）
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# バージョン確認
poetry --version

# 設定
poetry config virtualenvs.prefer-active-python true  # pyenvと連携するための設定
```

## 4. uv のインストール

```bash
# uv のインストール
curl -sSf https://astral.sh/uv/install.sh | sh

# PATH設定（必要な場合）
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# バージョン確認
uv --version
```

## 5. プロジェクト作成と環境セットアップ

```bash
# プロジェクトディレクトリ作成
mkdir -p プロジェクト名
cd プロジェクト名

# プロジェクト用のPythonバージョン指定
pyenv local 3.11.9  # .python-versionファイルが作成される

# Poetryプロジェクト初期化
poetry init --name プロジェクト名 --description "プロジェクト説明" --author "著者名" --python ">=3.11,<3.12" -n

# 依存パッケージの追加
poetry add パッケージ名1 パッケージ名2
poetry add --group dev pytest black

# 仮想環境のパスを確認
poetry env info --path

# 仮想環境を有効化
source $(poetry env info --path)/bin/activate

# uvを使ったパッケージ高速インストール（Poetryの仮想環境内）
uv pip install -U pip  # pipの更新
```

## 6. 基本的なプロジェクト構造セットアップ

```bash
# 標準的なディレクトリ構造を作成
mkdir -p src/プロジェクト名 tests docs

# .gitignoreの作成
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# 環境
.env
.venv
env/
venv/
ENV/
.python-version

# Poetry
poetry.lock

# VSCode
.vscode/

# Jupyter
.ipynb_checkpoints
EOF

# READMEの作成
cat > README.md << EOF
# プロジェクト名

プロジェクトの説明を書きます。

## 環境構築

\`\`\`bash
# 1. リポジトリのクローン
git clone リポジトリURL
cd プロジェクト名

# 2. Python環境のセットアップ
pyenv local 3.11.9  # または必要なバージョン
poetry install

# 3. 仮想環境の有効化
source \$(poetry env info --path)/bin/activate
\`\`\`

## 使い方

使用方法の説明を書きます。
EOF
```

## 7. 環境の検証

```bash
# Poetryの仮想環境が有効になっていることを確認
which python

# Pythonバージョンの確認
python --version

# インストールされたパッケージの確認
pip list

# 簡単なPythonコードで検証
python -c "import sys; print(sys.executable); print(sys.version)"
```

## 日常的な開発ワークフロー

```bash
# プロジェクトディレクトリに移動
cd プロジェクト名

# 環境を有効化
source $(poetry env info --path)/bin/activate

# 新しいパッケージを追加する場合
poetry add 新しいパッケージ名

# 開発用パッケージを追加する場合
poetry add --group dev 新しい開発用パッケージ名

# スクリプトを実行
poetry run python src/プロジェクト名/main.py

# またはアクティベート後に直接実行
python src/プロジェクト名/main.py

# テストを実行
poetry run pytest

# 環境を終了
deactivate
```

## トラブルシューティング

### pyenvでPythonインストールが失敗する場合

必要な依存パッケージが不足している可能性があります：

```bash
# Ubuntuの場合、依存パッケージの追加インストール
sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
```

### Poetryのインストールやコマンドが動かない場合

PATHが正しく設定されていない可能性があります：

```bash
# Poetry実行ファイルの場所を確認
find $HOME -name poetry -type f

# 見つかったパスをPATHに追加
export PATH="見つかったパスの親ディレクトリ:$PATH"
```

### uvコマンドが見つからない場合

```bash
# uvの実行ファイルの場所を確認
find $HOME -name uv -type f

# Cargoのbinディレクトリをパスに追加
export PATH="$HOME/.cargo/bin:$PATH"
```

## Tips

- **依存関係の解決速度向上**: `poetry add` に時間がかかる場合、`poetry config experimental.new-installer false` を設定すると改善することがあります
- **キャッシュクリア**: 問題が発生した場合 `poetry cache clear --all .` でキャッシュをクリアできます
- **uvとPoetry**: uvはPoetryの依存関係解決には影響せず、パッケージのインストール高速化のみに作用します

## まとめ

この環境構築ガイドに従うことで、最新のPython開発ベストプラクティスに基づいた開発環境を構築できます。pyenvでPythonバージョンを管理し、Poetryでプロジェクト依存関係を管理し、uvでインストールを高速化する組み合わせは、2025年現在の効率的な開発環境として推奨されています。
