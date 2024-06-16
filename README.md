# F0_Generator
「波音リツ」歌唱データベースVer2のデータをべースに、F0生成の実験を行います。

## 環境構築 Docker & poetry & venv
```
docker compose build --no-cache
docker compose up -d
docker exec -it f0_exp /bin/bash
poetry install
source .venv/bin/activate
bash install_vscode_extention.sh
```

### 新規ライブラリ追加
```
poetry add <package_name>
```

## データ処理
「波音リツ」歌声データベースVer2 : [Download](https://drive.google.com/drive/folders/1XA2cm3UyRpAk_BJb1LTytOWrhjsZKbSN)
```
python ./source/pipeline/preprocess/preprocess_svs.py --song_db_path ./path/to/「波音リツ」歌声データベースVer2/ --f0_method crepe --audio_normalize True
```

## 学習 & 評価
学習再開時は、`./config/train.yaml`内の`ckpt_path`にlogs等に保存されたモデルのファイルパスを記述することで再開可。
```
python source/pipeline/train/train.py --config_path /home/workdir/configs/ --config_name train.yaml
```
## テスト
モデルパスはconfig/eval.yamlに記述
```
python source/pipeline/test/test.py --config_path /home/workdir/configs/ --config_name eval.yaml
```

## コード管理 Git

```
# SSHのKeyを登録する＆空のリポジトリ作成する
git config --global user.name "user_name"
git config --global user.email "yourmail@address"

# コード複製
git clone url

# コード追加
git add .
git commit -m "コメント"

# コードアップロード
git push

# git add取り消し
git reset HEAD

# 既存コード反映
git pull

```