# F0_Generator
「波音リツ」歌唱データベースVer2のデータをべースに、F0生成の実験を行います。

```
docker compose build --no-cache
docker compose up -d
docker exec -it f0_exp /bin/bash
poetry install
source .venv/bin/activate
bash install_vscode_extention.sh
```

```
poetry add <package_name>
```

## データ処理

```
python ./source/pipeline/preprocess/preprocess_svs.py --song_db_path ./path/to/「波音リツ」歌声データベースVer2/ --f0_method crepe --audio_normalize True
```

## 学習 & 評価
学習再開時は、`./config/train.yaml`内の`ckpt_path`にlogs等に保存されたモデルのファイルパスを記述することで再開可能。
```
python source/pipeline/train/train.py --config_path /home/workdir/configs/ --config_name train.yaml
```
## テスト
モデルパスはconfigに記述
```
python source/pipeline/test/test.py --config_path /home/workdir/configs/ --config_name train.yaml
```
