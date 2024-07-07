# F0_Generator
「波音リツ」歌唱データベースVer2のデータをべースに、F0生成の実験を行います。

## 実行環境構築
```
docker compose build --no-cache
docker compose up -d
docker exec -it f0_exp /bin/bash
poetry install
source .venv/bin/activate
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install
```

## データ処理
「波音リツ」歌声データベースVer2 : [Download](https://drive.google.com/drive/folders/1XA2cm3UyRpAk_BJb1LTytOWrhjsZKbSN)
```
# docker exec -it f0_exp /bin/bash # DockerContainerへ
# source .venv/bin/activate # Python仮想環境ON
python ./source/pipeline/preprocess/preprocess_svs.py --song_db_path ./path/to/「波音リツ」歌声データベースVer2/ --f0_method crepe --audio_normalize True
```

## 学習 & 評価
学習再開時は、`./config/train.yaml`内の`ckpt_path`にlogs等に保存されたモデルのファイルパスを記述することで再開可。
```
# docker exec -it f0_exp /bin/bash # DockerContainerへ
# source .venv/bin/activate # Python仮想環境ON
python source/pipeline/train/train.py --config-path /home/user/project/configs/ --config-name train.yaml
```

## テスト
モデルパスはconfig/eval.yamlに記述
```
# docker exec -it f0_exp /bin/bash # DockerContainerへ
# source .venv/bin/activate # Python仮想環境ON
python source/pipeline/eval/eval.py --config_path /home/user/project/configs/ --config_name eval.yaml
```