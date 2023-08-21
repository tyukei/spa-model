# spa-model


https://github.com/tyukei/spa-model/assets/70129567/819cd06a-9eb0-4399-b79c-e3ce44fa3caf


## Description

https://spa-label-maker-0822-10.streamlit.app

こちらのリンクからファイルをアップロードすると
ラベル付きの予想グラフが表示されます

また、ラベルが追加されたファイルのダウンロードもできます

## algorithm

### 1 ラベルの順番
ラベル２、３、４の順番になるようにする

もし違う順番になるのであれば、正しい順番になるようにする

### 2 体表温度の波形の形
体表温度の1階微分、2階微分よりラベルの推定を行う　

まず、目視で規則性を特定　　

次に、グラフの移動平均や１階微分の近似０等微調整を行なった

<img width="783" alt="Screenshot 2023-08-22 at 2 39 03" src="https://github.com/tyukei/spa-model/assets/70129567/2daab8bf-0790-4faf-b81d-83a05c039667">


<img width="226" alt="Screenshot 2023-08-22 at 2 33 23" src="https://github.com/tyukei/spa-model/assets/70129567/bce31177-0d7b-4ad9-97cd-c13f1a479d34">


### 3 ラベル０の処理

ラベルが遷移する時、ラベル０を挟むように調整

また、ディバイス起動時とディバイス終了時には０になるように調整



## Setup
```
python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

streamlit run app.py
```
