# EMSVfilter：IGVのスクリーンショットから構造変異を判定するAIツール
EMSVfilterは、画像認識AIモデルがIntegrated Genomics Viewer (IGV) のスクリーンショットに構造変異が含まれるかどうかを判定する。  

## 開発情報

### 性能
IGVでコールされた場所(1225か所)のスクリーンショットから、構造変異の画像の100%、ノイズの画像の51%を正しく判定できた。
<table>
  <thead>
    <tr>
      <th colspan="2" rowspan="2"></th>
      <th colspan="2">モデルの判定</th>
    </tr>
    <tr>
      <th>Positive</th>
      <th>Negative</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2">実際の状態</th>
      <th>Positive</th>
      <td align="center">533 (True Positive)</td>
      <td align="center">0 (False Negative)</td>
    </tr>
    <tr>
      <th>Negative</th>
      <td align="center">342 (False Positive)</td>
      <td align="center">350 (True Negative)</td>
    </tr>
  </tbody>
</table>

### 動作概要
フォルダを再帰走査し、`sample_*/control_*` のペアを抽出。  
3つの画像認識モデル([ResNet](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)、
[Swin Transformer](https://huggingface.co/timm/swin_base_patch4_window7_224.ms_in1k)、
[Vision Transformer](https://huggingface.co/timm/vit_base_patch8_224.augreg_in21k))がSVである確率を計算。  
閾値判定を出し、いずれか1つでもモデルの判定がPositiveならツールの出力する判定はPositiveとなる。  
標準出力に判定が逐次表示。プログラム終了後に判定結果のTSVと詳細なJSON が出力される。

### モデルの学習方法  
トレーニングデータ：ノイズ 796ペア、構造変異 300ペア  
テストデータ：ノイズ 76ペア、構造変異 33ペア  
IGVのスクリーンショットを用意し、サンプルの画像とコントロールの画像を224×224にリサイズし、画像の画素差分の絶対値を1枚の画像にして学習に使用した。  

ImageNetで事前学習済みの
[ResNet](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)、
[Swin Transformer](https://huggingface.co/timm/swin_base_patch4_window7_224.ms_in1k)、
[Vision Transformer](https://huggingface.co/timm/vit_base_patch8_224.augreg_in21k)モデルをトレーニングデータで転移学習し、
その後3つのモデルを相互学習して汎化性能を高めた。  
さらにツールの判定のF1スコアを最大化するために、各モデルの判定の閾値を探索し、ResNet/Swin/ViT=0.7/0.075/0.975の閾値をデフォルトとした。

## 使用方法

FASTQファイルからGATK-SVを使ってSVをコールするまでの一連の流れは[GATK-SV-SS](https://github.com/c2997108/gatk-sv-ss)を参照

### インストール

A. 直接インストール

```
git clone https://github.com/Endo2001/EMSVfilter.git
cd EMSVfilter
python3 -m venv venv
venv/bin/python3 -m pip install --upgrade pip
venv/bin/python3 -m pip install torch torchvision timm pillow matplotlib numpy
wget https://github.com/Endo2001/EMSVfilter/releases/download/0.1/best_dml_all.pth
```

B. Docker使用

```
docker pull c2997108/emsvfilter:0.1
```

### 実行

A. 直接実行

```
venv/bin/python3 EMSVfilter.py image_dir > result.txt
```

B. Docker使用

```
docker run -it --rm --gpus all -v "$PWD:$PWD" -w "$PWD" c2997108/emsvfilter:0.1 EMSVfilter.py image_dir > result.txt
```

