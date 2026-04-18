# DATN_DeepVariant

Project train 4 model cho anh DeepVariant trong thu muc `tfrecords`:

- `InceptionV3`
- `ConvNeXtV2_Tiny`
- `EfficientNetV2_S`
- `ViT_Tiny`

Du lieu hien co metadata shape `[100, 221, 6]`. Loader doc cac shard `.tfrecord-*-of-*.gz`, parse feature mac dinh `image/encoded` va `label`, normalize anh ve `[0, 1]`, dung loss `SparseCategoricalCrossentropy(from_logits=True)`.

## Cai dat

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Neu may chua co Python tren PATH, cai Python 3.11 truoc:

```powershell
winget install -e --id Python.Python.3.11 --scope user
```

## Kiem tra TFRecord

```powershell
python -m deepvariant_train.inspect_tfrecord --data-dir tfrecords
```

Lenh nay in cac key trong mot example dau tien. Neu du lieu cua ban dung key khac voi `image/encoded`, `label`, hoac `image/shape`, truyen lai khi train bang `--image-key`, `--label-key`, `--shape-key`.

Smoke test loader va 4 model:

```powershell
python -m deepvariant_train.smoke_test --data-dir tfrecords
```

## Train tat ca model

```powershell
python -m deepvariant_train.train `
  --data-dir tfrecords `
  --model all `
  --epochs 20 `
  --batch-size 16
```

Train set mac dinh co shuffle:

- shuffle thu tu shard train
- shuffle record train voi `--shuffle-buffer 8192`
- `reshuffle_each_iteration=True` de moi epoch shuffle lai

Validation khong shuffle.

## Train tung model

```powershell
python -m deepvariant_train.train --data-dir tfrecords --model inceptionv3
python -m deepvariant_train.train --data-dir tfrecords --model convnextv2_tiny
python -m deepvariant_train.train --data-dir tfrecords --model efficientnetv2_s
python -m deepvariant_train.train --data-dir tfrecords --model vit_tiny
```

Alias cung dung duoc: `inception_v3`, `convnextv2-tiny`, `efficientnetv2-s`, `vit-tiny`.

## Output

Moi lan chay tao thu muc:

```text
runs/<timestamp>/<model_name>/
```

Ben trong co:

- `best.keras`: checkpoint tot nhat theo `val_loss`
- `last.keras`: model cuoi cung
- `history.csv`: log loss/accuracy
- `tensorboard/`: log TensorBoard
- `model_summary.txt`: kien truc model
- `config.json`: config va danh sach shard da train

## Tham so hay dung

```powershell
python -m deepvariant_train.train `
  --model convnextv2_tiny `
  --epochs 50 `
  --batch-size 8 `
  --learning-rate 1e-4 `
  --weight-decay 1e-4 `
  --shuffle-buffer 16384 `
  --mixed-precision
```

Smoke-test train 1 batch:

```powershell
python -m deepvariant_train.train `
  --model vit_tiny `
  --epochs 1 `
  --batch-size 2 `
  --steps-per-epoch 1 `
  --validation-steps 1 `
  --run-name smoke-train
```

## Train GPU bang WSL2

Neu TensorFlow tren Windows khong thay GPU, dung WSL2 Ubuntu. Script nay chay 4 model bang GPU, batch size mac dinh `8`, epochs mac dinh `20`, mixed precision bat san:

```powershell
wsl bash -lc "cd /mnt/d/DATN_DeepVariant && scripts/train_all_gpu_wsl.sh"
```

Theo doi log:

```powershell
wsl bash -lc "cd /mnt/d/DATN_DeepVariant && tail -f runs/<run-name>/train.log"
```

Kiem tra GPU:

```powershell
wsl nvidia-smi
```

Dung run nen:

```powershell
wsl bash -lc "cd /mnt/d/DATN_DeepVariant && kill $(cat runs/<run-name>/pid.txt)"
```

Tat shuffle train chi khi can debug:

```powershell
python -m deepvariant_train.train --model vit_tiny --no-train-shuffle
```
