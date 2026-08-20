# Streaming GRU activity model (Windows training)

This package trains a six-class unidirectional GRU from the existing
deployment-aligned ST-GCN archive. It does not modify the ST-GCN files or
require CML JSON to be processed again.

## Input contract

- Source archive: `N,C,T,V,M`, normally `(N,3,32,17,1)`.
- Classes, in fixed order: walking, standing, idle, bending, carrying,
  material_handling.
- GRU input: `(N,16,51)` using temporal stride 2.
- Feature order: all 17 x coordinates, all 17 y coordinates, then all 17
  confidence/presence values.
- Live inference: one `(17,3)` COCO pose at a time with recurrent state per
  worker.

The default stride converts the ~11 Hz training archive to ~5.5 Hz, close to
the measured 5.43 Hz Digital Twin pose cadence.

## 1. Copy into the repository

Copy this `gru` directory to:

```text
training\gru
```

Run all commands from `training\gru`.

## 2. Environment

The existing `training\stgcn\.venv` can be reused:

```powershell
cd training\gru
..\stgcn\.venv\Scripts\Activate.ps1
python -c "import torch,numpy,sklearn,yaml; print(torch.__version__, torch.cuda.is_available())"
```

Or create an independent environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the appropriate Windows PyTorch build from
https://pytorch.org/get-started/locally/ if Torch is not already installed.

## 3. Validate the existing data

```powershell
python inspect_data.py --data ..\stgcn\data\combined\cml_plus_local.npz
python smoke_test.py
```

Do not start the full run unless all six classes appear in train, validation,
and test output.

## 4. Two-epoch training check

```powershell
python train.py `
  --data ..\stgcn\data\combined\cml_plus_local.npz `
  --output runs\cml_plus_local_smoke `
  --device cuda:0 `
  --epochs 2
```

Use `--device cpu` if CUDA is unavailable.

## 5. Full training

```powershell
python train.py `
  --data ..\stgcn\data\combined\cml_plus_local.npz `
  --output runs\cml_plus_local_sqrt `
  --device cuda:0
```

Outputs:

- `best.pt`: checkpoint plus preprocessing/model metadata.
- `metrics.json`: accuracy, macro-F1, and per-class results.
- `confusion_matrix.csv`.
- `history.csv`.
- `splits.npz` and `split_manifest.json`: reproducible leakage-safe split.

The checkpoint is selected only by validation macro-F1. The test subset is
evaluated after training.

## 6. Live integration contract

`streaming_inference.py` contains `StreamingGRUPredictor`. The Mac backend
should call `update(worker_id, pose)` only when a new Pi COCO-17 pose arrives.
It automatically:

- masks eye/ear indices 1-4;
- binarizes confidence;
- centres and scales the pose identically to CML preprocessing;
- retains one hidden state per worker;
- waits for five observations before declaring the model ready;
- resets state after a two-second gap.

Keep GRU output diagnostic (`activity.gru`) until it passes a new locked live
test. Do not train on the prior synchronized evaluation and then continue to
describe that same session as an unseen test.

