# Windows CML → ST-GCN training scaffold

This first milestone prepares training without changing Mac inference or running a long job. It emits the live backend convention `N,C,T,V,M`, with each sample shaped `(3,32,17,1)` (`x,y,confidence`; COCO-17; one person).

## CML assumptions and items to confirm

The CML paper and authors' repository document JSON samples at 30 Hz with metadata, `tdata` (frame-major joint coordinates), and `bdata` (joint-major series), using 15- or 20-joint 3D skeletons in metres. The parser uses `tdata`. CML `Head` maps to COCO `nose`; its absent eye/ear nodes are zero-confidence. For 15 joints, `Hand`/`Foot` map to COCO wrist/ankle; the 20-joint endpoints are used when present.

The downloaded v3 release confirms that labels use the `label` field, source
sample identity uses `source file`, and `tdata` is a frame-keyed object whose
values are flattened `V*3` coordinate lists. Remaining decisions/gaps are:

- CML `z` is vertical, so the configured projection maps source `x,z` into
  deployment `x,y`. This preserves body height but remains an orthographic
  approximation of the Raspberry Pi camera view.
- CML has no `idle` label. Its `standing` samples are tagged from original
  label `still`, so they cannot also supervise `idle` without contradictory
  labels. Another labelled source or a project-specific annotation pass is
  required for a genuine six-class model.
- subject/session fields are absent in inspected samples. Splitting therefore
  groups by `data source` plus `source file`, preventing windows from one source
  sample crossing splits.

Confirmed mappings are in `configs/classes.yaml`. Standing and idle come only
from the local Raspberry Pi recordings. The full `All_DATA/20_nodes` tree is
needed for `carry suitcase`; the construction-only subset omits it. Use only
one skeleton tree to avoid duplicating every sample.

## PowerShell setup (Windows)

Check the GPU and driver first:

```powershell
nvidia-smi
Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion
```

Create the environment from the repository root:

```powershell
cd training\stgcn
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'runtime',torch.version.cuda,'gpu',torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

If the generic PyTorch install reports no CUDA, use the current Windows/Pip/CUDA command from https://pytorch.org/get-started/locally/ and rerun the final check. A separate CUDA Toolkit is not required by the PyTorch wheel; a compatible NVIDIA driver is required.

## Dataset and commands

Download the CML public release (not SARD) from DOI `10.6084/m9.figshare.20480787`, then extract JSON under `training\stgcn\data\cml\raw\` (ignored by Git).

```powershell
python preprocess.py --input data\cml\raw\All_DATA\All_DATA\20_nodes --inspect-labels
python preprocess.py --input data\cml\raw\All_DATA\All_DATA\20_nodes --output data\cml\processed\cml_coco17_11hz_w32.npz
python smoke_test.py
```

Full training, only after mappings and data inspection are complete:

```powershell
python train.py --data data\cml\processed\cml_coco17_11hz_w32.npz --output runs\cml_stgcn --device cuda:0
```

## Local standing and idle recordings

MKV files are accepted directly. Put independent clips below `data/local/videos/standing`
and `data/local/videos/idle`. Use the same pose model as the Raspberry Pi where possible.
The original `.pt` file is fastest on Windows; the exported NCNN pose-model directory is
also accepted by Ultralytics.

For cross-domain alignment, local COCO eye/ear nodes 1-4 are zeroed because CML
provides only one Head joint, and retained joint confidences are binarized because
CML has no detector-confidence values. The tensor remains `(3,32,17,1)`. When the
trained model is integrated later, the same facial-node mask and confidence
binarization must be applied immediately before inference; live Mac code is not
changed in this milestone.

Install the added video dependencies without replacing the working CUDA Torch build:

```powershell
pip install opencv-python "ultralytics>=8.3,<9"
python -c "import torch,cv2,ultralytics; print(torch.cuda.is_available(),cv2.__version__,ultralytics.__version__)"
```

Smoke-test one clip for five seconds first:

```powershell
python preprocess_local_video.py --input data\local\videos --model C:\path\to\matching-pose-model.pt --output data\local\processed\local_smoke.npz --max-videos 1 --max-seconds 5
```

Review the printed detection rate and `.diagnostics.json`, then process all clips:

```powershell
python preprocess_local_video.py --input data\local\videos --model C:\path\to\matching-pose-model.pt --output data\local\processed\standing_idle.npz
```

Create CML windows from only one representation, then combine them:

```powershell
python preprocess.py --input data\cml\raw\All_DATA\All_DATA\20_nodes --output data\cml\processed\cml_coco17_11hz_w32.npz
python combine_datasets.py --inputs data\cml\processed\cml_coco17_11hz_w32.npz data\local\processed\standing_idle.npz --output data\combined\cml_plus_local.npz
python train.py --data data\combined\cml_plus_local.npz --output runs\cml_plus_local --device cuda:0
```

Use `--device cpu` for a CPU run. Training writes `best.pt`, `metrics.json`,
`confusion_matrix.csv`, and `history.csv`; it uses deterministic seeds,
square-root-balanced class weights, validation-driven learning-rate reduction,
checkpoint selection by validation macro-F1, and source sample/session group
splits. Full inverse-frequency weights remain available with
`class_weights: balanced`, but strongly over-emphasized the rare CML classes in
the first experiment.
