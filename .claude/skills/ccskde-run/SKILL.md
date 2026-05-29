---
name: ccskde-run
description: Run CCSKDE / SeeKer training and the full experiment matrix on ShanghaiTech (baseline, proximity, vehicle "car-skeleton", FiLM, shuffled controls). Use when asked to train, reproduce, launch experiments, or extract YOLO context for this project.
---

# Running CCSKDE training & experiments

CCSKDE extends the SeeKer (SKDE) skeleton-based video-anomaly-detection baseline
with environmental context. This skill covers extraction + training + the
experiment matrix on **ShanghaiTech**.

## Environment
- Windows + **RTX 5080 (Blackwell)** → PyTorch must be the **cu128** build. Never
  let anything downgrade torch to cu121.
- venv at repo root: `.venv\Scripts\python.exe`. Always `set PYTHONPATH=<repo root>`
  before invoking `ccskde/` or `scripts/` modules (seeker uses bare imports).
- ShanghaiTech is uniformly **856×480**. Data lives under `data/ShanghaiTech/`
  (poses in `pose/{train,test}`, GT in `gt/test_frame_mask`, test frames in
  `shanghaitech/testing/frames`, train videos in `shanghaitech/training/videos`).

## Step 1 — context extraction (offline, once)
Box (axis-aligned) detections — the original cache, reusable:
```
.venv\Scripts\python scripts\extract_yolo_detections.py --data-root data\ShanghaiTech --split test
.venv\Scripts\python scripts\extract_yolo_detections.py --data-root data\ShanghaiTech --split train
```
Oriented "car-skeleton" detections (YOLOv11-seg → minAreaRect keypoints), needed
for the `vehicle*` configs:
```
.venv\Scripts\python scripts\extract_yolo_detections.py --data-root data\ShanghaiTech --split test  --oriented
.venv\Scripts\python scripts\extract_yolo_detections.py --data-root data\ShanghaiTech --split train --oriented
```
Caches: box → `data/ShanghaiTech/context/<split>/`, oriented →
`data/ShanghaiTech/context_oriented/<split>/`. A full train+test cache also
ships under `colab_results/results/context_cache/` (box only).

## Step 2 — run the experiment matrix
The unified runner loads poses ONCE and runs all selected configs:
```
set PYTHONPATH=%CD%
.venv\Scripts\python scripts\run_experiments.py ^
  --data_dir data ^
  --proximity_cache colab_results\results\context_cache ^
  --oriented_cache data\ShanghaiTech\context_oriented ^
  --device cuda --epochs 10 --batch_size 1024 --seg_len 24 --seg_stride 1 ^
  --configs baseline proximity vehicle vehicle_film vehicle_shuffled ^
  --out colab_results\results_v2
```
Configs: `baseline` (vanilla SeeKer), `proximity` (corrected per-class
[1/d_min,count]), `proximity_shuffled`, `vehicle` (oriented car-skeleton matrix
as context), `vehicle_film` (+ FiLM covariance modulation), `vehicle_shuffled`
(negative control), and **`scene`** — the unified scene-SKDE where cars are
extra skeletal joints in ONE autoregressive density (N'=18+6M keypoints,
vehicles first; pedestrian-keypoint readout for scoring). The runner resumes:
re-run with `--configs scene` to add it to an existing results file. Output:
`experiment_results.json` (per-epoch AUROC, best, mean) + `auroc_curves.png`,
written incrementally after each config.

## Single training run (one config)
Baseline: `cd seeker && python seeker.py --dataset ShanghaiTech --data_dir <abs> --exp_dir <abs> --device cuda ...`
CCSKDE:
```
.venv\Scripts\python ccskde\seeker_ctx.py --dataset ShanghaiTech --data_dir data ^
  --exp_dir exp_dir\run1 --device cuda --context_cache_dir data\ShanghaiTech\context_oriented ^
  --context_mode vehicle --film_cov --seg_len 24 --batch_size 1024 --epochs 10 --seed 42 --num_workers 0
```

## Gotchas
- On Windows use `--num_workers 0` (DataLoader multiprocessing is flaky).
- Do NOT pipe a native exe through PowerShell `*>`/`2>&1` — it can swallow all
  output (0-byte logs). Let the harness capture stdout, or write to a file from
  Python.
- Pose loading is ~3–4 min per process (single-threaded JSON). The runner pays
  this once; per-config subprocess launches pay it each time.
- Per-epoch validation on ShanghaiTech is intentional (val == test); see
  `ccskde/training.py`.
