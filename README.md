# CCSKDE — Context/Scene-Conditioned Sequential Keypoint Density Estimation
### Skeleton-based traffic-hazard detection by modelling road agents jointly

CCSKDE extends **SeeKer / SKDE** (ICCV 2025), an autoregressive keypoint-density
estimator for skeleton-based video anomaly detection. SeeKer models *only* a
pedestrian's 18-keypoint skeleton and is **environmentally blind** — a person
walking calmly and the same person walking into a vehicle's path receive
identical likelihoods. We make the density **aware of the surrounding road
agents**.

> **Ultimate target.** Extract skeletal information from road agents — cars,
> buses, cycles (and, where possible, road elements) — and model them jointly
> with pedestrians under a **single autoregressive keypoint-density formula**, so
> that one likelihood-based score detects hazardous on-road situations
> efficiently and effectively.

The headline model — **Unified Scene-SKDE** — turns each vehicle into a compact
6-keypoint "skeleton" (oriented box → corners + center + heading) and appends it
to the pedestrian skeleton to form one **scene skeleton** of `N' = 18 + 6M`
keypoints. SeeKer's factorization is applied to the joint set, vehicles ordered
*before* the pedestrian, so the pedestrian is predicted **given** the cars. The
single hazard score is

```
S(t) = − Σ_n  c_{t,n} · ln p_θ( Z_{t,n} | Z_{t,<n}, Z_Δ )
```

over the scene-skeleton keypoints `Z` (pedestrian + vehicles). The interaction is
captured by the cross-agent conditioning — not by hand-crafted features.

---

## What this project achieved

1. **Diagnosed and fixed the core defect** behind the original null result: the
   environmental context was computed in the wrong coordinate frame (pedestrian
   centroid taken from the *normalised* pose while detections are in image
   space), making it pedestrian-independent. Fixed → context started helping.
2. **Oriented "car-matrix" context** — each nearest hazard as a 6-keypoint
   oriented skeleton from a YOLOv11-seg mask, in pedestrian-relative,
   scale-invariant coordinates.
3. **FiLM covariance modulation** — context gates the predicted covariance
   (the "contextual covariance penalty"), realised as a zero-initialised FiLM
   head; autoregressive structure provably preserved.
4. **Unified Scene-SKDE** — cars modelled as first-class skeletal agents in one
   joint density (the single-formula model above). To our knowledge the first
   time SeeKer's keypoint density is taken beyond a single human skeleton.
5. **Honest hazard-subset evaluation** — isolates pedestrian-vehicle events and
   measures score-vs-proximity correlation (the question the milestone never
   answered).
6. **Engineering** — unified experiment runner, hazard-eval driver, a `tests/`
   suite (incl. an autoregressive-mask Jacobian check), and two project skills.

### Results (ShanghaiTech, 10 epochs, batch 1024, seg-len 24, stride 1, seed 42, RTX 5080)

| Model | Best AUROC | Mean AUROC |
|---|---|---|
| SeeKer baseline | 0.735 | 0.720 |
| CCSKDE proximity (coord-fixed) | **0.803** | 0.786 |
| CCSKDE car-matrix (vehicle skeleton) | 0.802 | 0.786 |
| CCSKDE car-matrix + FiLM | 0.797 | **0.790** |
| car-matrix, shuffled context (control) | 0.789 | 0.784 |
| **Unified Scene-SKDE** | 0.798 | 0.786 |

Context/scene gives **+6–7 pp mean AUROC** over the baseline. The hazard-focused
view is sharper:

| Model | Vehicle-hazard AUROC | Score↔proximity ρ |
|---|---|---|
| SeeKer baseline | 0.79 | 0.16 |
| CCSKDE car-matrix + FiLM | **0.965** | **0.64** |
| Unified Scene-SKDE | 0.95 | 0.60 |

The car-aware models make the anomaly score fire specifically when a pedestrian
is near a vehicle — the behaviour the project set out to obtain. (Single-seed;
~1 pp of the AUROC gain is genuine spatial alignment per the shuffled control,
the rest is structured-input capacity. See `report/` for the full discussion.)

---

## Repository layout

```
seeker/        vendored SeeKer baseline (read-only; portability patches only)
ccskde/
  context/     perception front-end (offline): detect.py (YOLO box + oriented seg),
               build.py (proximity C_t), vehicle.py (car-skeleton), scene.py
               (scene-skeleton builder), config.py (ContextSpec)
  models/      made_partial_context.py (context MADE + FiLM), scene_made.py
               (generalised MADE for N' scene keypoints)
  data/        contextual_dataset.py (pose + C_t), scene_dataset.py (scene skeleton)
  eval/        hazard.py (hazard-subset metrics + score-vs-proximity)
  training.py, seeker_ctx.py
scripts/       extract_yolo_detections.py (--oriented), run_experiments.py,
               evaluate_hazard.py
tests/         context coords, vehicle geometry, AR-mask, hazard eval, scene model
report/        ICLR-style write-up (main.tex / main.pdf)
.claude/skills/ ccskde-run, ccskde-eval
```

## Setup

- Windows + **NVIDIA RTX 5080 (Blackwell)** → PyTorch **must** be the **cu128**
  build (`pip install torch torchvision --index-url
  https://download.pytorch.org/whl/cu128`). cu121 wheels will not run.
- `python -m venv .venv` at repo root; `pip install -r requirements.txt`.
- Always `set PYTHONPATH=%CD%` before running `ccskde/` or `scripts/`.
- ShanghaiTech (uniformly 856×480) under `data/ShanghaiTech/`.

## Quick start

```bat
set PYTHONPATH=%CD%

:: 1. extract oriented car-skeleton context (offline, once)
.venv\Scripts\python scripts\extract_yolo_detections.py --data-root data\ShanghaiTech --split test  --oriented
.venv\Scripts\python scripts\extract_yolo_detections.py --data-root data\ShanghaiTech --split train --oriented

:: 2. run the experiment matrix (incl. the unified `scene` model)
.venv\Scripts\python scripts\run_experiments.py --data_dir data ^
  --proximity_cache colab_results\results\context_cache ^
  --oriented_cache data\ShanghaiTech\context_oriented ^
  --device cuda --epochs 10 --batch_size 1024 --seg_len 24 --seg_stride 1 ^
  --configs baseline proximity vehicle vehicle_film vehicle_shuffled scene ^
  --out colab_results\results_v2

:: 3. hazard-subset evaluation of a checkpoint
.venv\Scripts\python scripts\evaluate_hazard.py --checkpoint <best.pth> ^
  --data_dir data --context_mode scene --scene_readout joint --device cuda

:: 4. tests
.venv\Scripts\python tests\test_ar_mask.py
```

See the **`ccskde-run`** and **`ccskde-eval`** skills (`.claude/skills/`) for the
authoritative, current command lines. Engineering log: `project_progress_report.md`.
Open decisions for the maintainer: `check-here-doruk.md`.

## References

- Delić, Grcić, Šegvić. *Sequential Keypoint Density Estimator.* ICCV 2025.
- Germain et al. *MADE: Masked Autoencoder for Distribution Estimation.* ICML 2015.
- Perez et al. *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI 2018.
- Liu et al. *Future Frame Prediction for Anomaly Detection.* CVPR 2018 (ShanghaiTech).
