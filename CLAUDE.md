# CCSKDE — Context/Scene-Conditioned Sequential Keypoint Density Estimation

## Ultimate target
**Extract skeletal information from road agents — cars, buses, cycles (and, where
possible, road elements) — and model them jointly with pedestrians under a single
autoregressive keypoint-density formula, so that one likelihood-based score
detects hazardous on-road situations efficiently and effectively.**

We take SeeKer/SKDE (ICCV 2025), which models *only* a pedestrian's 18-keypoint
skeleton and max-pools people independently, and **merge cars into the density
itself**: each vehicle becomes a compact 6-keypoint "skeleton" (oriented box →
corners+center+heading), appended to the pedestrian skeleton to form one scene
skeleton of `N' = 18 + 6M` keypoints. SeeKer's autoregressive factorization is
applied to the joint set, with vehicles ordered *before* the pedestrian so the
pedestrian's keypoints are predicted **given** the surrounding vehicles. The
single hazard score is the confidence-weighted negative log-likelihood

```
S(t) = - sum_n  c_{t,n} * ln p_theta( Z_{t,n} | Z_{t,<n}, Z_Delta )
```

over the scene-skeleton keypoints Z (pedestrian + vehicles). A pedestrian moving
normally in open space and the same motion into the path of a vehicle now yield
*different* likelihoods — the interaction is captured by the cross-agent
conditioning, not by hand-crafted features.

This generalises the earlier CCSKDE idea (cars as a side "context vector") into a
unified scene model. Both live in the repo and are compared.

## Environment (READ FIRST)
- Windows 11 + **NVIDIA RTX 5080 (Blackwell, sm_120)** → PyTorch **must** be the
  **cu128** build. cu121 wheels will not run; never downgrade torch.
- venv at repo root: `.venv\Scripts\python.exe`. Set `PYTHONPATH=%CD%` (repo root)
  before running anything under `ccskde/` or `scripts/` (seeker uses bare imports).
- ShanghaiTech is uniformly **856×480**. Data under `data/ShanghaiTech/`.
- Windows DataLoader: use `--num_workers 0`.
- Do NOT pipe native exes (python/pdflatex) through PowerShell `*>`/`2>&1` — it
  produces 0-byte logs. Let the harness capture, or log from Python.

## Layout
- `seeker/` — vendored SeeKer baseline. **Never edit** except the existing
  portability patches; to change behaviour, subclass/override in `ccskde/`.
- `ccskde/`
  - `context/` — perception front-end (offline): `detect.py` (YOLO box +
    `extract_clip_oriented` seg→oriented keypoints), `build.py` (proximity C_t),
    `vehicle.py` (oriented car-skeleton matrix), `config.py` (`ContextSpec`),
    `scene.py` (**scene-skeleton builder — the unified model**).
  - `models/` — `made_partial_context.py` (context MADE + FiLM),
    `scene_made.py` (**generalised MADE for N' scene keypoints**).
  - `data/` — `contextual_dataset.py` (pose + C_t), `scene_dataset.py`
    (**augmented scene skeleton**).
  - `eval/hazard.py` — hazard-subset metrics + score-vs-proximity.
  - `training.py`, `seeker_ctx.py` — context trainer + entry point.
- `scripts/` — `extract_yolo_detections.py` (`--oriented`),
  `run_experiments.py` (full matrix), `evaluate_hazard.py`.
- `tests/` — context coords, vehicle geometry, AR-mask preservation, hazard eval,
  scene model.
- `.claude/skills/` — `ccskde-run`, `ccskde-eval`.

## Conventions
- The autoregressive mask must be preserved: any model change is validated by
  `tests/test_ar_mask.py` (must report **0 forbidden-region violations**).
- Perception (YOLO, AlphaPose) is **offline preprocessing only** — zero
  train-time cost. Caches under `data/ShanghaiTech/context*` and
  `colab_results/`.
- Append engineering decisions to `project_progress_report.md` (numbered log).
  Blockers / things needing the user go in `check-here-doruk.md`.

## Key commands
See the `ccskde-run` and `ccskde-eval` skills for the authoritative, current
command lines (extraction, experiment matrix, hazard eval, tests).

## Current status (resume here)
Implemented, tested, run, and written up — **uncommitted on `main`**.
- Coordinate-frame bug fixed; oriented car-matrix context, FiLM covariance gate,
  and the unified Scene-SKDE all built and validated (tests green; AR mask: 0
  violations). Results in `colab_results/results_v2/`.
- Context/scene give **+6–7 pp mean AUROC** over baseline; hazard score↔proximity
  ρ 0.16→0.64, near-vehicle-hazard AUROC 0.79→0.965.
- **To continue:** read `README.md` + `HANDOFF.md`; open decisions in
  `check-here-doruk.md` (commit-to-branch, YOLO-World road elements,
  yolo11l-seg, multi-seed, joint-readout default, full 2×2 covariance).
  Full chronological log: `project_progress_report.md` #20–#29.
