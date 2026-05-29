# CCSKDE Handoff — resume notes

How to pick up CCSKDE (Context/Scene-Conditioned Sequential Keypoint Density
Estimation for traffic-hazard detection) in a fresh session.

## TL;DR of current state (2026-05-29)
The project **works now**. The original milestone tie was traced to a
coordinate-frame bug (fixed). On top of that we built the oriented **car-matrix**
context, a **FiLM** covariance gate, and the headline **Unified Scene-SKDE**
(cars modelled as extra skeletal joints in one autoregressive density). All code
is implemented, tested, run, and written up. Nothing is committed yet.

**Results (ShanghaiTech, 10 ep, batch 1024, seg-len 24, stride 1, seed 42, RTX 5080):**
- baseline 0.735 best / 0.720 mean → context & scene models **0.80 / 0.79**
  (**+6–7 pp mean**).
- Hazard-focused: score↔proximity ρ **0.16 → 0.64**, near-vehicle-hazard AUROC
  **0.79 → 0.965** (car-matrix + FiLM).
- Numbers + per-config checkpoints in `colab_results/results_v2/`.

Read `README.md` for the overview, `project_progress_report.md` #20–#29 for the
full log, `check-here-doruk.md` for open decisions.

## Machine / environment
- Windows 11, **RTX 5080 (Blackwell, sm_120)** → PyTorch **cu128** only (never
  downgrade to cu121). venv at repo root: `.venv\Scripts\python.exe`.
- `set PYTHONPATH=%CD%` before running `ccskde/` or `scripts/`.
- Windows DataLoader: `--num_workers 0`. Do NOT pipe native exes
  (python/pdflatex) through PowerShell `*>`/`2>&1`/`| Out-Null` — it produces
  0-byte logs or hangs the process. Run plainly and let the harness capture, or
  log from Python.
- Data under `data/ShanghaiTech/` (856×480). Oriented car caches already
  extracted: `data/ShanghaiTech/context_oriented/{train,test}` (107 + 330).

## What exists (all done)
- `ccskde/context/`: `detect.py` (`extract_clip_oriented`), `vehicle.py`
  (car-skeleton matrix), `scene.py` (scene-skeleton builder), `build.py`
  (proximity C_t, coord-fixed), `config.py`.
- `ccskde/models/`: `made_partial_context.py` (context MADE + FiLM),
  `scene_made.py` (generalised MADE, N'=18+6M).
- `ccskde/data/`: `contextual_dataset.py` (+`precompute()`), `scene_dataset.py`.
- `ccskde/eval/hazard.py`; `ccskde/training.py` (`CCSKDETrainer`, `SceneTrainer`).
- `scripts/`: `run_experiments.py` (configs: baseline, proximity,
  proximity_shuffled, vehicle, vehicle_film, vehicle_shuffled, **scene**),
  `evaluate_hazard.py` (`--context_mode {none,proximity,vehicle,scene}`,
  `--scene_readout {ped,joint}`), `extract_yolo_detections.py --oriented`.
- `tests/`: context coords, vehicle geometry, AR-mask (0 violations), hazard
  eval, scene model. Run any with `.venv\Scripts\python tests\<name>.py`.
- `report/main.tex` (+ compiled `main.pdf`, 8 pp). Skills: `.claude/skills/
  {ccskde-run,ccskde-eval}`.

## To reproduce / continue (commands)
See the **`ccskde-run`** and **`ccskde-eval`** skills for the authoritative,
copy-pasteable command lines (extraction, full matrix incl. `scene`,
hazard eval, tests). Results write to `colab_results/results_v2/` incrementally;
the runner **resumes** (skips configs already present).

## Suggested next steps (open, non-blocking — details in check-here-doruk.md)
1. **Commit** the work to a branch (currently uncommitted on `main`).
2. **Road elements** as scene keypoints via YOLO-World (open-vocab; ≤50 MB).
3. **yolo11l-seg** re-extraction for cleaner car skeletons (~1 h GPU).
4. **Multi-seed** runs (the comparison is single-seed; baseline is high-variance).
5. **Joint-readout scene scoring** as the default for vehicle-intrinsic hazards
   (a car driven into a pedestrian zone) — `--scene_readout joint`.
6. Full 2×2 context-rotated covariance (currently diagonal + FiLM).
