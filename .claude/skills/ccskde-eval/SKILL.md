---
name: ccskde-eval
description: Evaluate CCSKDE — read experiment AUROC results, run the hazard-subset analysis (isolate pedestrian-vehicle events), score-vs-proximity correlation, and the test suite. Use when asked to evaluate, compare configs, run ablations, or assess whether context helps traffic-hazard detection.
---

# Evaluating CCSKDE

The headline metric is frame-level **AUROC** on ShanghaiTech. But generic AUROC
mixes vehicle-hazard anomalies with motion-intrinsic ones (running, fighting),
so it cannot test the project's actual claim. Use the hazard-subset analysis for
that.

## 1. Read the experiment results
`<out>/experiment_results.json` holds, per config: `aucs` (per-epoch),
`best`, `best_epoch`, `mean`, `minutes`. Compare `vehicle*` vs `baseline` and
check the `*_shuffled` negative controls — a real signal means the aligned
context beats its shuffled counterpart.

## 2. Hazard-subset evaluation (the honest test)
Isolates the pedestrian-vehicle interaction regime. Checkpoints live at
`colab_results\results_v2\ShanghaiTech_<config>\<ts>\checkpoint_best.pth`.
```
set PYTHONPATH=%CD%
:: car-matrix + FiLM
.venv\Scripts\python scripts\evaluate_hazard.py ^
  --checkpoint colab_results\results_v2\ShanghaiTech_vehicle_film\<ts>\checkpoint_best.pth ^
  --data_dir data --oriented_cache data\ShanghaiTech\context_oriented ^
  --proximity_cache colab_results\results\context_cache ^
  --context_mode vehicle --film_cov --device cuda
:: unified scene model (joint readout catches vehicle-intrinsic hazards)
.venv\Scripts\python scripts\evaluate_hazard.py ^
  --checkpoint colab_results\results_v2\ShanghaiTech_scene\<ts>\checkpoint_best.pth ^
  --data_dir data --oriented_cache data\ShanghaiTech\context_oriented ^
  --proximity_cache colab_results\results\context_cache ^
  --context_mode scene --scene_readout joint --device cuda
:: baseline reference: --context_mode none
```
Reports `full`, `interaction_regime`, and `vehicle_hazard` AUROCs plus the
**Spearman score↔proximity** correlation over anomalous frames. NOTE: prefer
`vehicle_hazard` AUROC + score↔proximity ρ; the `interaction_regime` AUROC is
unreliable on ShanghaiTech because near-vehicle frames are ~91% anomalous
(near-degenerate subset). The proximity used for labelling comes from the box
cache (`--proximity_cache`, whose row layout has the center at cols 2:4).

Core metric functions live in `ccskde/eval/hazard.py`
(`hazard_subset_metrics`, `score_proximity_correlation`, `clip_frame_proximity`,
`sweep_tau`) and are pure numpy/scipy — reuse them directly.

### Latest numbers (2026-05-29, RTX 5080) for comparison
| config | best / mean AUROC | vehicle-hazard AUROC | score↔proximity ρ |
|---|---|---|---|
| baseline | 0.735 / 0.720 | 0.79 | 0.16 |
| vehicle_film | 0.797 / 0.790 | 0.965 | 0.64 |
| scene (joint) | 0.798 / 0.786 | 0.95 | 0.60 |

## 3. Run the test suite
```
.venv\Scripts\python tests\test_vehicle_context.py    # car-skeleton geometry
.venv\Scripts\python tests\test_ar_mask.py            # AR-causality preservation
.venv\Scripts\python tests\test_hazard_eval.py        # hazard-subset metrics
.venv\Scripts\python tests\test_context_coords.py     # C_t coordinate-frame fix
```
`test_ar_mask.py` is the load-bearing one: it must report **0 forbidden-region
violations** (no future-keypoint leakage) for the methodological claim to hold.

## What "working" looks like
1. context/scene configs beat `baseline` on **mean** AUROC (+6–7 pp; prefer mean
   over the noisy single-epoch best).
2. `vehicle` beats `vehicle_shuffled` (the ~1 pp gap = genuine spatial alignment;
   the rest of the gain is structured-input capacity — state this honestly).
3. **vehicle-hazard AUROC** and **score↔proximity ρ** are much higher for the car
   models than baseline (0.79→0.965, 0.16→0.64) — context helps where it should.
4. `tests\test_ar_mask.py` reports **0 forbidden-region violations**.
