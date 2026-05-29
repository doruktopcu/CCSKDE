# CCSKDE Checkpoint

> ## Update — 2026-05-29 (uplift session, Windows / RTX 5080)
>
> The project was resumed on a new machine and substantially uplifted. The
> headline is that the milestone's null result was traced to a **bug**, not a
> dead idea (see progress log #20–#26).
>
> **Root cause found.** The context vector was computed in the wrong coordinate
> frame: the pedestrian centroid came from the *normalised* pose (zero-mean /
> unit-std) while YOLO boxes are in `[0,1]` image space, so `C_t` was
> effectively pedestrian-independent. Fixed (centroid from raw px ÷ (856,480)).
> A 1-epoch sanity run already moved corrected-proximity from a *tie* to
> baseline 0.742 → **0.790**.
>
> **Three contributions added on top:**
> 1. **Car-matrix context** — each nearest hazard as a 6-keypoint oriented
>    skeleton (YOLOv11-seg → minAreaRect), pedestrian-relative & scale-invariant.
> 2. **FiLM covariance modulation** — realises the proposal's "contextual
>    covariance penalty"; zero-init identity; AR-preserving (Jacobian test: 0
>    violations).
> 3. **Hazard-subset evaluation** — isolates pedestrian-vehicle events +
>    score-vs-proximity correlation (the question the milestone never answered).
>    Baseline reference: vehicle-hazard AUROC ≈ 0.84–0.88, Spearman ≈ 0.31.
>
> **Also:** report covariance description corrected (diagonal, not Cholesky —
> matches code & SeeKer's headline config); `tests/` suite added; two project
> skills (`ccskde-run`, `ccskde-eval`); unified `scripts/run_experiments.py`.
>
> **In flight at write time:** oriented YOLO-seg extraction on the train split,
> then the full experiment matrix → results land in `colab_results/results_v2/`
> and progress-log #27.

---

**Date:** 2026-05-23
**Branch:** `main` (clean except `seeker` submodule edits, intentional)
**Stage:** Milestone report submitted; planning remediation of the traffic-hazard evaluation gap.

---

## Where we are

### What was proposed
Context-Conditioned SKDE (CCSKDE): inject an environmental vector $C_t$ (YOLOv11-derived pedestrian-to-vehicle proximity) into the SeeKer autoregressive MADE so the model predicts $p_\theta(X_{t,n} \mid X_{t,<n}, X_\Delta, C_t)$. Purpose: **skeleton-based traffic hazard detection** on ShanghaiTech.

### What was built
- **Vendored baseline:** `seeker/` clone of Delić et al. (ICCV 2025), patched only for portability (8 lines, commit `3706984`, local-only).
- **Extension package:** `ccskde/` (sibling, never edits `seeker/`):
  - `ccskde/context/` — YOLOv11 detection cache + $C_t \in \mathbb{R}^{10}$ builder (`[1/(d_min+ε), n_{≤r}]` per hazard class: bicycle, car, motorcycle, bus, truck; r=0.15 normalized).
  - `ccskde/models/made_partial_context.py` — `MADEPartialContext` + `PartialAutoregressiveContextFC`. Input-layer mask appends all-ones columns for $C_t$ so context shortcut is full while keypoint AR causality is bit-for-bit preserved.
  - `ccskde/data/contextual_dataset.py` — wrapper that joins SeeKer's `SkeletonSequenceDataset` with cached $C_t$.
  - `ccskde/training.py` — `CCSKDETrainer` with `shuffled_context` ablation toggle.
  - `ccskde/seeker_ctx.py` — entry point mirroring `seeker/seeker.py`.
- **AR-preservation evidence:** Jacobian test — 0 forbidden-region violations, 258/260 allowed keypoint inputs non-zero, **80/80 $C_t$ inputs non-zero**.
- **Data:** ShanghaiTech Campus (Liu et al. 2018) + STG-NF preprocessed AlphaPose JSONs at `data/ShanghaiTech/pose/{train,test}/`. GT masks at `data/ShanghaiTech/gt/test_frame_mask/`.
- **Compute:** local M1 MPS used for smoke tests; full experiments moved to Colab A100 after MPS NaN at epoch 2.
- **Report:** `report/main.{tex,pdf}` (ICLR 2025 template), 7 pages, milestone submitted 2026-05-04.

### Results (A100, 10 epochs, batch 256, seed 42)

| Model                | Best AUROC | Epoch | Mean AUROC |
|----------------------|------------|-------|------------|
| Baseline (SeeKer)    | 0.7808     | 7     | 0.7308     |
| CCSKDE (ours)        | 0.7786     | 3     | **0.7496** |
| CCSKDE (shuffled $C_t$) | 0.7547  | 8     | 0.7465     |

Reproduced baseline 0.7808 vs. paper 0.855 — known reproducibility gap; all comparisons relative to our baseline. Artifacts under `colab_results/results/` (checkpoints, logs, `auroc_curves.png`, `experiment_results.json`, full `context_cache/{train,test}/` for all 437 clips).

### Honest verdict (where this checkpoint exists to address)
- **Methodology:** delivered. Conditioning preserves AR mask, shortcut is wired, ablation includes a proper negative control.
- **Headline claim of proposal — beating baseline on traffic hazards:** *not demonstrated.*
  - Peak AUROC is a tie (0.78 ≈ 0.78).
  - Mean AUROC +1.9 pp (stability win); shuffled gap 2.4 pp (signal-alignment win) — both real but not what was promised.
  - **Critical gap:** evaluation used generic ShanghaiTech AUROC over *all* anomaly types (cycling, skating, running, fighting, vehicles, …). No isolation of pedestrian-vehicle hazard events. The proposal's actual question was never tested.

### Repo state (as of checkpoint)
- Main repo `main`, up to date with origin.
- `seeker/` submodule: 1 unpushed commit (intentional), uncommitted edits to `seeker.py`/`training.py` (intentional, untracked artifacts: `__pycache__`, `exp_dir/`, `runs/`).
- Large local-only artifacts (gitignored): `ccskde_*.zip`, `results-*.zip`, `yolo11n.pt`, ICCV PDF.
- Key files: [project_proposal.md](project_proposal.md), [project_progress_report.md](project_progress_report.md), [report/main.tex](report/main.tex), [colab_results/results/experiment_results.json](colab_results/results/experiment_results.json).

---

## Remediation paths considered (2026-05-23 ideation)

- **A. Hazard-subset re-evaluation on ShanghaiTech** — programmatic relabel of test frames using YOLO caches + AlphaPose centroids; compute AUROC/AP on vehicle-proximity subset only. ~1 day, zero retraining.
- **B. Score-vs-proximity correlation analysis** — Spearman/Pearson between anomaly score and $1/d_{\min}$ during anomalous segments. ~0.5 day. Qualitative complement to A.
- **C. PET/TTC surrogate-safety relabeling** — ground the hazard subset in established traffic-safety thresholds (Ansariyar 2023, PET < 0.7s serious / < 1.31s general). ~2 days. Methodological weight.
- **D. Migrate to JAAD/PIE** — canonical pedestrian-traffic benchmarks; reframe CCSKDE as unsupervised hazard scoring. ~1 week. Scope-change, not remediation.
- **E. Synthetic hazard injection** — composite vehicles into pedestrian-only clips at controlled distances; plot score-vs-distance. ~2 days. Supplementary.

**Tentative recommendation:** A + B + C as a combined add-on. Reuses every artifact, no new training, directly answers the proposal's actual question. D is the path if scope expands to a final-thesis artifact.

---

## Target

> *(to be filled in — user will define what "done" looks like for this remediation. Candidate framings:*
> - *Demonstrate CCSKDE > baseline AUROC/AP on a defined traffic-hazard subset of ShanghaiTech, with a documented hazard-event definition (e.g. PET-based).*
> - *Produce a final report section showing score-vs-proximity correlation evidence that $C_t$ operates as the proposal described.*
> - *Migrate to JAAD or PIE and reframe CCSKDE as a skeleton-based hazard/crossing-risk scorer.*
> - *Other.)*

---

## Open items (carried forward)

- `seeker` submodule has uncommitted local edits — leave as-is per user (2026-05-23).
- Decide whether `report/` and `report_overleaf/` continue as the live document or a new section is appended for remediation results.
- No unit test directory yet; AR-mask Jacobian check exists only as a session-log script.
