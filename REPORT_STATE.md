# Report state & versioning

## Active version: **v3** (work from here onward)
- **Folder:** `cmp_719_final_report_v3/cmp719_final_latex/` (`main.tex`)
- **Figure generators:** `cmp_719_final_report_v3/make_figures.py`,
  `make_qualitative.py`, `make_demo.py`
- **Scope of v3:** everything in v2 **plus** the cross-dataset study
  (UBnormal: ours vs SeeKer; Street Scene/MERL if downloaded) and any further
  experiments. All new edits, figures, and tables go in **v3**.

## Frozen snapshot: **v2** (submission-ready, do NOT edit)
- **Folder:** `cmp_719_final_report_v2/cmp719_final_latex/`
- **State:** compiles clean, **11 pages**, 0 undefined refs / 0 warnings.
- **Contents (all real data, ShanghaiTech):** abstract/intro/method/related,
  dataset EDA, results with the metric suite (AUROC/AUPRC/EER/per-scene),
  paired significance (DeLong + clip-level bootstrap), counterfactual interaction
  probe, controlled ablations (M-sweep + agent ordering), qualitative figure,
  real-time **detection demo** figure, limitations/discussion/future, refs.
- Kept as the guaranteed-good fallback for the CMP719 submission.

## How v3 was created
Copied verbatim from v2 on 2026-06-05 (`Copy-Item -Recurse v2 v3`). v3 starts
identical to v2; new content is additive.

## Datasets (see `dataset_options.md` for the full dossier)
- **ShanghaiTech** — primary, done (results in `colab_results/`).
- **UBnormal** — chosen first for the cross-dataset *ours vs SeeKer* comparison
  (SeeKer-comparable; poses small). Needs RGB for our YOLO vehicle extraction.
- **Street Scene (MERL/Zenodo, 49 GB single zip)** — user will download (has
  bandwidth); the scientifically best mixed-traffic / vehicles-normal benchmark.

## Code shared by both report versions (in the repo, not the report folder)
`ccskde/`, `scripts/` (`run_experiments.py`, `evaluate_hazard.py`, `run_phase_a.py`,
`compare_models.py`, `realtime_demo.py`), `tests/`. Results caches in `colab_results/`.
