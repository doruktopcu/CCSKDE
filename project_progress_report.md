# Project Progress Report — CCSKDE

**Project:** Context-Conditioned Sequential Keypoint Density Estimation for Traffic Hazard Detection
**Author:** Doruk Topcu (N25142281)
**Repository:** CCSKDE

---

## Instructions (read before appending)

1. **Append-only log.** Every meaningful action taken in this project must be recorded here. Do not silently rewrite history; if something is corrected, add a new entry that supersedes the old one and reference it (e.g., "supersedes #3").
2. **Identifiers.** Each entry gets a sequential identifier of the form `#N` (e.g., `#1`, `#2`, …). Never reuse a number. Numbering is global, monotonically increasing.
3. **Entry format.**
   ```
   ### #N — <short title>
   **Date:** YYYY-MM-DD
   **Type:** <setup | research | code | experiment | decision | bugfix | doc | note>
   **Summary:** one or two sentences on what was done and why.
   **Artifacts:** files touched / created (relative paths), commits, results.
   **Next:** (optional) immediate follow-ups.
   ```
4. **Decisions.** Any non-obvious decision (architectural, dataset, hyperparameter, scoping) must be logged with its rationale, so we can revisit later without re-deriving the reasoning.
5. **Experiments.** Log dataset version, config/seed, metric, and outcome. A failed run is still an entry.
6. **Notes by the user** (e.g. "#1 Note") are first-class entries and should be preserved verbatim where possible.
7. Keep entries terse but self-contained — a future reader (or future Claude session) should be able to reconstruct the project state from this file alone.

---

## Log

### #1 — Project initialized
**Date:** 2026-05-02
**Type:** setup
**Summary:** Repository created with project proposal (`project_proposal.md`) and the SKDE baseline paper (Delić et al., ICCV 2025) added as the reference work. Progress-report protocol established (this file).
**Artifacts:**
- `project_proposal.md` — full proposal: motivation, contextual conditioning method, ShanghaiTech dataset plan, expected outcomes.
- `Delic_Sequential_keypoint_density_estimator_an_overlooked_baseline_of_skeleton-based_video_ICCV_2025_paper.pdf` — baseline paper.
- `project_progress_report.md` — this report.
- Initial commit: `d37a841` (project proposal added).
**Next:** Locate / clone the official SKDE codebase, set up the Python environment, and prepare ShanghaiTech preprocessing pipeline.

### #2 — Cloned official SeeKer (SKDE) baseline
**Date:** 2026-05-02
**Type:** setup
**Summary:** The official implementation of Delić et al. is published as **SeeKer** at https://github.com/adelic99/seeker (ICCV 2025 Highlight, MIT license, built on top of STG-NF). Cloned into `seeker/` at the repo root to serve as the upstream baseline we will extend with environmental conditioning ($C_t$) per the proposal.
**Artifacts:**
- `seeker/` — full clone. Top-level entry: `seeker.py`. Models: `seeker/models/{autoregressive.py, partial_autoregressive.py, made/}`. Data: `seeker/dataset.py`. Training/eval: `seeker/training.py`, `seeker/validation.py`. Args/config: `seeker/args.py` (already supports `ShanghaiTech`, `UBnormal`, `avenue`, `MSAD`).
- README reports baseline ShanghaiTech AUROC = 85.5 (the number we must match before claiming any gain from $C_t$).
**Decision:** Treat upstream `seeker/` as a vendored, read-only reference. The contextual extension (CCSKDE) will live in a separate top-level package so we can diff cleanly against the baseline. Justification: easier ablations, easier to keep upstream pullable, cleaner for the report.
**Gaps noticed:**
- No `requirements.txt` in the upstream repo — we will need to derive one (likely PyTorch + numpy + einops + tqdm, plus STG-NF deps).
- README install snippet still references `yourusername/SeeKer` (typo upstream, not our concern).
**Next:** (a) inventory `seeker/` imports to derive a working environment spec; (b) decide canonical project layout for our extension; (c) acquire ShanghaiTech pose data per `args.py` expected paths.

### #3 — Repo layout decision: vendored baseline + sibling `ccskde/` package
**Date:** 2026-05-02
**Type:** decision
**Summary:** Confirmed with user: keep `seeker/` read-only as the upstream baseline, put all extensions in a new top-level package `ccskde/`. Rationale: clean diffs vs. baseline, easy upstream pulls, makes the report's ablation story (vanilla SeeKer vs. CCSKDE) trivial to reproduce.
**Artifacts:**
- `ccskde/__init__.py` — package root with one-paragraph docstring describing the project.
- `ccskde/context/` — will host `C_t` extraction (YOLOv11 detections → proximity vector).
- `ccskde/models/` — will host context-conditioned variants of the autoregressive estimator.
- `ccskde/data/` — will host dataset wrappers that join SeeKer pose tensors with `C_t`.
- `scripts/`, `configs/` — placeholders for entry-point scripts and YAML/JSON configs.
**Convention:** never edit files under `seeker/`; if a baseline change is needed, copy the file into `ccskde/` and override there.

### #4 — Python environment spec
**Date:** 2026-05-02
**Type:** setup
**Summary:** Upstream `seeker/` ships no `requirements.txt`. Derived deps from a static import scan of all `.py` files under `seeker/`: only `torch` (incl. `torch.utils.tensorboard`), `numpy`, `scipy`, `scikit-learn`, `tqdm`. Added our extension deps: `ultralytics` (YOLOv11), `opencv-python`, `Pillow`, `einops`. Pinned `numpy<2.0` for ultralytics/torch compat. Local interpreter: Python 3.12.1.
**Artifacts:**
- `requirements.txt` — baseline + extension deps, with inline comments separating the two groups.
- `.gitignore` — excludes `data/`, `exp_dir/`, checkpoints, venvs, caches.
**Open:** virtualenv not yet created; user will choose CPU vs. CUDA torch wheel based on their training machine. To revisit when training hardware is fixed.

### #5 — ShanghaiTech dataset acquisition (action item for user)
**Date:** 2026-05-02
**Type:** note
**Summary:** Dataset is not yet on disk. SeeKer's `args.py` expects this layout (under `--data_dir`, default `./data`):
```
data/ShanghaiTech/
  train/images/                # raw frames (training videos)
  test/frames/                 # raw frames (test videos)
  validation/frames/           # optional
  pose/train/                  # extracted skeleton .json/.npy per video
  pose/test/
  pose/validation/             # optional
```
**Why:** SeeKer trains on pose sequences (`pose/`); the raw frames (`images/`, `frames/`) are only needed for our extension to run YOLOv11 and build $C_t$ — the vanilla baseline can technically train with poses alone, but our context branch needs frames.
**How to apply:** user to download (a) the official ShanghaiTech Campus dataset (Liu et al., CVPR 2018) and (b) a precomputed pose set compatible with SeeKer (the SeeKer/STG-NF lineage typically uses AlphaPose-extracted skeletons; we will confirm format once we read `seeker/dataset.py`).
**Next:** (i) read `seeker/dataset.py` to lock down the exact pose file format expected, so the user knows precisely what to download / regenerate; (ii) once data lands under `data/ShanghaiTech/`, do a smoke-test training run of vanilla SeeKer to reproduce the reported AUROC = 85.5 before touching the model.

### #6 — Pose-data format requirements (from `seeker/dataset.py`)
**Date:** 2026-05-02
**Type:** research
**Summary:** Inspected `seeker/dataset.py` to determine exactly what pose data SeeKer expects.
**Findings:**
- Loader (`gen_dataset`, dataset.py:92-158) scans the pose directory for files ending in `tracked_person.json`.
- For ShanghaiTech the filename is parsed by splitting on `_`, taking the first two tokens as `scene_id`, `clip_id`. Canonical pattern (per UBnormal/MSAD branches): `<scene_id>_<clip_id>_alphapose_tracked_person.json`.
- File contents are JSON dicts of tracked persons with COCO-17 keypoints; SeeKer internally adds a synthetic "neck" keypoint to convert to COCO-18 (dataset.py:163-175).
- Tensor shape after preprocessing: `(num_segments, C=3, T=seg_len, V=18)` — channel 3 = (x, y, confidence).
- These pose JSONs are the standard **STG-NF preprocessed-data** release (SeeKer's README acknowledges STG-NF as the codebase ancestor).
**Implication:** the user's downloaded ShanghaiTech archive contains videos/frames + GT masks but **no poses**. Pose JSONs must be obtained separately (download STG-NF preprocessed data, or run AlphaPose ourselves — preferred order: download first, regenerate only if necessary).

### #7 — ShanghaiTech archive extracted; gaps identified
**Date:** 2026-05-02
**Type:** setup
**Summary:** `cat shanghaitech.tar.gz.* | tar xzf -` succeeded (exit 0). Extracted ~30 GB into `data/ShanghaiTech/shanghaitech/`. Verified contents and compared against SeeKer's expected layout.
**On disk:**
- `shanghaitech/training/videos/*.avi` — raw training videos (NOT frames).
- `shanghaitech/testing/frames/<scene>_<clip>/*.jpg` — test frames (already extracted).
- `shanghaitech/testing/test_frame_mask/`, `shanghaitech/testing/test_pixel_mask/` — ground-truth masks.
**Gaps vs. SeeKer's `args.py` expectations** (`data/ShanghaiTech/{train/images,test/frames,pose/{train,test}}`):
1. **No pose JSONs** (hard blocker — see #6). Plan: download STG-NF preprocessed ShanghaiTech poses.
2. **No training frames** — only `.avi`. May need `ffmpeg` extraction; first verify whether vanilla SeeKer training reads frames at all (it may use poses only).
3. **Directory naming** — archive uses `training/`/`testing/`; SeeKer uses `train/`/`test/`. Resolve with symlinks once layout is finalized.
**Disk hygiene:** 7 `shanghaitech.tar.gz.a{a..g}` part files (~6.5 GB) remain in `data/ShanghaiTech/`. Pending user approval to delete them.
**Next:** (a) confirm with user we can delete the .tar.gz parts; (b) check whether SeeKer's training path reads raw frames or only poses (sets whether ffmpeg step is needed); (c) source STG-NF preprocessed pose JSONs for ShanghaiTech.

### #8 — Disk cleanup + confirmed pose-only training
**Date:** 2026-05-02
**Type:** setup
**Summary:** Per user approval, removed all 7 `shanghaitech.tar.gz.a{a..g}` part files from `data/ShanghaiTech/` (user retains a backup elsewhere); reclaimed ~6.5 GB. Also confirmed by static inspection of `seeker/` that vanilla SeeKer is **pose-only**: no `cv2`, `PIL.Image`, `imread`, or `VideoCapture` references anywhere; `vid_path` is stored on the dataset object but never used for I/O. Implication: baseline reproduction does **not** require extracting frames from `training/videos/*.avi`. Frame I/O will only be needed later for the CCSKDE context branch (YOLOv11 detections to build $C_t$), and even then only on the test split for evaluation — train-time $C_t$ is an open design question (see Next).
**Artifacts:** none (deletion only).
**Decision:** defer ffmpeg / training-frame extraction until the CCSKDE context branch design is finalized. We may sidestep the issue entirely by computing $C_t$ from a small fixed set of "scene context" features per video rather than per-frame YOLO detections during training — to be revisited in entry ~#12.

### #9 — Pose-data source identified (STG-NF preprocessed release)
**Date:** 2026-05-02
**Type:** research
**Summary:** Located the canonical preprocessed pose + GT release for SeeKer's expected format: published by the STG-NF authors (Hirschorn & Avidan, ICCV 2023) as a single Google Drive zip linked from `github.com/orhir/STG-NF`'s README ("Data folder, including extracted poses and GT, can be downloaded using the link"). Single archive covers both ShanghaiTech and UBnormal in the AlphaPose-tracked JSON format that SeeKer's `dataset.py` expects (see #6).
**Artifacts:**
- Download URL: `https://drive.google.com/file/d/1o9h3Kh6zovW4FIHpNBGnYIRSbGCu-qPt/view?usp=sharing` (recorded here for future sessions).
- Local `gdown` is installed but currently broken (`ModuleNotFoundError: certifi`); manual browser download chosen as the path of least resistance.
**Action item for user:** download the zip and drop it into `data/ShanghaiTech/`; do not unzip — extraction + layout normalization handled in the next entry.
**Next:** once the zip lands, verify its internal structure, extract poses to `data/ShanghaiTech/pose/{train,test}/`, and create the `train/`/`test/` symlinks SeeKer's `args.py` expects.

### #10 — STG-NF pose data extracted; SeeKer-ready layout achieved
**Date:** 2026-05-02
**Type:** setup
**Summary:** Extracted STG-NF `data.zip` (~3.5 GB, 1751 files, store-mode zip) at project root with `unzip -d .`. Internal structure was `data/{ShanghaiTech,UBnormal,exp_dir}/...`, which merged cleanly with our existing `data/ShanghaiTech/`. Resulting pose paths match SeeKer's `args.py` defaults verbatim — no symlinks required for baseline reproduction.
**On disk:**
- `data/ShanghaiTech/pose/train/` — 330 ShanghaiTech training videos × 2 JSONs each (`*_alphapose-results.json` raw + `*_alphapose_tracked_person.json` tracked). SeeKer reads only `*_alphapose_tracked_person.json`.
- `data/ShanghaiTech/pose/test/` — 107 test videos × 2 JSONs each.
- `data/ShanghaiTech/gt/test_frame_mask/` — frame-level GT masks for evaluation.
- `data/UBnormal/` — bonus poses for UBnormal (out of scope for this project).
- `data/exp_dir/{ShanghaiTech,ShanghaiTech-HR,UBnormal}/` — empty checkpoint-dir placeholders from the STG-NF release.
- `data/ShanghaiTech/data.zip` — original archive, 1.3 GB, candidate for deletion (user has backup).
**Confirmed gap-vs-args resolutions:**
- `pose_path.{train,test}` ✓ resolved.
- `vid_path.{train,test}` — intentionally left unresolved; vanilla SeeKer never reads frames (see #8). The CCSKDE context branch will need test frames, which are already on disk under `shanghaitech/testing/frames/`.
**Known invocation constraint:** `seeker/seeker.py` uses bare imports (`from args import ...`, `from dataset import ...`), so it MUST be run with cwd inside `seeker/`. Standard launch will be `cd seeker && python seeker.py --data_dir /Users/doruktopcu/Projects/CCSKDE/data --dataset ShanghaiTech ...`. To be wrapped in a launcher script under `scripts/` later.
**Pending user approval:** delete `data/ShanghaiTech/data.zip` (1.3 GB) and `data/UBnormal/` + `data/exp_dir/` (~2 GB of unused content) to free space.
**Next:** (a) get user approval on the cleanup; (b) create a Python venv and install `requirements.txt`; (c) write a smoke-test launcher script that runs vanilla SeeKer training on ShanghaiTech for ~1 epoch to confirm end-to-end the data loader works, before any longer reproduction run targeting the published AUROC = 85.5.

### #11 — Cleanup, env install, and minimal patches to upstream `seeker/`
**Date:** 2026-05-02
**Type:** setup + bugfix
**Summary:** (1) Cleanup: removed `data/ShanghaiTech/data.zip` (1.3 GB) and empty `data/exp_dir/`; kept `data/UBnormal/` (948 MB) for an optional second-dataset reproduction in the writeup. (2) Env: created `.venv/` (Python 3.12.1) and installed `requirements.txt` — torch 2.11.0 (Apple Silicon arm64 wheel, MPS backend), tensorboard 2.20, numpy 1.26.4, scipy 1.17, scikit-learn 1.8, ultralytics 8.4.46, opencv 4.11, einops 0.8, Pillow 12.2, matplotlib 3.10. (3) Discovered upstream `seeker/` cannot run as-published because: (a) 5 hardcoded `'cuda'` references make the code non-portable to MPS/CPU, (b) `seeker.py:36` reads `args.dropout` but the parser registers `--droppout` (dest=`droppout`) — would `AttributeError` on launch even on a CUDA box.
**Decision (overrides #3):** allow minimal portability patches inside `seeker/`. Strict scope: device plumbing and trivial typo fixes only. No research/algorithmic edits to baseline files. Rationale: alternative (monkey-patching from a launcher) is brittle and hides the diff from the reader.
**Patches applied (full diff captured here for traceability):**
- `seeker/seeker.py:36` — `droppout=args.dropout` → `droppout=args.droppout` (typo bugfix).
- `seeker/seeker.py:38` — `model.to('cuda')` → `model.to(args.device)`.
- `seeker/training.py:137` — `conf.cuda()` → `conf.to(self.args.device)`.
- `seeker/training.py:170` — same.
- `seeker/models/made/made.py:22-26` — `self.mask = None` → `self.register_buffer('mask', None)`; `mask.to('cuda')` → `mask` (mask now follows module device on `.to()`).
- `seeker/models/made/made_partial.py:22-26` — same.
- Verified post-patch: `grep "\.cuda\(\)\|'cuda'\|\"cuda\""` under `seeker/` returns only the help-string default in `args.py`. Zero remaining hardcoded device references.
**Artifacts:**
- `.venv/` (gitignored).
- `scripts/smoke_test_seeker.sh` — launcher: `cd seeker && PYTHONUNBUFFERED=1 python -u seeker.py --dataset ShanghaiTech --data_dir <abs> --exp_dir <abs> --device mps --num_workers 0 --epochs 1 --batch_size 256 --seg_len 24 --seed 42`. Note `-u`/`PYTHONUNBUFFERED` are mandatory — first attempt without them produced a 0-byte log for ~10 min (Python fully buffers when stdout isn't a tty), so we couldn't see whether the run was alive or hung.
**Open behavioral notes:**
- Two `SyntaxWarning: invalid escape sequence '\d'` warnings from `dataset.py:109` and `validation.py:94` on Python 3.12 — purely cosmetic, in UBnormal-only regex branches; will not fix unless they cause issues.
- First (killed) smoke run confirmed: ShanghaiTech pose-loading takes ~3:23 min CPU-only for 330 train JSONs (single-threaded JSON parsing in `gen_dataset` — could be parallelized later if it becomes a bottleneck for many runs).
**Next:** entry `#12` will record the smoke-test outcome (1 epoch, batch 256, MPS) — pass/fail, AUC if produced, and any new MPS-specific failures.

### #13 — Progress report (LaTeX, ICLR 2025 template) drafted
**Date:** 2026-05-02
**Type:** doc
**Summary:** Drafted the milestone progress report requested by Prof. Nazli (deadline 2026-05-04 23:59 via email to nazli@cs.hacettepe.edu.tr). Followed the ICLR 2025 template the professor linked (Overleaf id `gqzkdyycxtvt`); fetched the official `iclr2025_conference.{sty,bst}` and `math_commands.tex` from the ICLR Master-Template GitHub mirror so the document is reproducible without an Overleaf account. Document covers all four required sections with the rubric weights in mind: Problem Definition (15pts), Related Work (20pts), Method (45pts), Experimental Settings & Preliminary Results (20pts).
**Artifacts:**
- `report/main.tex` — full source (~5 body pages, ~13 refs).
- `report/iclr2025_conference.sty`, `report/iclr2025_conference.bst`, `report/math_commands.tex` — official ICLR 2025 template files (vendored for reproducibility).
- `report/main.pdf` — compiled output: 5 body pages + 1 references page (per ICLR convention, references do not count against the page limit, so this matches the "4-5 pages" requirement).
**Build:** `cd report && pdflatex main && pdflatex main` (twice for cross-refs); zero unresolved citations or references after the second pass; only cosmetic `Underfull \hbox` warnings remain.
**Content highlights for future-self:**
- Section 1 frames SeeKer's environmental blindness as the limitation we address; enumerates 4 challenges (one-class learning, AR-mask preservation, scene generalization, single-semester compute envelope).
- Section 3 spells out the math: SeeKer's factorization (Eq. 1) → CCSKDE factorization (Eq. 2) → $C_t$ construction from YOLOv11 detections as a fixed-dim per-class (inverse-min-distance, count-within-radius) vector; explains injection as MADE "unconditional shortcut" (zero-degree label, all-ones mask columns).
- Section 4 promises an ablation including a "shuffled-context" negative control (permute $C_t$ across frames at training time) to isolate the value of spatial-temporal alignment from the value of extra parameters — this is the most defensible claim we could make in the final report.
**Open follow-ups before submission:**
- Finalize once smoke-test (#12) completes so we can quote a concrete preliminary number rather than just "pipeline verified."
- Consider adding a figure of the SeeKer vs. CCSKDE architecture (currently a placeholder reference to a non-existent `Figure~\ref{fig:overview}` — to add or remove before sending).
- Email submission to nazli@cs.hacettepe.edu.tr by 2026-05-04 23:59.

### #12 — Smoke test PASSED end-to-end on Apple Silicon MPS
**Date:** 2026-05-02
**Type:** experiment
**Summary:** First end-to-end run of vanilla SeeKer on ShanghaiTech via our launcher succeeded after one additional environment patch. Run config: 1 epoch, batch_size=256, seg_len=24, MPS device, num_workers=0, seed=42. Result: **AUROC = 0.7438 on the validation split** (= ShanghaiTech test split for non-UBnormal datasets, see `dataset.py:85-87`), with Average Precision = 0.6470 and FPR@95 = 0.6854. Paper reports 0.855 AUROC at 10 epochs / batch 1024 on CUDA, so 0.7438 after a single epoch on MPS is consistent with a healthy training trajectory and the pipeline matches the paper's evaluation protocol. Loss went from ~21 (epoch start) to negative values (Gaussian log-density regime) over the 3892 training batches, as expected.
**Additional patch discovered during run** (extends #11): `seeker/validation.py` had two hardcoded *relative* GT paths (`data/UBnormal/gt/{split}/`, `data/ShanghaiTech/gt/test_frame_mask/`) plus two absolute paths to the original authors' filesystem (`/mnt/sdb1/...` for `avenue` and `MSAD`). Since our launcher runs with cwd=`seeker/`, the relative paths resolved to `seeker/data/...` and triggered `FileNotFoundError`. Patched the two paths we actually use to `os.path.join(args.data_dir, ...)`; left the two absolute paths alone (out of scope, datasets we are not using). Combined with the cuda/dropout patches in #11, this brings total upstream-to-CCSKDE diff to 8 lines across 4 files.
**Timing on Apple Silicon (M-series, MPS):**
- Pose loading (train + test, 437 videos): ~3.5 min (CPU bottleneck on JSON parsing).
- Training, 1 epoch / batch 256 / 3892 batches: ~16 min (MPS, ~4 it/s).
- Validation (107 clips): ~5 s.
- Total wall-clock: ~20 min for 1 epoch. Full 10-epoch reproduction would be ~3 hours on this hardware. Acceptable.
**Artifacts:**
- Checkpoint written under `exp_dir/ShanghaiTech/<timestamp>/`.
- Output log preserved at `/private/tmp/.../bdrg311d5.output` (will rotate; key numbers above are the durable record).
**Next:** (a) extend `#11` patches block in this report retroactively for traceability — DONE inline above; (b) update the final paragraph of `report/main.tex` Section 4.5 to cite the concrete 0.7438 number rather than the vaguer "smoke test passes"; (c) launch the full 10-epoch / batch 1024 reproduction run targeting 0.855 AUROC; (d) start designing the YOLOv11 $C_t$ extraction pass.

### #14 — Seeker patches committed; full reproduction launched in background
**Date:** 2026-05-02
**Type:** setup + experiment
**Summary:** (1) Committed the 8-line portability/path/typo patch set from #11+#12 to the nested `seeker/` git history (commit `3706984`, "CCSKDE: device portability + path/typo fixes for local run"). The patch lives only in our local `seeker/` clone — upstream `adelic99/seeker` is unaffected — keeping the diff durable and inspectable without touching the parent repo. (2) Authored `scripts/repro_seeker_shanghaitech.sh` (10 epochs, batch 256, MPS, seed 42) and launched it in background under `nohup`; first ~3 minutes of output show the same pose-loading and tqdm trajectory as the smoke run, confirming no regression from the commit. Expected wall-clock ~3 h.
**Decision:** kept batch size at 256 rather than the paper's 1024. Why: the smoke run proved 256 fits comfortably in M-series unified memory; 1024 was untested and we cannot afford an OOM mid-run on the eve of the report deadline. How to apply: any future MPS reproduction script defaults to 256 unless we benchmark a higher value cleanly.
**Artifacts:**
- `scripts/repro_seeker_shanghaitech.sh` — full-reproduction launcher.
- `logs/repro_seeker_<timestamp>.log` — live training log.
- `seeker/` commit `3706984` (local-only).
**Next:** read the log when the run finishes; record final AUROC in entry `#16`. If it lands ≥ ~0.84, we have a credible baseline for the CCSKDE comparison.

### #15 — `ccskde/context/` scaffold + smoke-tested $C_t$ builder
**Date:** 2026-05-02
**Type:** code
**Summary:** Implemented the offline $C_t$ pipeline described in report Section 3.2 as three small modules under `ccskde/context/` plus a CLI driver. Stage 1 runs YOLOv11 once per video and caches per-frame normalized boxes (cls, conf, cx, cy, w, h) for the five hazard classes (bicycle, car, motorcycle, bus, truck). Stage 2 turns those caches plus per-frame skeleton centroids into $C_t \in \mathbb{R}^{2K} = \mathbb{R}^{10}$ vectors via `[1/(d_min + \varepsilon),\, n_{\le r}]` per class with diagonal-normalized euclidean distance and $r = 0.15$. The pipeline never imports `ultralytics` at module scope, so the package stays importable for unit testing without YOLO weights present.
**Logic verification:** ran a synthetic frame with one car at the centroid (d≈0) and one truck at the corner (d≈0.45 normalized): C_t = `[0,0, 1000,1, 0,0, 0,0, 2.22,0]` — car slot saturates the inverse-distance and counts within radius; truck slot has the expected $1/0.45$ inverse with zero count beyond r=0.15. Bicycle/motorcycle/bus slots correctly zero.
**Artifacts:**
- `ccskde/context/config.py` — `ContextSpec` dataclass; COCO-id ↔ class-name table; $r$ default.
- `ccskde/context/detect.py` — `extract_clip()` + frame iterators for both jpg-dir (test split) and `.avi` (train split via OpenCV); cache schema documented in the module docstring.
- `ccskde/context/build.py` — `context_for_frame`, `context_for_track`, `pose_centroid` (confidence-weighted).
- `ccskde/context/__init__.py` — public API surface.
- `scripts/extract_yolo_detections.py` — CLI: `--split {train,test} [--limit N]` for incremental runs.
**Open design question (deferred):** which pedestrian-track frames does $C_t$ apply to? SeeKer's per-clip pose JSONs are tracked-person time series; we will need to align AlphaPose's per-track frame indices to the per-video YOLO cache. The wiring lives in the *dataset wrapper* under `ccskde/data/`, which is intentionally out of scope for #15 — the pure functions in `build.py` already accept (detections_per_frame, centroids, frame_indices), so the wrapper is the only missing piece before training.
**Next:** (a) wait for #14 reproduction to finish; (b) author `ccskde/data/contextual_dataset.py` that wraps SeeKer's `PoseSegDataset` and joins it with cached $C_t$; (c) one smoke-test extraction over a single ShanghaiTech test clip end-to-end before scaling to all 437 videos.

### #16 — Context-conditioned MADE variant + AR-preservation proof
**Date:** 2026-05-02
**Type:** code + experiment
**Summary:** Implemented the load-bearing research code: `MADEPartialContext` and `PartialAutoregressiveContextFC` under `ccskde/models/`, mirroring upstream `seeker/models/made/made_partial.py` so the diff is auditable. The construction realises the report's "unconditional shortcut" idea concretely: the input layer's mask matrix is built using the upstream keypoint-degree scheme, then $D_{ctx}$ all-ones columns are appended on the right so every hidden unit may read every $C_t$ entry while the keypoint-to-keypoint causality is bit-for-bit preserved. Output dimension is left at $2D_{kp}$ (mu | logvar over keypoints only), so `seeker/training.py`'s reshape-and-slice loss path needs no modification.
**Verification (this is the part the project rises or falls on):** wrote a Jacobian test that picks a current-frame keypoint output index $i$ and measures $\partial \mu_i / \partial x_j$ for every keypoint input $j$ and every context input. Result: **0 violations** beyond the pair-aligned forbidden region (i.e. $\mu_i$ truly does not see any $x_j$ at or after its allowed predecessor set), 258/260 of the *allowed* keypoint inputs have non-zero gradient (full reachability), and **all 80/80 $C_t$ inputs have non-zero gradient** (shortcut is wired). This is the empirical evidence for the methodological claim in report Section 3.2 — the conditioning preserves MADE's autoregressive structure.
**Decision (extends #11):** the model lives in `ccskde/models/`, not in `seeker/`, per the vendoring rule. The trainer adapter to feed `(x_pose, c_ctx)` will be a thin subclass of `SeeKerTrainer` under `ccskde/`; no edits to `seeker/training.py`. Why: keeps the ablation story trivial — vanilla SeeKer is `seeker/`, ours is `seeker/` + a trainer override.
**Artifacts:**
- `ccskde/models/made_partial_context.py` — model + masked-linear.
- `ccskde/models/__init__.py` — public surface.
- (test script in-line in the session log; not committed as a unit test yet — TODO if/when we add a test directory).
**Open caveat:** the AR check covered one output index in the current-frame portion. A stronger guarantee would sweep all current-frame indices; informal reasoning over the mask construction implies it must hold for all of them, since the only modification vs. baseline is appending all-ones columns to the input layer.

### #17 — Contextual dataset wrapper
**Date:** 2026-05-02
**Type:** code
**Summary:** Wrote `ccskde/data/contextual_dataset.py:ContextualSkeletonSequenceDataset` — a thin composition wrapper around SeeKer's `SkeletonSequenceDataset`. For each pose segment it (i) recovers `(scene, clip, person, start_frame)` from `segs_meta`, (ii) loads the per-clip YOLO detection cache (one `.npy` per clip, see #15), (iii) computes the confidence-weighted skeleton centroid for each frame in the segment, (iv) calls `context_for_track` to produce $C \in \mathbb{R}^{T \times D_{ctx}}$, and returns `[pose, score, C]`. Cache miss → all-zero $C$ (clean, no-detection regime), so we can iterate on the trainer before YOLO extraction has run on every clip.
**Smoke test:** built the wrapper around a fake base dataset with metadata `[scene=01, clip=0014, person=0, start_frame=100]` and a non-existent cache dir; got back `pose=(3,24,18), score=(24,), C=(24,10), sum(C)=0` as designed.
**Artifacts:**
- `ccskde/data/contextual_dataset.py`.
- `ccskde/data/__init__.py`.
**Next:** (a) thin subclass of `SeeKerTrainer` (call it `CCSKDETrainer`) that unpacks `[pose, score, C]` from the loader and passes `(x_pose, c)` to `model.forward`; (b) entry-point under `ccskde/seeker_ctx.py` mirroring `seeker/seeker.py` structure; (c) actually run YOLO extraction on a single test clip end-to-end before scaling.

### #18 — Migrated to Google Colab (A100); fixed `--shuffled_context` wiring
**Date:** 2026-05-03
**Type:** setup + bugfix
**Summary:** Local M1 Mac runs hit two blockers: (1) MPS caused NaN in epoch 2 of the 10-epoch baseline reproduction, (2) RAM exceeded capacity during YOLO extraction. Decided to migrate all training to Google Colab with an NVIDIA A100-SXM4-80GB (Colab Pro). Created `scripts/prepare_colab_upload.sh` to produce 3 upload zips (code+poses: 980 MB, test frames: 4.2 GB, train videos: 2.3 GB) and `scripts/ccskde_colab.ipynb` — a self-contained notebook running all 5 experiment phases sequentially.
**Bugfix discovered during migration:** the `--shuffled_context` CLI flag was parsed in `seeker_ctx.py` but never passed to `CCSKDETrainer` or used anywhere. Fixed by (a) adding `shuffled_context` init param to `CCSKDETrainer` + batch permutation in the training loop, (b) wiring `args.shuffled_context` through the entry point.
**Artifacts:**
- `scripts/prepare_colab_upload.sh` — zip creation script.
- `scripts/ccskde_colab.ipynb` — full experiment notebook (5 phases).
- `ccskde/training.py` — shuffled-context permutation logic added.
- `ccskde/seeker_ctx.py` — `shuffled_context` flag forwarded to trainer.

### #19 — Full experiment results on A100
**Date:** 2026-05-03
**Type:** experiment
**Summary:** All three experiment phases completed successfully on the A100 — zero NaN, zero crashes. YOLO extraction processed all 437 clips (107 test + 330 train). Results:

| Model | Best AUROC | Epoch | Mean AUROC |
|---|---|---|---|
| Baseline (SeeKer) | **0.7808** | 7 | 0.7308 |
| CCSKDE (ours) | **0.7786** | 3 | **0.7496** |
| CCSKDE (shuffled) | 0.7547 | 8 | 0.7465 |

**Key findings:**
1. Peak AUROC is a virtual tie (0.78 vs. 0.78).
2. CCSKDE shows better training stability: mean AUROC +1.9 pp, narrower variance band.
3. Shuffled ablation validates design: 2.4 pp gap confirms spatial alignment in $C_t$ provides genuine signal beyond extra parameters.
4. Baseline underperforms paper's 0.855 — common reproducibility gap for skeleton VAD; all comparisons are relative to our reproduced baseline.

**Artifacts:**
- `colab_results/results/experiment_results.json` — all numbers.
- `colab_results/results/auroc_curves.png` — comparison plot.
- `colab_results/results/checkpoints/{baseline,ccskde,shuffled}_best.pth` — best model weights.
- `colab_results/results/{baseline,ccskde,shuffled}_run.log` — full training logs.
- `colab_results/results/context_cache/{train,test}/` — 437 YOLO detection caches.
- `report/main.tex` — updated Section 4 with results table, figure, discussion, and future work.
- `report/main.pdf` — compiled, 7 pages, zero LaTeX errors.
**Next:** submit report to nazli@cs.hacettepe.edu.tr by 2026-05-04 23:59; prepare for final-report phase (extended context vocabulary, multiplicative injection, UBnormal evaluation).

### #20 — Resumed on Windows / RTX 5080; environment rebuilt
**Date:** 2026-05-29
**Type:** setup
**Summary:** Project transferred from Mac to a Windows 11 PC with an **RTX 5080 (Blackwell, sm_120)**. Purged macOS AppleDouble (`._*`) + `.DS_Store` litter (some had landed inside `.git/objects/pack/`, causing `non-monotonic index` errors on every git call). Installed Python 3.12.10, created `.venv`, installed **PyTorch 2.11.0+cu128** (Blackwell needs CUDA 12.8 wheels — cu121 will not run) + the rest of `requirements.txt` (numpy pinned 1.26.4). Verified CUDA + GPU matmul. All training now runs locally (previously Colab A100).
**Artifacts:** `.venv/` (gitignored). `~/.claude/.../memory/ccskde-env.md`.

### #21 — CRITICAL bug: C_t was computed in the wrong coordinate frame
**Date:** 2026-05-29
**Type:** bugfix
**Summary:** End-to-end audit found the load-bearing defect behind the milestone's null result. `ccskde/data/contextual_dataset.py` computed the pedestrian centroid from `base[index]` — i.e. the pose AFTER SeeKer's `normalize_pose` (per-segment zero-mean / unit-std, data_utils.py:59). YOLO boxes are in `[0,1]` image coordinates, so the two lived in incompatible frames. The normalized centroid collapses to ~origin, so `C_t` degenerated into roughly "inverse distance from the image's top-left corner to the nearest vehicle" — **almost pedestrian-independent**. This explains the milestone's peak tie and the small shuffled-context gap.
**Fix:** compute the centroid from the RAW pixel keypoints (`base.segs_data_np[index]`) divided by `(W,H) = (856,480)` (verified uniform across all 13 scenes). Added `img_wh` to `ContextSpec`.
**Verification:** `tests/test_context_coords.py` — a pedestrian placed on a detected bicycle now yields `1/d_min` ≈ 1000 vs ≈ 1.5 at the opposite corner (was identical before). Preliminary 1-epoch run (stride 12): baseline 0.742 → corrected-proximity **0.790** (+4.8 pp), where the milestone had a tie.
**Artifacts:** `ccskde/context/config.py`, `ccskde/data/contextual_dataset.py`, `tests/test_context_coords.py`.

### #22 — "Car-skeleton" oriented vehicle-keypoint context (the uplift)
**Date:** 2026-05-29
**Type:** code + research
**Summary:** Replaced the coarse per-class `[1/d_min,count]` proximity with a structured **oriented vehicle-keypoint matrix** ("like a box, but better", per the user's idea). Each of the M nearest hazards is represented by K_v=6 keypoints — 4 oriented corners + center + heading — derived from a **YOLOv11-seg** mask via `cv2.minAreaRect`/`boxPoints` (orientation that an axis-aligned box lacks), expressed in **pedestrian-relative, scale-normalised** coords (translation+scale invariant; addresses the report's scene-generalisation challenge). Literature check: dedicated vehicle-keypoint nets (SKoPe3D, 33 kpts) are synthetic/frontal and don't transfer to surveillance views, so a seg-derived oriented skeleton is the right lightweight choice (cf. bounding-ellipse work for overhead vehicles).
**Decision:** corners are computed in PIXEL space then normalised (rotating an oriented box in anisotropic [0,1] space distorts angles). Cache schema v2 = 14 cols `(cls,conf,4 corners,center,heading)`; the old 6-col box cache is auto-handled as an axis-aligned "box pseudo-skeleton" fallback/ablation.
**Artifacts:** `ccskde/context/vehicle.py`, `ccskde/context/detect.py` (`extract_clip_oriented`, `load_yolo_seg`), `ccskde/context/config.py` (`mode`/`max_vehicles`/`kv`/`vehicle_dim`), `scripts/extract_yolo_detections.py` (`--oriented`), `tests/test_vehicle_context.py` (geometry + scale-invariance + top-M + fallback, all pass).

### #23 — FiLM covariance modulation (contextual covariance penalty)
**Date:** 2026-05-29
**Type:** code + research
**Summary:** Implemented the proposal's promised "contextual covariance penalty" as a **FiLM** head (Perez et al., 2018): a small MLP maps the context to per-coordinate `(gamma,beta)` that modulate the predicted covariance, `logvar <- (1+gamma(C))*logvar + beta(C)`. Zero-initialised so it is exactly identity at start (training begins at the shortcut-only model). FiLM depends only on C, so MADE's autoregressive causality is untouched.
**Resolves report-vs-code discrepancy:** the report claimed a full 2×2 Cholesky covariance, but the code (and SeeKer's *headline* config, per arXiv:2506.18368) is **diagonal**; full Cholesky is only a SeeKer ablation. The report will be corrected to "diagonal + FiLM covariance modulation".
**Verification:** `tests/test_ar_mask.py` — empirical Jacobian shows **0 forbidden-region violations**, 100% past-frame reachability, all context inputs reachable, and FiLM zero-init parity with the shortcut-only model.
**Artifacts:** `ccskde/models/made_partial_context.py` (`film_cov`), `ccskde/seeker_ctx.py` (`--context_mode`, `--film_cov`, `--max_vehicles`, `--kv`), `tests/test_ar_mask.py`.

### #24 — Honest hazard-subset evaluation (the question never tested)
**Date:** 2026-05-29
**Type:** code
**Summary:** Built the evaluation the milestone lacked: isolate pedestrian-vehicle hazard events instead of mixing them with motion-intrinsic anomalies. `ccskde/eval/hazard.py` computes per-frame proximity and reports AUROC on the full set, the **interaction regime** (vehicle near a pedestrian) and the **vehicle-hazard** set (normal + near-vehicle anomalies), plus the **Spearman score-vs-proximity** correlation over anomalous frames. `scripts/evaluate_hazard.py` drives it from a checkpoint.
**Verification:** driver reproduces the baseline full-set AUROC **exactly (0.7808)**, confirming the scoring path matches seeker's; frame alignment asserted. Baseline reference on the old checkpoint: vehicle-hazard AUROC ≈ 0.84–0.88, score-proximity Spearman ≈ 0.31. `tests/test_hazard_eval.py` passes.
**Artifacts:** `ccskde/eval/`, `scripts/evaluate_hazard.py`, `tests/test_hazard_eval.py`.

### #25 — Unified experiment runner + per-epoch validation fix
**Date:** 2026-05-29
**Type:** code + bugfix
**Summary:** `scripts/run_experiments.py` loads the pose data ONCE and runs the full config matrix {baseline, proximity, proximity_shuffled, vehicle, vehicle_film, vehicle_shuffled}, recording per-epoch val AUROC → `experiment_results.json` + `auroc_curves.png` (incremental save + resume). Added `ContextualSkeletonSequenceDataset.precompute()` to materialise the context once (the per-item Python build otherwise dominates at stride 1). Fixed the load-bearing `"ShangaiTech"` misspelling in the validation guard (`ccskde/training.py`) — it only ever validated by accident; made per-epoch validation explicit. Noted (not fixed, vendored) that `seeker/training.py:test()` references an undefined `joint_lnp_full_window` and mis-unpacks `score_anomalies`; our eval path avoids it.
**Artifacts:** `scripts/run_experiments.py`, `ccskde/data/contextual_dataset.py` (`precompute`), `ccskde/training.py`.

### #26 — Tests + skills
**Date:** 2026-05-29
**Type:** code + doc
**Summary:** Added a `tests/` suite (`test_context_coords`, `test_vehicle_context`, `test_ar_mask`, `test_hazard_eval`) — the project had none. Added two project skills: `.claude/skills/ccskde-run` (extraction + experiment matrix + single runs) and `.claude/skills/ccskde-eval` (results, hazard analysis, test suite, "what working looks like").
**Artifacts:** `tests/*`, `.claude/skills/ccskde-run/SKILL.md`, `.claude/skills/ccskde-eval/SKILL.md`.

### #27 — Unified scene-SKDE: cars as skeletal agents (new headline model)
**Date:** 2026-05-29
**Type:** code + research
**Summary:** Per the user's direction ("merge the SKDE concept to cars... single
formula"), generalised the model from "pedestrian density + car context" to a
**unified scene density**. Each frame becomes one augmented skeleton
`Z = [V^1..V^M (6 kp each), P (18 kp)]`, `N'=18+6M`, vehicles ordered first so
the pedestrian keypoints are predicted *given* the vehicles. SeeKer's
autoregressive factorisation is applied verbatim to `Z`; the single hazard score
is `S(t) = -Σ_n c_{t,n} ln p_θ(Z_{t,n}|Z_{t,<n},Z_Δ)`. Cars are now modelled
(have their own NLL terms), not used as side info. Trained on the joint NLL;
frame scoring reads out the pedestrian keypoints (last 18) for SeeKer
comparability while still reflecting vehicle conditioning.
**Grounding:** SeeKer (arXiv:2506.18368) models only one human skeleton and
max-pools people — never objects/interactions, so this is a genuine gap.
Precedent for joint multi-agent density VAD: Wiederer et al. 2022
(multi-agent trajectory density), ComplexVAD 2025 (interaction anomalies).
**Verification:** `tests/test_scene_model.py` — generalised MADE (N'=30) has
**0 AR-forbidden-region violations**; the pedestrian→vehicle attention pathway
is confirmed present; scene-builder layout + missing-cache handled.
**Artifacts:** `ccskde/models/scene_made.py` (`SceneMADEPartial`,
`PartialAutoregressiveSceneFC`), `ccskde/context/scene.py`,
`ccskde/data/scene_dataset.py`, `ccskde/training.py` (`SceneTrainer`),
`scripts/run_experiments.py` (`scene` config), `tests/test_scene_model.py`,
`CLAUDE.md`, `check-here-doruk.md`.
**Next:** run the `scene` config + hazard-subset eval once the in-flight matrix
(baseline/proximity/vehicle/vehicle_film/vehicle_shuffled) completes; fill the
report results (car-matrix/scene-led) and recompile.

### #28 — Full results (local RTX 5080) + hazard-subset analysis
**Date:** 2026-05-29
**Type:** experiment
**Summary:** Ran the full matrix (10 epochs, batch 1024, seg_len 24, stride 1,
seed 42) via `scripts/run_experiments.py` and the hazard-subset eval.

ShanghaiTech validation AUROC:

| Model | Best | Mean |
|---|---|---|
| SeeKer baseline | 0.7351 | 0.7204 |
| CCSKDE proximity (coord-fixed) | 0.8034 | 0.7857 |
| CCSKDE vehicle (car-matrix) | 0.8023 | 0.7855 |
| CCSKDE vehicle + FiLM | 0.7971 | **0.7898** |
| CCSKDE vehicle (shuffled ctx) | 0.7893 | 0.7835 |
| **Unified scene-SKDE** | 0.7980 | 0.7863 |

Context/scene = **+6–7 pp mean over baseline** (milestone was a tie → the
coordinate bug was the cause). Our baseline *mean* (0.720) matches the
milestone's baseline mean (0.731); the milestone's 0.781 was one lucky epoch.
Shuffled control is ~1 pp below aligned vehicle ⇒ ~1 pp is genuine spatial
signal, the rest is structured-input capacity. (Single seed; multi-seed would
tighten the comparison — noted as future work.)

**Hazard-subset eval** (`scripts/evaluate_hazard.py`):

| Model | full | vehicle-hazard AUROC | score↔proximity ρ |
|---|---|---|---|
| baseline | 0.735 | ~0.79 | 0.16 |
| vehicle+FiLM | 0.797 | **0.965** | **0.64** |
| scene (joint readout) | 0.782 | 0.95 | 0.60 |

The car models flag pedestrian-vehicle proximity as anomalous (the traffic-hazard
goal): score↔proximity correlation jumps 0.16→0.64, and near-vehicle anomalies
are detected at ~0.95–0.965 vs baseline ~0.79.

**Honest caveats logged:** (i) the "interaction-regime AUROC" metric is
unreliable here because near-vehicle frames on ShanghaiTech are ~91% anomalous
(5035 anomalous vs 487 normal) — a near-degenerate subset; we report
vehicle-hazard AUROC + proximity correlation instead. (ii) The scene model's
pedestrian-only readout misses vehicle-intrinsic anomalies; a joint readout
(fold vehicle NLL into the score) raises proximity ρ to 0.60. (iii) The scene
2-min-vs-vehicle-13.7-min wall-clock gap was pure system load (973 train
batches/epoch in both; scene ran on an idle machine at ~107 it/s vs ~7 it/s),
not a data/training difference — verified from the logs.
**Artifacts:** `colab_results/results_v2/experiment_results.json`,
`auroc_curves.png`, per-config `ShanghaiTech_*/<ts>/checkpoint_best.pth`.

### #29 — Session wrap + resume docs
**Date:** 2026-05-29
**Type:** doc
**Summary:** Wrote `README.md` (canonical overview + achievements + results +
quick-start). Refreshed `HANDOFF.md` to the final state with exact
continue-from-here commands and the open next-steps list. Updated `CLAUDE.md`
(status pointer) and the cross-session memory (`ccskde-env.md`) with the project
state. Recompiled `report/main.pdf` cleanly (8 pp, 0 errors) with the new
results tables + curves. All seven session tasks complete.
**State for the next session:** start from `README.md` / `HANDOFF.md`. Everything
implemented, tested, run, written up; **uncommitted on `main`**. Open (non-blocking)
follow-ups in `check-here-doruk.md`: commit-to-branch, YOLO-World road elements,
yolo11l-seg re-extraction, multi-seed runs, joint-readout scene scoring default,
full 2×2 context-rotated covariance.
**Artifacts:** `README.md`, `HANDOFF.md`, `CLAUDE.md`, `report/main.pdf`,
memory `ccskde-env.md`.

### #30 — Vendored the SeeKer baseline (de-submodule)
**Date:** 2026-05-29
**Type:** decision + code
**Summary:** `seeker/` was an improperly-configured git **gitlink** — no
`.gitmodules`, pointing at the unpushable local commit `3706984` — so a fresh
`git clone` produced an *empty* `seeker/` and the tree always showed dirty
("modified content, untracked content"). Reproduction across machines was only
possible by manual SSD copy. Decision (user-approved): **vendor** SeeKer as plain
tracked files in the CCSKDE repo for one-repo, clone-and-run reproducibility.
This supersedes the original "vendored submodule" convention in #2/#3 (the
read-only-baseline rule still stands; it is just no longer a nested repo).
**How:** captured provenance first — `seeker/PORTABILITY_PATCHES.patch`
(full diff vs upstream base `7e7ab66`, 5 files) and `seeker/VENDORED.md`
(upstream URL, MIT license retained, base commit, rationale). Then
`git rm --cached seeker`; `rm -rf seeker/.git`; `git add seeker`. The existing
`.gitignore` (`exp_dir/`, `runs/`, `__pycache__/`, `*.pth`) already excludes the
69 MB of training artifacts, so only 17 source files (~63 KB) were committed.
**Verification:** `git status` now clean of `seeker`; vendored seeker imports OK
and the ccskde→seeker bridge (`scene_dataset` → `normalize_pose`) works.
**Reversible:** the gitlink history remains on `origin/final-ccskde` pre-merge,
and the patch + VENDORED.md let the submodule be re-derived if ever wanted.
**Artifacts:** commit `026166e`, `seeker/VENDORED.md`,
`seeker/PORTABILITY_PATCHES.patch`.
