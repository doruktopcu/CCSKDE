# UBnormal cross-dataset setup (ours vs SeeKer)

Goal: reproduce a SeeKer-style pose-density **baseline** on UBnormal and compare
our **vehicle-augmented** model (car-matrix context + Scene-SKDE) on the same
inputs. UBnormal is synthetic and SeeKer-comparable; some scenes are streets with
**cars that are normal**, so the vehicle subset tests the interaction signal.

## What SeeKer's code already gives us
`seeker/` supports `--dataset UBnormal` (`args.py`, `dataset.py`). Expected layout:
```
data/UBnormal/
  pose/{train,validation,test}/   *_alphapose_tracked_person.json
  train/images/  validation/frames/  test/frames/     # RGB
```
Pose filenames are parsed as
`(abnormal|normal)_scene_<scene>_scenario<...>_alphapose_*.json`.

## The honest design choice (recommended): one perception stack we control
SeeKer does **not** publish its UBnormal AlphaPose files, and hunting third-party
pose releases (STG-NF/MoCoDAD) means a format converter + a pose mismatch. Cleaner
and fully self-consistent:

1. Download **UBnormal RGB + frame-level GT** (official repo, Google Drive).
2. Run **YOLO11-pose** (pedestrians) → SeeKer-format pose JSONs, and
   **YOLO11-seg** (vehicles) → our oriented car-matrix cache, on the same frames.
3. Train **baseline** (pose-only density) and **ours** (vehicle/scene) on these
   identical poses → an apples-to-apples ours-vs-baseline comparison.

Trade-off: numbers won't match SeeKer's published 77.9 exactly (different pose
extractor), but the *relative* ours-vs-baseline claim is clean — and it mirrors
how we already treat ShanghaiTech. We state this clearly in the report.

## Download targets (verify exact sizes on the page before pulling)
- **RGB + annotations:** UBnormal official repo
  https://github.com/lilygeorgescu/UBnormal (Google Drive:
  `https://drive.google.com/file/d/1KbfdyasribAMbbKoBU1iywAhtoAt9QI0`). Contains
  the rendered videos/frames + frame-level abnormal annotations. Size: on the
  order of tens of GB (confirm on Drive). License: research, CC-BY-NC-SA.
- (Optional comparability) third-party UBnormal poses from STG-NF
  https://github.com/orhir/STG-NF or MoCoDAD — only if we want SeeKer-exact poses.

## Commands (once data is on disk)
```bat
set PYTHONPATH=%CD%
:: 1) perception (offline, one-time): poses + oriented vehicles for UBnormal
.venv\Scripts\python scripts\extract_yolo_detections.py --data-root data\UBnormal --split train --oriented
.venv\Scripts\python scripts\extract_yolo_detections.py --data-root data\UBnormal --split test  --oriented
:: (+ a YOLO11-pose pass writing seeker-format pose JSONs -> data\UBnormal\pose\)

:: 2) baseline vs ours (one-class, train on normal)
.venv\Scripts\python scripts\run_experiments.py --configs baseline vehicle scene --dataset UBnormal --out colab_results\results_ubnormal --seed 42
:: 3) hazard / cross-dataset metrics
.venv\Scripts\python scripts\evaluate_hazard.py --checkpoint <ckpt> --dataset UBnormal --context_mode vehicle ...
```

## Code changes — status
1. ✅ **DONE — YOLO11-pose extractor** `scripts/extract_yolo_poses.py`: emits
   SeeKer-format `*_alphapose_tracked_person.json` (schema unit-tested against the
   real ShanghaiTech files, `tests/test_pose_extract.py`). Supports `--frames-root`
   (per-clip frame dirs) or `--video-glob` (per-clip videos). UBnormal video stems
   already match SeeKer's `(normal|abnormal)_scene_*_scenario*` regex.
2. ✅ **DONE — runner is dataset-aware + fine-tune-capable** `run_experiments.py`:
   `--dataset {ShanghaiTech,UBnormal}` (threaded through `build_args`) and
   `--init_ckpt <path>` for transfer/fine-tuning (e.g. ShanghaiTech -> UBnormal).
   Auto-derives `data/UBnormal/context_oriented` cache path.
3. ⏳ **PENDING data** — `scripts/extract_yolo_detections.py` UBnormal clip
   discovery (its videos are `.mp4`; current globs `.avi`/frame dirs). Trivial; will
   finalise against the real layout once UBnormal is on disk.
4. ⏳ **VERIFY on data** — that the oriented vehicle cache + pose JSON clip ids line
   up with what the context/scene datasets expect (and UBnormal frame-level GT path).

## Fine-tuning (your idea) — supported
Train on ShanghaiTech, then continue on UBnormal:
```bat
.venv\Scripts\python scripts\run_experiments.py --dataset UBnormal --configs baseline vehicle ^
   --init_ckpt colab_results\results_v2\ShanghaiTech_vehicle\<ts>\checkpoint_best.pth ^
   --out colab_results\results_ubnormal_finetune --seed 42
```
(Works because input dims match when seg_len / max_vehicles / kv are unchanged.)

## Expectation (set honestly)
UBnormal's anomalies are mostly **non-vehicle** (like ShanghaiTech), so on
**overall** AUROC ours ≈ baseline; the vehicle context should help on the
**vehicle-scene subset / hazard metric**. The full mixed-traffic payoff comes on
**Street Scene** (all scenes are traffic, vehicles normal).
