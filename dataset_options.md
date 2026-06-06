# Cross-dataset options for a mixed-traffic test (planning only — nothing downloaded)

Goal: a second benchmark where **vehicles are NORMAL** so we can separate
*hazard* from *object-presence* — the confound ShanghaiTech cannot resolve and
our counterfactual presence-vs-interaction probe is built for. Constraint:
**≤ 20 GB download** (limited internet). Our method needs, per scene:
RGB frames (to run YOLO-seg vehicles) + pedestrian **poses** (released, or we run
AlphaPose) + **frame-level** normal/abnormal labels, one-class (train on normal).

---

## TL;DR verdict

| Dataset | Mixed-traffic fit | Peds+poses | Vehicles normal | Size | Partial DL? | Verdict for us |
|---|---|---|---|---|---|---|
| **Street Scene** | ★★★ bullseye | peds yes / poses **no** | **yes** | **49 GB single zip** | **No** (one .zip) | Ideal content, **over cap & all-or-nothing** |
| **UBnormal** | ★★ (some street scenes) | yes / poses **yes** (skel-VAD repos) | partly | RGB tens of GB; **poses tiny** | yes (poses only) | **Most feasible** real cross-dataset; synthetic |
| **OTA (HuggingFace)** | ★★ traffic interactions | **no peds** (vehicles only) | **yes** | 172 GB total; **per-scene/test subset** | **Yes** (HF `allow_patterns`) | Great for a *vehicle-only* variant; no ped–vehicle |
| **CHAD** | ★ parking lot | yes / poses **yes** | background only | ~1.15 M frames (large) | partial (per-cam, GDrive) | Human-action VAD, **not** ped–vehicle |
| **NWPU Campus** | ★ campus | yes / poses no | partly | **76.6 GB** | no | Too big; ShanghaiTech-like confound |
| **MSAD-HR** | ★★ multi-scenario incl. traffic | yes | yes | request form; **features only** | n/a | Ships I3D/Swin features, **no raw frames** → unusable for our YOLO/pose |

**Bottom line:** under a hard 20 GB cap there is **no full** pedestrian-vehicle
mixed-traffic pose-VAD dataset that fits. The realistic moves are
(1) **UBnormal poses-only** (feasible, SeeKer-comparable, synthetic), or
(2) **Street Scene** if the cap can be relaxed once (the scientifically best
answer), or (3) **OTA test-subset** for a *vehicle-only* multi-agent extension.

---

## 1. Street Scene  — the scientific bullseye (but 49 GB, all-or-nothing)
- **Link:** Zenodo https://zenodo.org/records/10870472 (`StreetScene.zip`, 49.0 GB,
  MD5 a74c51e9…); mirror MERL https://www.merl.com/research/downloads/StreetScene
- **License:** CC-BY-SA-4.0. **Paper:** Ramachandra & Jones, WACV 2020 (arXiv:1902.05872).
- **Content:** static USB cam over a **two-lane street + bike lanes + sidewalks**,
  daytime. 46 train / 35 test seqs, 202,545 frames @ 1280×720. **205 anomalies /
  17 types**: jaywalking, illegal U-turn, car-outside-lane, biker-on-sidewalk,
  loitering, car-illegally-parked, dog-on-sidewalk, etc. Anomaly GT = per-frame
  **bounding-box regions** (→ trivially reduce to frame labels).
- **Why ideal:** cars, bikes and pedestrians **coexist as NORMAL**; the anomaly is
  the *configuration/interaction*. Exactly what our counterfactual probe needs.
- **Blocker:** distributed as **one 49 GB zip** → cannot grab a subset, and it
  exceeds the 20 GB cap. No HuggingFace per-file mirror found.
- **Effort if downloaded:** extract pedestrian poses (AlphaPose/YOLO-pose) + YOLO-seg
  vehicles (our existing pipeline); convert region GT → frame labels. ~half-day.

## 2. UBnormal — most feasible real cross-dataset (synthetic)
- **Link:** https://github.com/lilygeorgescu/UBnormal (RGB via Google Drive);
  poses available from skeleton-VAD repos (STG-NF https://github.com/orhir/STG-NF,
  MoCoDAD). **Paper:** Acsintoae et al., CVPR 2022.
- **License:** open for research (CC-BY-NC-SA per repo). **SeeKer uses it** → directly
  comparable to our baseline lineage.
- **Content:** fully **synthetic** virtual scenes, 268 train / 64 val / 211 test
  videos, **frame- AND pixel-level** labels, open-set. Several scenes are
  **streets with moving cars** (vehicles normal); anomalies include running,
  car-related events, falls, etc.
- **Size:** full RGB is tens of GB (not listed on the page — verify on the Drive),
  **but the pose release is small (≪1 GB)** → fits the cap easily if we use poses
  and skip RGB. (Trade-off: without RGB we can't run our *own* YOLO vehicles, so we
  rely on whatever object info exists, or limit to the pedestrian-only cross-dataset
  generalisation number.)
- **Why use it:** lowest bandwidth, SeeKer-comparable, gives an honest
  cross-dataset **generalisation** result; weakness: synthetic, and "vehicles
  normal" is only partial.
- **Effort:** low if poses-only (drop-in to our pose pipeline); medium if we also
  pull RGB for YOLO vehicles.

## 3. OTA — Overhead Traffic Anomalies (HuggingFace, partial-downloadable)
- **Link:** https://huggingface.co/datasets/starwit/overhead-traffic-anomalies
  (HF `snapshot_download` with `allow_patterns` → **per-scene / test-only subset**).
- **License:** CC-BY-NC-SA-4.0 (non-commercial). Source: City of Carmel, IN.
- **Content:** static **overhead** cams at intersections/roundabouts; 3 scenes,
  24 h train + 48 h test each; **~1 M frames @ 320×180**; **YOLOv8 detections +
  tracking IDs + geo-coords already provided**; **1,027 anomalies / 32 types** with
  **severity 0–4**: wrong-way, cutting-off, collisions, speeding, broken-down,
  off-road. Frame/interval labels (test split).
- **Size:** **172 GB full**, but HF lets you pull **one scene or test-only** (a few
  GB; tiny 320×180 frames + JSON) → **fits the cap**.
- **Catch for us:** **vehicles only, NO pedestrians/poses.** It does not test
  pedestrian-vehicle hazard. It *would* support a **vehicle-only multi-agent
  Scene-SKDE** variant (cars-as-keypoints, no ped) — a different, interesting
  experiment, and detections are pre-supplied (no YOLO needed).
- **Effort:** low-medium (build cars-only scene skeletons from the provided
  detections; new label adapter). Good "bonus extension," not the ped-vehicle test.

## 4. CHAD — pose-annotated, but human-action (not ped-vehicle)
- **Link:** https://github.com/TeCSAR-UNCC/CHAD (Google Drive). Springer/arXiv:2212.09258.
- **Content:** commercial **parking lot**, 4 cams, **1.15 M frames** (1.09 M normal /
  59 K abnormal); **bbox + Re-ID + pose** provided; frame-level .npy labels.
  **21 anomalies = human actions** (fighting, running, riding, littering…). **No
  vehicle/object annotations**; cars are unlabeled background.
- **Verdict:** poses are ready, but anomalies are **not** ped-vehicle, so it does
  not address our confound. Large download. Skip for this purpose.

## 5. NWPU Campus / MSAD-HR — noted, not feasible here
- **NWPU Campus** https://campusvad.github.io/ — 43 scenes, 28 anomaly classes,
  scene-dependent, **76.6 GB**. Real but over cap and campus-confounded like ShanghaiTech.
- **MSAD-HR** https://github.com/Tom-roujiang/MSAD — SeeKer benchmark, but the public
  release ships **extracted features** (I3D / Video-Swin), not raw frames, and the
  video needs a request form → we can't run our YOLO/pose pipeline. Unusable as-is.

---

## Recommended plan (pick one)

**Option A — UBnormal poses-only (feasible now, ≤1 GB).** Honest cross-dataset
*generalisation* number, SeeKer-comparable. Pedestrian-only (no new YOLO), so it
tests transfer of the pose density, not the ped-vehicle interaction per se.

**Option B — Street Scene (needs ~49 GB once).** The *correct* experiment for the
mixed-traffic / vehicles-normal claim. Only if the 20 GB cap can be relaxed for a
single bulk download (e.g. campus/uni network). Highest scientific payoff.

**Option C — OTA test-subset (HF partial, few GB).** A *vehicle-only* multi-agent
Scene-SKDE extension with severity labels; detections pre-supplied. Different
story (no pedestrians) but cheap and novel.

### Integration steps (same shape for any choice)
1. `ccskde/data/<name>_dataset.py`: adapter → our `(pose, conf, context/scene)`
   tensors; reuse `normalize_pose`, `ContextSpec`, `build_scene_skeleton`.
2. Labels: convert region/interval GT → per-frame 0/1 to match `score_anomalies`.
3. Perception: Street Scene → run `extract_yolo_detections.py --oriented` + poses;
   OTA → wrap provided YOLOv8 detections into our oriented schema; UBnormal → use
   released poses.
4. Train baseline + vehicle/scene (one-class on normal), eval with
   `evaluate_hazard.py` + `run_phase_a.py` (AUROC/AUPRC/EER/per-scene + DeLong +
   clip bootstrap + counterfactual).
5. Report: a cross-dataset table; on Street Scene specifically, the counterfactual
   should finally separate hazard from presence (vehicles are normal there).
