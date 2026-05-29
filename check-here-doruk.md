# check-here-doruk.md

Things that need your decision, or that I could not fully do under the current
constraints. Newest on top. (Created 2026-05-29.)

---

## 0. The two datasets you shared — honest verdict: neither fits this pipeline (don't download)

I checked both before downloading (to respect the data cap). Neither advances
the current ShanghaiTech skeleton-density method, for concrete reasons:

**Karlsruhe Objects (cvlibs)** — ~2,000 street-level images, ~10k bounding boxes
(cars + pedestrians), with *discrete orientation classes* (8 for cars, 4 for
peds). It has **no keypoints, no 3D pose, no segmentation**. So:
- It cannot give us a "car skeleton" (no keypoints) — only a box + coarse
  heading, which we already obtain in-domain from YOLO-seg `minAreaRect`.
- Street-level dashcam-style viewpoint ≠ ShanghaiTech overhead campus
  surveillance → a real domain gap; a detector trained on it would transfer
  poorly to our frames.
- It would NOT let us remove YOLO either — we'd still need detection on the
  actual ShanghaiTech pixels.

**Car Accidents & Deformation (Kaggle)** — annotated images of *crashed/damaged*
cars (deformation labels), built for insurance/damage assessment. This is a
supervised image task on accident photos: no surveillance video, no skeletons,
no pedestrian-vehicle interaction. Our method is *one-class unsupervised* VAD
trained only on **normal** ShanghaiTech and scoring hazard by likelihood — we
never train on "accident" examples, so this dataset has no place in the loop
(and isn't a usable evaluation benchmark for us either).

**Why no external car dataset really helps here:** we must stay on ShanghaiTech
(SeeKer comparability + it natively contains the pedestrian↔vehicle anomalies).
Car skeletons are already extracted *in-domain* from ShanghaiTech via YOLO-seg.
An external dataset could only help by training a *better* car extractor — but
(a) we already extract in-domain, and (b) every candidate (Karlsruhe, KITTI,
ApolloCar3D, PASCAL3D+) is automotive/street-level, so the surveillance domain
gap would hurt more than help.

**If you still want to push richer car structure**, the higher-value moves are:
(i) `yolo11l-seg` for cleaner masks (≤60 MB, in-domain), or (ii) **YOLO-World**
open-vocab for "road elements" (crosswalk/sign), both well under 20 GB. Tell me
and I'll proceed. Otherwise I recommend keeping the current in-domain YOLO-seg
car skeletons and not spending your data cap on these two.

---

## 1. "Can we completely take YOLO out of the workload?" — honest answer: not fully, but it already costs nothing at train time

**Short version:** the car skeletons have to come from the pixels via *some*
detector. We cannot conjure vehicle keypoints for ShanghaiTech without one
(the dataset ships no vehicle/road annotations, and the pedestrian poses
themselves already come from an external model — AlphaPose). So a perception
front-end is unavoidable. **But** YOLO in this project is **offline
preprocessing only** — it runs once, caches to `.npy`, and adds **zero
train-time / inference cost** to the density model (exactly like AlphaPose for
the pedestrian poses). In that sense it is already "out of" the learning
workload.

Options I considered for removing/replacing YOLO:
- **Replace with another detector/keypointer** (MMDetection, Detectron2): same
  role, heavier deps, no benefit. Rejected.
- **Use a dataset that already has vehicle annotations** and drop detection
  entirely: would mean leaving ShanghaiTech, losing comparability with the
  SeeKer baseline. See item 3.
- **Consolidate to ONE model for both people and vehicles** (e.g. YOLO11-pose
  for humans + YOLO-seg for vehicles, dropping AlphaPose): elegant, but the
  SeeKer baseline is defined on the *released AlphaPose* poses; swapping the
  pedestrian poses would make our baseline non-comparable. Could be a clean
  "single-front-end" ablation later.

**Decision (override if you disagree):** keep YOLO as the offline vehicle
front-end (it is not in the training loop), keep AlphaPose poses for the
pedestrians (baseline comparability). If you specifically want YOLO *gone*
even offline, tell me and I'll switch the vehicle skeletons to a
YOLO11-pose/seg single front-end ablation.

## 2. "Road elements" (crosswalks, lanes, signs) — need a decision

The ultimate target mentions **road elements**. COCO-pretrained YOLO has **no
crosswalk / lane / traffic-sign / road classes** — only vehicles and people.
To include road elements I would need one of:
- **YOLO-World** (open-vocabulary, prompt with "crosswalk", "road sign", ...):
  ~25–50 MB weights, well under your 20 GB budget, BUT open-vocab reliability
  on low-res 856×480 surveillance is uncertain and would need spot-checking.
- A **road/lane segmentation model** or a dataset with road-element labels
  (e.g. Mapillary, BDD100K) — these are large (BDD100K ~6.5 GB images) and
  off-distribution from ShanghaiTech campus footage.

**What I'm doing now:** building the unified scene-skeleton with the five
vehicle hazard classes (car, bus, bicycle, motorcycle, truck) only — these are
the agents that actually move and cause the pedestrian-vehicle hazards
ShanghaiTech labels. **Road elements are deferred** pending your call on
YOLO-World. If you want them, say so and I'll pull `yolov8s-worldv2`/
`yolo11-world` and prompt for crosswalk/sign/curb (within the 20 GB budget).

## 3. Do we need a new dataset?

Not for the core method — ShanghaiTech is the right benchmark (it natively has
pedestrian↔vehicle/bicycle anomalies and is SeeKer's benchmark, so the baseline
is comparable). I did **not** download any new dataset. A larger/cleaner
traffic dataset (e.g. for road elements or for a dedicated hazard split) would
be a scope change — flag me if you want it and I'll keep the download ≤ 20 GB.

## 4. Vehicle-skeleton quality: model size

Current oriented car-skeletons were extracted with **yolo11s-seg** (~20 MB,
already done for all 437 clips). Upgrading to **yolo11l-seg** (~50 MB) would
give cleaner masks → better oriented boxes, at the cost of re-extracting all
clips (~1 h, GPU). I will **not** re-extract automatically (it would contend
with the running experiment and burn time). Tell me if you want the
higher-quality re-extraction; otherwise yolo11s-seg is the default.

## 5. (resolved items / FYI)
- Local `pdflatex` auto-compile produced a stale PDF (PowerShell `*>` ate its
  output); LaTeX source is valid. I recompile cleanly via Python-captured logs
  when finalising the report.
