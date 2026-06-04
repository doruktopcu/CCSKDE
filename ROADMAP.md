# CCSKDE Roadmap — review-driven sprints

Every item closes a specific weakness from the adversarial review (`W1`–`W12`)
or pursues an ideation direction (`#1`–`#10`). `[x]` = done. Sprint 1 = easy,
zero-new-data wins we can land immediately.

---

## 🟢 Sprint 1 — easy wins (do now, no downloads)

- [x] **W7 — terminology.** Now described as an *oriented bounding box rendered as
      keypoints* (4 corners + center + heading), explicitly **not** an articulated
      skeleton. Fixed in `report/main.tex`, `README.md`.
- [x] **W11 — efficiency numbers.** `scripts/measure_efficiency.py`: 2.24 / 2.97 /
      6.22 M params, 0.30 / 0.41 / 0.72 ms/batch, >3e7 frame-scores/s (RTX 5080);
      added to report.
- [x] **W1/W2 — counterfactual interaction probe.** `evaluate_hazard.py
      --counterfactual`: zeroed-context ≈ baseline (0.741); interaction term
      (real−cf) → veh-hazard AUROC **0.967**, ρ **0.70**, full only 0.69 →
      interaction-specific, not capacity/presence. In report.
- [x] **W8 — balanced hazard metric.** `balanced_hazard_auroc` (bootstrap CI):
      **0.62 [0.59, 0.66]** within near-vehicle frames. In report.
- [x] **W3 — multi-seed harness.** `run_experiments.py --seeds` loops seeds &
      aggregates mean±std (capability; full run is Ambitious).
- [x] **W6 — joint readout default.** Scene model defaults to joint read-out in
      `evaluate_hazard.py` + `ccskde-eval` skill.
- [x] **W7/W12 — report writing.** Added multi-agent/interaction positioning
      (ComplexVAD'25, Wiederer'22) + a scope/theory paragraph (what the joint
      density can/can't represent).
- [x] **W4/W5 — correct SeeKer facts.** Datasets = UBnormal / ShanghaiTech(+HR) /
      MSAD-HR (not Avenue); full-cov ≈ diagonal (77.8 vs 77.9) noted → dropped
      full-covariance from the roadmap.

---

## 🟡 Ambitious goals (real compute / engineering)

- [ ] **W4 — reproduce baseline ≈ 0.855** on ShanghaiTech (debug lr / epochs /
      seg-len / stride / normalization / conf-threshold). Until this lands, all
      relative gains are suspect.
- [ ] **W3 — full multi-seed matrix (≥5 seeds)** + significance tests (paired
      bootstrap / Wilcoxon) for every config comparison.
- [ ] **W9 — ablation sweeps:** M (#vehicles), K_v, agent ordering
      (vehicles-first vs after), FiLM on/off *on the hazard metric*, ± velocity.
- [ ] **W10 — perception-robustness study:** inject detection noise / dropout /
      false-positive vehicles; measure score stability.
- [ ] **#4 — expressive conditional:** replace per-keypoint diagonal Gaussian with
      a small normalizing-flow (or diffusion) head; compare against MoCoDAD.
- [ ] **#5 — joint crowd-SKDE:** model multiple *people* jointly (drop SeeKer's
      independence + max-pool) → human–human interaction anomalies.
- [ ] **W6 — prove scene > context** with the joint read-out on a setting where it
      should matter (vehicle-intrinsic anomalies).

---

## 🔴 Ultra goals (new domains / multi-month)

- [ ] **W5 / #2 — second & third datasets:** UBnormal and **MSAD-HR** (needs
      downloads). MSAD-HR is the priority: vehicles can be *normal* there, the
      setting that can actually prove "hazard ≠ object presence" (W2).
- [ ] **#1 — universal keypoint scene-SKDE:** open-vocabulary agents via
      X-Pose / UniPose — any human/animal/object becomes a keypoint agent in one
      density. The real generalization of "objects are keypoints."
- [ ] **#2 — animal / livestock behavior SKDE** (DeepLabCut / AP-10K / MABe):
      lameness, distress, disease onset. Uncrowded field, clean datasets.
- [ ] **#3 — counterfactual causal interaction framework** (formalized do-operator
      scoring; disentangles interaction from presence by construction).
- [ ] **#6 — hazard anticipation / time-to-event** (roll out future keypoints,
      score predicted near-collision; PET/TTC surrogate).
- [ ] **#8 — viewpoint-invariant SKDE** (SE(2)/SE(3)-equivariant or 3D-lifted) for
      cross-dataset generalization.
- [ ] **Cross-domain paper:** "Universal Scene Density Estimation" spanning
      traffic + animal + human VAD with counterfactual hazard scoring.

---

_Sprint 1 owner: this session. Ambitious/Ultra: planned. Update checkboxes as we go._
