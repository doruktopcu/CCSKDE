# Vendored SeeKer baseline

This directory is a **vendored copy** of SeeKer (the SKDE baseline we extend).

- **Upstream:** https://github.com/adelic99/seeker
- **License:** MIT — see [`LICENSE`](LICENSE) (retained unchanged; attribution preserved).
- **Reference:** Delić, Grčić, Šegvić, *Sequential Keypoint Density Estimator: an
  overlooked baseline of skeleton-based video anomaly detection*, ICCV 2025.

## Provenance
Vendored from upstream commit **`7e7ab66`** ("added code", = `origin/main` at
vendoring time), with local **portability patches** applied (originally the
nested-repo commit `3706984`, "device portability + path/typo fixes for local
run", plus minor MPS/numerical-stability tweaks). The complete diff against the
upstream baseline is saved here as
[`PORTABILITY_PATCHES.patch`](PORTABILITY_PATCHES.patch) (5 files: `seeker.py`,
`training.py`, `validation.py`, `models/made/made.py`,
`models/made/made_partial.py`).

## Why vendored (was a submodule)
`seeker/` was previously an **improperly configured git submodule** — a gitlink
with **no `.gitmodules`**, pointing at an **unpushable local commit** (we cannot
push to the upstream repo). Fresh `git clone`s therefore produced an *empty*
`seeker/`, and the working tree always showed dirty. It is now plain, tracked
files inside the CCSKDE repository for one-repo, clone-and-run reproducibility.

## Convention
Treat `seeker/` as a **read-only baseline**. Per project convention, do not edit
it to change behaviour — subclass / override in `ccskde/`. The only edits ever
applied are the device-portability + path/typo fixes captured in the patch above
(see `project_progress_report.md` #11, #12).

_Vendored 2026-05-29._
