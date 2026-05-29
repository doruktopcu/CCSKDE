# CCSKDE Handoff

Use this when resuming work on a different machine. Paste the prompt below into Claude Code after the folder is in place.

## Prompt

I'm continuing a research project (`CCSKDE`) that I previously worked on from my Mac. The full directory has been transferred via SSD — same layout, same files.

**Project state:**
- Main repo: `CCSKDE` on branch `main`, clean and in sync with `github.com/doruktopcu/CCSKDE`.
- Submodule: `seeker/` (upstream `github.com/adelic99/seeker`, which I cannot push to). It has:
  - 1 local commit ahead of origin/main
  - Uncommitted edits to `seeker.py` and `training.py`
  - Untracked training artifacts in `runs/`, `exp_dir/`, `models/` that were transferred manually since they aren't in git
- Recent commits on main: "Moving to PC", "Report generated.", "CCSKDE initialized.", "Progressed in the initial report.", "Project proposal added."

**First, please:**
1. Run `git status` in both `CCSKDE` and `seeker/` to confirm the state matches the above.
2. Verify the untracked artifacts (`seeker/runs/`, `seeker/exp_dir/`, `seeker/models/`) made it across.
3. Check that Python dependencies are installed (look for `requirements.txt`, `pyproject.toml`, or `environment.yml`) and tell me what's needed to run `seeker/training.py`.

Then ask me what I want to work on next.
