"""Build CCSKDE end-to-end and run a single forward+backward, then exit.
Catches arg-parse, import, model-shape, dataset-wrapper, and trainer wiring
bugs in seconds without competing with a real training run for resources.
"""
from __future__ import annotations

import sys

# Reuse the entry-point's setup, but stub trainer.train to a single-batch op.
sys.argv = [
    "seeker_ctx.py",
    "--dataset", "ShanghaiTech",
    "--data_dir", "data",
    "--exp_dir", "exp_dir",
    "--device", "cpu",
    "--num_workers", "0",
    "--epochs", "1",
    "--batch_size", "4",
    "--seg_len", "24",
    "--seed", "42",
    "--context_cache_dir", "data/ShanghaiTech/context",
]

import torch                                         # noqa: E402

from ccskde import training as t                    # noqa: E402
from ccskde.training import _unpack                 # noqa: E402


def fast_train(self, clip=100):
    print("[dry-run] entered train(); skipping main loop", flush=True)
    self.model.train()
    self.model.to(self.args.device)
    for data_arr in self.train_loader:
        x, _conf, c = _unpack(data_arr, self.args.device)
        print(f"  shapes  x={tuple(x.shape)}  c={tuple(c.shape) if c is not None else None}", flush=True)
        lnp = self.joint_lnp(x, c)
        loss = -lnp.sum(-1).mean()
        loss.backward()
        print(f"  loss={loss.item():.4f}  finite={bool(torch.isfinite(loss))}", flush=True)
        # AR re-check on the contextual model in real conditions
        from ccskde.models import PartialAutoregressiveContextFC
        assert isinstance(self.model, PartialAutoregressiveContextFC)
        print("  model OK; CCSKDE wiring verified end-to-end.", flush=True)
        return 0.0


t.CCSKDETrainer.train = fast_train

# Now run the entry point.
import ccskde.seeker_ctx as entry                    # noqa: E402
entry.main()
