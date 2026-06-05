"""Unified CCSKDE experiment runner.

Loads the ShanghaiTech pose data ONCE and runs a set of configurations,
recording per-epoch validation AUROC for each. Replaces the old multi-phase
Colab notebook with a single reproducible local entry point.

Configurations (select with --configs):
    baseline          vanilla SeeKer (no context)
    proximity         CCSKDE, corrected per-class [1/d_min,count] context
    proximity_shuffled  proximity + shuffled-context negative control
    vehicle           CCSKDE, oriented car-skeleton matrix (needs oriented cache)
    vehicle_film      vehicle + FiLM covariance modulation
    vehicle_shuffled  vehicle + shuffled-context negative control

Example:
    set PYTHONPATH=%CD%
    .venv\\Scripts\\python scripts\\run_experiments.py ^
        --data_dir data ^
        --proximity_cache colab_results\\results\\context_cache ^
        --oriented_cache data\\ShanghaiTech\\context_oriented ^
        --device cuda --epochs 10 --batch_size 1024 --seg_len 24 --seg_stride 1 ^
        --configs baseline proximity vehicle vehicle_film vehicle_shuffled ^
        --out colab_results\\results_v2
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEKER = os.path.join(ROOT, "seeker")
for p in (SEEKER, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from args import init_parser, init_sub_args, create_exp_dirs   # noqa: E402
from dataset import SkeletonSequenceDataset                     # noqa: E402
from models.partial_autoregressive import PartialAutoregressiveFC  # noqa: E402
from training import SeeKerTrainer                               # noqa: E402
from utils.training_utils import init_optimizer                 # noqa: E402

from ccskde.context import ContextSpec                          # noqa: E402
from ccskde.data import ContextualSkeletonSequenceDataset       # noqa: E402
from ccskde.data.scene_dataset import SceneSkeletonDataset      # noqa: E402
from ccskde.models import (                                     # noqa: E402
    PartialAutoregressiveContextFC, PartialAutoregressiveSceneFC,
)
from ccskde.training import CCSKDETrainer, SceneTrainer          # noqa: E402
from ccskde.context.scene import PED_KP                          # noqa: E402


class BaselineTrainer(SeeKerTrainer):
    """SeeKer trainer that validates every epoch and records the history."""

    def train(self, clip=100):
        self.all_val_auc: dict[int, float] = {}
        self.model = self.model.to(self.args.device)
        for epoch in range(self.args.epochs):
            self.model.train()
            print(f"[baseline] epoch {epoch + 1}/{self.args.epochs}")
            pbar = tqdm(self.train_loader)
            loss_val = float("nan")
            for data_arr in pbar:
                x = torch.permute(data_arr[0], (0, 2, 3, 1))[..., :2]
                x = x.to(self.args.device, non_blocking=True).float()
                lnp = self.joint_lnp(x)
                loss = -lnp.sum(-1).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss_val = loss.item()
                pbar.set_description(f"Loss: {loss_val:.3f}")
            self.log_writer.add_scalar("NLL Loss", loss_val, epoch)
            auc = self.validate()
            self.all_val_auc[epoch] = auc
            if auc == max(self.all_val_auc.values()):
                self.save_checkpoint(epoch=epoch, filename="checkpoint_best.pth")
        return auc


# ---------------------------------------------------------------------------
CONFIGS = {
    "baseline":          dict(kind="baseline"),
    "proximity":         dict(kind="context", mode="proximity", cache="proximity"),
    "proximity_shuffled":dict(kind="context", mode="proximity", cache="proximity", shuffled=True),
    "vehicle":           dict(kind="context", mode="vehicle",   cache="oriented"),
    "vehicle_film":      dict(kind="context", mode="vehicle",   cache="oriented", film=True),
    "vehicle_shuffled":  dict(kind="context", mode="vehicle",   cache="oriented", shuffled=True),
    # unified scene-SKDE: cars are extra skeletal joints in one density
    "scene":             dict(kind="scene",   cache="oriented"),
    # ablation: pedestrian ordered FIRST (vehicles after) -> removes the
    # pedestrian-given-vehicles conditioning at identical capacity.
    "scene_pedfirst":    dict(kind="scene",   cache="oriented", ped_first=True),
}


def build_args(a):
    parser = init_parser()
    argv = [
        "--dataset", a.dataset, "--data_dir", a.data_dir,
        "--device", a.device, "--epochs", str(a.epochs),
        "--batch_size", str(a.batch_size), "--seg_len", str(a.seg_len),
        "--seg_stride", str(a.seg_stride), "--seed", str(a.seed),
        "--num_workers", str(a.num_workers), "--exp_dir", a.out,
    ]
    args = parser.parse_args(argv)
    args, _ = init_sub_args(args)
    return args


def load_base(args):
    print("[runner] loading pose data once (train + test)...")
    base = {}
    for split in ("train", "test"):
        stride = args.seg_stride if split == "train" else 1
        base[split] = SkeletonSequenceDataset(
            args.pose_path[split], path_to_vid_dir=args.vid_path[split],
            evaluate=(split == "test"), filter_conf=args.filter_conf,
            seg_len=args.seg_len, dataset=args.dataset,
            train_seg_conf_th=args.train_seg_conf_th, seg_stride=stride,
            vid_path=args.vid_path[split], split=split,
        )
    return base


def make_loader(ds, args, shuffle):
    return DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                      pin_memory=True, shuffle=shuffle)


def build_loaders(name, cfg, base, args, a):
    """Build datasets + loaders for a config ONCE (including the heavy
    precompute), so they are REUSED across all seeds. This is the fix for the
    per-seed memory leak: the context is seed-independent, so we must not
    re-precompute (each ~3 GB) per seed. Returns a dict the per-seed trainer
    consumes; call `del` on it after the config to free the precompute."""
    if cfg["kind"] == "baseline":
        return dict(kind="baseline", meta=base["test"].metadata,
                    tr=make_loader(base["train"], args, shuffle=True),
                    te=make_loader(base["test"], args, shuffle=False))
    if cfg["kind"] == "scene":
        ped_first = cfg.get("ped_first", False)
        spec = ContextSpec(mode="vehicle", max_vehicles=a.max_vehicles, kv=a.kv,
                           ped_first=ped_first)
        n_prime = PED_KP + a.max_vehicles * a.kv
        ds = {s: SceneSkeletonDataset(base[s], os.path.join(a.oriented_cache, s), spec=spec).precompute()
              for s in ("train", "test")}
        return dict(kind="scene", ds=ds, n_prime=n_prime, ped_first=ped_first,
                    meta=ds["test"].metadata,
                    tr=make_loader(ds["train"], args, shuffle=True),
                    te=make_loader(ds["test"], args, shuffle=False))
    spec = ContextSpec(mode=cfg["mode"], max_vehicles=a.max_vehicles, kv=a.kv)
    cache_root = a.oriented_cache if cfg["cache"] == "oriented" else a.proximity_cache
    ds = {s: ContextualSkeletonSequenceDataset(base[s], os.path.join(cache_root, s), spec=spec).precompute()
          for s in ("train", "test")}
    return dict(kind="context", ds=ds, spec=spec, film=cfg.get("film", False),
                shuffled=cfg.get("shuffled", False), meta=ds["test"].metadata,
                tr=make_loader(ds["train"], args, shuffle=True),
                te=make_loader(ds["test"], args, shuffle=False))


def train_one_seed(name, L, args, a, writer, seed, dir_suffix=""):
    """Train one seed reusing the shared loaders in L. Frees the model/optimizer
    (not the loaders) at the end."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_kp = 2 * 18 * args.seg_len
    hidden = args.n_layers * [args.expansion_factor * (n_kp - 2)]
    args.ckpt_dir = create_exp_dirs(args.exp_dir, dirmap=f"{args.dataset}_{name}{dir_suffix}")
    opt = init_optimizer(args.model_optimizer, lr=args.model_lr)

    if L["kind"] == "baseline":
        model = PartialAutoregressiveFC(dim=n_kp, hidden_dims=hidden, droppout=args.droppout)
        trainer = BaselineTrainer(args, model, L["tr"], L["te"], L["te"], L["meta"], L["meta"],
                                  optimizer_f=opt, log_writer=writer, dataset=args.dataset)
    elif L["kind"] == "scene":
        n_in = 2 * L["n_prime"] * args.seg_len
        hidden_scene = args.n_layers * [args.expansion_factor * (n_in - 2)]
        model = PartialAutoregressiveSceneFC(dim=n_in, n_current_kp=L["n_prime"],
                                             hidden_dims=hidden_scene, droppout=args.droppout)
        trainer = SceneTrainer(args, model, L["tr"], L["te"], L["te"], L["meta"], L["meta"],
                               optimizer_f=opt, log_writer=writer, dataset=args.dataset)
        trainer.ped_first = L.get("ped_first", False)
    else:
        model = PartialAutoregressiveContextFC(
            dim=n_kp, ctx_dim=L["spec"].dim * args.seg_len, hidden_dims=hidden,
            droppout=args.droppout, film_cov=L["film"])
        trainer = CCSKDETrainer(args, model, L["tr"], L["te"], L["te"], L["meta"], L["meta"],
                                optimizer_f=opt, log_writer=writer, dataset=args.dataset,
                                shuffled_context=L["shuffled"])
    if getattr(a, "init_ckpt", None):
        ck = torch.load(a.init_ckpt, map_location=args.device, weights_only=False)
        missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
        print(f"[{name} seed={seed}] fine-tuning from {a.init_ckpt} "
              f"(missing={len(missing)} unexpected={len(unexpected)})")
    t0 = time.time()
    trainer.train()
    dt = time.time() - t0
    aucs = [trainer.all_val_auc[e] for e in sorted(trainer.all_val_auc)]
    res = dict(aucs=aucs, best=max(aucs), best_epoch=int(np.argmax(aucs)),
               mean=float(np.mean(aucs)), minutes=round(dt / 60, 1))
    print(f"[{name} seed={seed}] best={res['best']:.4f}@{res['best_epoch']} "
          f"mean={res['mean']:.4f} ({res['minutes']}min)")
    del model, trainer, opt
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return res


def plot(results, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print("plot skipped:", e)
        return
    plt.figure(figsize=(8, 5))
    for name, r in results.items():
        if "aucs" not in r:
            continue
        plt.plot(range(1, len(r["aucs"]) + 1), r["aucs"], marker="o", label=name)
    plt.xlabel("epoch"); plt.ylabel("validation AUROC")
    plt.title("CCSKDE — ShanghaiTech validation AUROC"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=130)
    print("wrote", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--dataset", default="ShanghaiTech", choices=["ShanghaiTech", "UBnormal"],
                    help="benchmark; UBnormal enables the cross-dataset study")
    ap.add_argument("--init_ckpt", default=None,
                    help="checkpoint to initialise from before training "
                         "(fine-tuning / transfer, e.g. ShanghaiTech -> UBnormal)")
    ap.add_argument("--proximity_cache", default="colab_results/results/context_cache")
    ap.add_argument("--oriented_cache", default="data/ShanghaiTech/context_oriented")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--seg_len", type=int, default=24)
    ap.add_argument("--seg_stride", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="multiple seeds -> per-config mean±std (W3). Overrides --seed.")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_vehicles", type=int, default=2)
    ap.add_argument("--kv", type=int, default=6)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS),
                    choices=list(CONFIGS))
    ap.add_argument("--out", default="colab_results/results_v2")
    ap.add_argument("--force", action="store_true",
                    help="re-run configs even if already present in the results file")
    a = ap.parse_args()

    # auto-derive per-dataset cache paths if the user left the ShanghaiTech defaults
    if a.dataset == "UBnormal" and a.oriented_cache == "data/ShanghaiTech/context_oriented":
        a.oriented_cache = "data/UBnormal/context_oriented"

    os.makedirs(a.out, exist_ok=True)
    json_path = os.path.join(a.out, "experiment_results.json")
    results = {}
    if os.path.exists(json_path) and not a.force:
        results = json.load(open(json_path)).get("results", {})
        if results:
            print(f"[runner] resuming; already have: {list(results)}")

    todo = [c for c in a.configs if a.force or c not in results]
    if not todo:
        print("[runner] nothing to do (all requested configs already present).")
        plot(results, os.path.join(a.out, "auroc_curves.png"))
        return

    args = build_args(a)
    base = load_base(args)
    writer = SummaryWriter(log_dir=os.path.join(a.out, "tb"))
    seeds = a.seeds if a.seeds else [a.seed]
    multi = len(seeds) > 1

    for name in todo:
        print("\n" + "=" * 70 + f"\n  {name}  (seeds={seeds})\n" + "=" * 70)
        L = build_loaders(name, CONFIGS[name], base, args, a)   # precompute ONCE
        per_seed = []
        for seed in seeds:
            suffix = f"_s{seed}" if multi else ""
            per_seed.append(train_one_seed(name, L, args, a, writer, seed, suffix))
        del L                                                   # free precompute
        gc.collect()
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()
        if not multi:
            results[name] = per_seed[0]
        else:
            bests = [r["best"] for r in per_seed]
            means = [r["mean"] for r in per_seed]
            results[name] = dict(
                per_seed=per_seed, seeds=seeds, aucs=per_seed[0]["aucs"],
                best_mean=float(np.mean(bests)), best_std=float(np.std(bests)),
                mean_mean=float(np.mean(means)), mean_std=float(np.std(means)))
            print(f"[{name}] best {np.mean(bests):.4f}±{np.std(bests):.4f} | "
                  f"mean {np.mean(means):.4f}±{np.std(means):.4f}  over {len(seeds)} seeds")
        # persist after each config so a crash never loses completed runs
        meta = dict(epochs=a.epochs, batch_size=a.batch_size, seg_len=a.seg_len,
                    seg_stride=a.seg_stride, seeds=seeds,
                    max_vehicles=a.max_vehicles, kv=a.kv,
                    device=a.device, gpu=(torch.cuda.get_device_name(0)
                                          if torch.cuda.is_available() else "cpu"))
        with open(os.path.join(a.out, "experiment_results.json"), "w") as f:
            json.dump(dict(results=results, meta=meta), f, indent=2)

    plot(results, os.path.join(a.out, "auroc_curves.png"))
    print("\n[runner] done. Results in", a.out)


if __name__ == "__main__":
    main()
