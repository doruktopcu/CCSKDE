"""CCSKDE entry point. Mirrors seeker/seeker.py but builds the contextual
dataset + context-conditioned model + CCSKDETrainer.

Run from the project root:
    PYTHONPATH=. .venv/bin/python ccskde/seeker_ctx.py \\
        --dataset ShanghaiTech \\
        --data_dir /abs/path/to/data \\
        --exp_dir  /abs/path/to/exp_dir \\
        --device   mps \\
        --context_cache_dir /abs/path/to/data/ShanghaiTech/context \\
        [--shuffled_context]
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Make seeker/ importable as a flat package (matches its bare-import style).
_SEEKER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "seeker")
if _SEEKER_DIR not in sys.path:
    sys.path.insert(0, _SEEKER_DIR)

from args import create_exp_dirs, init_parser, init_sub_args   # noqa: E402
from dataset import SkeletonSequenceDataset                     # noqa: E402
from utils.training_utils import init_optimizer                 # noqa: E402

from ccskde.context import ContextSpec
from ccskde.data import ContextualSkeletonSequenceDataset
from ccskde.models import PartialAutoregressiveContextFC
from ccskde.training import CCSKDETrainer


def _add_ctx_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--context_cache_dir", type=str, required=True,
                        help="dir with <split>/<scene>_<clip>.npy YOLO caches")
    parser.add_argument("--shuffled_context", action="store_true",
                        help="negative-control ablation: permute C across "
                             "segments at training time")
    parser.add_argument("--context_mode", choices=["proximity", "vehicle"],
                        default="proximity",
                        help="proximity = per-class [1/d_min,count]; "
                             "vehicle = oriented car-skeleton matrix")
    parser.add_argument("--film_cov", action="store_true",
                        help="add a FiLM head that modulates the covariance "
                             "from the context (contextual covariance penalty)")
    parser.add_argument("--max_vehicles", type=int, default=2,
                        help="vehicle mode: number of nearest hazards kept (M)")
    parser.add_argument("--kv", type=int, default=6,
                        help="vehicle mode: keypoints per vehicle")


def _build_loaders(args, spec: ContextSpec):
    ds_args = {
        "seg_len": args.seg_len,
        "dataset": args.dataset,
        "train_seg_conf_th": args.train_seg_conf_th,
    }
    splits = ["train", "test"] if args.dataset != "UBnormal" else ["train", "validate", "test"]
    ds, ld = {}, {}
    for split in splits:
        ds_args["seg_stride"] = args.seg_stride if split == "train" else 1
        ds_args["vid_path"] = args.vid_path[split]
        ds_args["split"] = split
        base = SkeletonSequenceDataset(args.pose_path[split], path_to_vid_dir=args.vid_path[split],
                                       evaluate=(split == "test"), filter_conf=args.filter_conf,
                                       **ds_args)
        ds[split] = ContextualSkeletonSequenceDataset(
            base, os.path.join(args.context_cache_dir, split), spec=spec
        )
        ld[split] = DataLoader(ds[split], batch_size=args.batch_size,
                               num_workers=args.num_workers, pin_memory=True,
                               shuffle=(split == "train"))
    if args.dataset != "UBnormal":
        ds["validate"] = ds["test"]
        ld["validate"] = ld["test"]
    return ds, ld


def main() -> None:
    parser = init_parser()
    _add_ctx_args(parser)
    args = parser.parse_args()

    if args.seed == 999:
        args.seed = torch.initial_seed()
        np.random.seed(0)
    else:
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(args.seed)
        np.random.seed(0)

    args, _model_args = init_sub_args(args)
    args.ckpt_dir = create_exp_dirs(args.exp_dir, dirmap=args.dataset)

    spec = ContextSpec(mode=args.context_mode,
                       max_vehicles=args.max_vehicles, kv=args.kv)
    n_ctx_per_segment = spec.dim * args.seg_len
    n_kp = 2 * 18 * args.seg_len
    hidden = args.n_layers * [args.expansion_factor * (n_kp - 2)]

    model = PartialAutoregressiveContextFC(
        dim=n_kp, ctx_dim=n_ctx_per_segment, hidden_dims=hidden,
        droppout=args.droppout, film_cov=args.film_cov,
    )
    print(f"[ccskde] context_mode={spec.mode} dim_per_frame={spec.dim} "
          f"n_ctx={n_ctx_per_segment} film_cov={args.film_cov}")
    print(model)
    model.to(args.device)

    ds, ld = _build_loaders(args, spec)
    writer = SummaryWriter()
    trainer = CCSKDETrainer(
        args, model, ld["train"], ld["test"], ld["validate"],
        ds["validate"].metadata, ds["test"].metadata,
        optimizer_f=init_optimizer(args.model_optimizer, lr=args.model_lr),
        log_writer=writer, dataset=args.dataset,
        shuffled_context=args.shuffled_context,
    )
    trainer.train()


if __name__ == "__main__":
    main()
