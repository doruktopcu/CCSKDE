"""CCSKDE trainer — thin override of seeker.training.SeeKerTrainer.

Differences from upstream:
  * Loader yields [pose, score, C] (not just [pose, score]).
  * Model is `PartialAutoregressiveContextFC`, called as `model(x_pose, c)`.
  * `joint_lnp` accepts and forwards `c`.

Everything else (loss, optimizer, checkpointing, validation flow) is
inherited from upstream so the baseline-vs-CCSKDE diff stays small.
"""
from __future__ import annotations

import math
import os
import sys

import torch
from tqdm import tqdm

# Allow importing seeker's modules without polluting the parent package.
_SEEKER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "seeker")
if _SEEKER_DIR not in sys.path:
    sys.path.insert(0, _SEEKER_DIR)

from training import SeeKerTrainer            # noqa: E402  (after sys.path tweak)
from validation import score_anomalies        # noqa: E402


class SceneTrainer(SeeKerTrainer):
    """Unified scene-SKDE trainer.

    The model consumes an augmented scene skeleton (B, T, N', 2) with
    N' = 18 + 6M keypoints (vehicles first, pedestrian last) and is trained on
    the JOINT negative log-likelihood over all N' keypoints — cars are modelled,
    not just used as side information. SeeKer's `joint_lnp` is keypoint-count
    agnostic, so it is inherited unchanged. For frame-level scoring we read out
    the PEDESTRIAN keypoints' NLL (the last 18), keeping the score directly
    comparable to SeeKer while reflecting the vehicle conditioning.
    """

    PED_KP = 18

    def train(self, clip=100):
        import math  # noqa: F401  (parity with base)
        self.all_val_auc: dict[int, float] = {}
        self.model = self.model.to(self.args.device)
        for epoch in range(self.args.epochs):
            self.model.train()
            print(f"[scene] epoch {epoch + 1}/{self.args.epochs}")
            pbar = tqdm(self.train_loader)
            loss_val = float("nan")
            for data_arr in pbar:
                x = torch.permute(data_arr[0], (0, 2, 3, 1))[..., :2]
                x = x.to(self.args.device, non_blocking=True).float()
                lnp = self.joint_lnp(x)                 # (B, N') joint
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

    def validate(self):
        self.model.eval().to(self.args.device)
        probs = torch.empty(0).to(self.args.device)
        for data_arr in tqdm(self.val_loader):
            xc = torch.permute(data_arr[0], (0, 2, 3, 1))
            x = xc[..., :2].to(self.args.device).float()      # (B,T,N',2)
            conf = xc[..., -1].to(self.args.device)           # (B,T,N')
            B, T, Np, _ = x.shape
            with torch.no_grad():
                lnp = self.joint_lnp(x)                        # (B, N')
                lnp_ped = lnp[:, -self.PED_KP:]                # pedestrian keypoints
                pad = torch.randn(B, (T - 1) * self.PED_KP, device=self.args.device)
                nll = -torch.cat((pad, lnp_ped), dim=1)
                conf_ped = conf[..., -self.PED_KP:]            # (B,T,18)
                nll = (nll.view(B, T, self.PED_KP) * conf_ped).flatten(start_dim=1)
            probs = torch.cat((probs, nll), dim=0)
        scores = probs.cpu().detach().numpy().squeeze().copy(order="C")
        auc = score_anomalies(scores, self.val_metadata, args=self.args, split="validation")
        print("AUC on val (scene, pedestrian readout):", auc)
        return auc


def _unpack(data_arr, device):
    """Returns (x_pose [B,T,N,2], conf [B,T,N], c [B,T,Dctx])."""
    x_full = torch.permute(data_arr[0], (0, 2, 3, 1))  # (B, T, N, 3)
    x = x_full[..., :2].to(device, non_blocking=True).float()
    conf = x_full[..., -1].to(device, non_blocking=True)
    if len(data_arr) >= 3:
        c = data_arr[2].to(device, non_blocking=True).float()
    else:
        c = None
    return x, conf, c


class CCSKDETrainer(SeeKerTrainer):
    """Override the data path so we feed (x, c) into the model."""

    def __init__(self, *args, shuffled_context: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.shuffled_context = shuffled_context

    # ---- model invocation
    def joint_lnp(self, x, c):
        B, T, N, A = x.shape
        out = self.model(x, c)
        mu, logvar = torch.chunk(out, 2, dim=1)
        mu = mu.reshape(B, T * N, A)[:, (-N - 1):-1]
        logvar = logvar.reshape(B, T * N, A)[:, (-N - 1):-1]
        logvar = logvar.clamp(min=-10.0, max=10.0)  # MPS numerical stability
        var = torch.exp(logvar)
        E_inv = torch.diag_embed(1.0 / var)
        x = x.reshape(B, T * N, A)
        x_tgt = x[:, -N:]
        lnp = -0.5 * (
            torch.einsum("bti,btij,btj->bt", x_tgt - mu, E_inv, x_tgt - mu)
            + torch.sum(logvar, dim=-1)
            + x_tgt.shape[-1] * math.log(math.pi * 2)
        )
        return lnp

    # ---- training loop
    def train(self, clip=100):
        num_epochs = self.args.epochs
        self.model.train()
        self.model = self.model.to(self.args.device)
        all_val_auc: dict[int, float] = {}
        for epoch in range(num_epochs):
            self.model.train()
            print(f"Starting Epoch {epoch + 1} / {num_epochs}")
            pbar = tqdm(self.train_loader)
            for _, data_arr in enumerate(pbar):
                x, _conf, c = _unpack(data_arr, self.args.device)
                if self.shuffled_context and c is not None:
                    c = c[torch.randperm(c.size(0))]
                lnp = self.joint_lnp(x, c)
                neg_ll = -lnp.sum(-1).mean()
                neg_ll.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
                pbar.set_description(f"Loss: {neg_ll.item()}")
            self.log_writer.add_scalar("NLL Loss", neg_ll.item(), epoch)

            # Per-epoch validation. For ShanghaiTech val==test, so we always
            # score to produce the epoch-level AUROC curve. NOTE: upstream
            # seeker/training.py guards this on `dataset not in ["ShangaiTech",
            # "MSAD"]` — "ShangaiTech" is misspelled, so it never matched the
            # real "ShanghaiTech" arg and validation ran *by accident*. We make
            # that intent explicit here (validate whenever a val loader exists).
            if self.val_loader is not None:
                auc_val = self.validate()
                all_val_auc[epoch] = auc_val
            else:
                auc_val = 0.0

            is_best = auc_val == max(all_val_auc.values(), default=0)
            if is_best:
                self.save_checkpoint(epoch=epoch, filename="checkpoint_best.pth")
        self.all_val_auc = all_val_auc      # expose per-epoch history to runners
        return auc_val

    # ---- validation
    def validate(self):
        self.model.eval()
        self.model.to(self.args.device)
        pbar = tqdm(self.val_loader)
        probs = torch.empty(0).to(self.args.device)
        print("Starting Eval")
        for _, data_arr in enumerate(pbar):
            x, conf, c = _unpack(data_arr, self.args.device)
            B, T, N, _ = x.shape
            with torch.no_grad():
                lnp = self.joint_lnp(x, c)
                pad = torch.randn(B, (T - 1) * N).to(self.args.device)
                nll = -torch.cat((pad, lnp), dim=1)
                nll = nll.view(B, T, N) * conf
                nll = nll.flatten(start_dim=1)
            probs = torch.cat((probs, nll), dim=0)
        scores = probs.cpu().detach().numpy().squeeze().copy(order="C")
        auc = score_anomalies(scores, self.val_metadata, args=self.args, split="validation")
        print("AUC on val:", auc)
        return auc
