import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import numpy as np


# =========================================================
# Dataset (UPDATED: uRWell + DC, det_id=0/1)
# =========================================================
class TrackDataset(Dataset):
    """
    Each sample:
        uRWell hits (xo, yo, xe, ye, z)   -> det_id = 0
        DC hits     (doca, xm, xr, yr, z) -> det_id = 1
        state        (x, y, tx, ty, Q)
    """

    def __init__(self,
                 ur_hits_list,
                 dc_hits_list,
                 states,
                 normalize=True,
                 hit_stats=None,
                 state_stats=None):

        assert len(ur_hits_list) == len(dc_hits_list) == len(states)

        self.ur_hits_list = ur_hits_list
        self.dc_hits_list = dc_hits_list
        self.states = np.array(states, dtype=np.float32)

        self.normalize = normalize

        # =========================
        # unified hit stats
        # =========================
        self.hit_stats = hit_stats or {
            "xo_mean": 0.0,
            "xo_std": 1.0,
            "yo_mean": 0.0,
            "yo_std": 1.0,
            "xe_mean": 0.0,
            "xe_std": 1.0,
            "ye_mean": 0.0,
            "ye_std": 1.0,
            "doca_mean": 0.0,
            "doca_std": 1.0,
            "xm_mean": 0.0,
            "xm_std": 1.0,
            "xr_mean": 0.0,
            "xr_std": 1.0,
            "yr_mean": 0.0,
            "yr_std": 1.0,
            "z_mean": 0.0,
            "z_std": 1.0
        }

        self.state_stats = state_stats or {
            k: (0.0, 1.0) for k in ["x", "y", "tx", "ty", "Q"]
        }

    def __len__(self):
        return len(self.states)

    # -----------------------------------------------------
    # merge two detectors into one sequence
    # -----------------------------------------------------
    def _merge_hits(self, ur_hits, dc_hits):

        hits = []

        # =========================
        # uRWell FIRST (det_id=0)
        # =========================
        for h in ur_hits:
            xo, yo, xe, ye, z = h
            hits.append([
                xo, yo, xe, ye,
                0, 0, 0, 0,
                z,
                0   # uRWell
            ])

        # =========================
        # DC SECOND (det_id=1)
        # =========================
        for h in dc_hits:
            doca, xm, xr, yr, z = h
            hits.append([
                0, 0, 0, 0,
                doca, xm, xr, yr,
                z,
                1   # DC
            ])

        hits = np.array(hits, dtype=np.float32)

        # IMPORTANT: keep physical ordering
        hits = hits[np.argsort(hits[:, 8])]

        return hits

    def __getitem__(self, idx):

        ur_hits = self.ur_hits_list[idx]
        dc_hits = self.dc_hits_list[idx]

        hits = self._merge_hits(ur_hits, dc_hits)
        state = self.states[idx].copy()

        # =====================================================
        # SEPARATE NORMALIZATION (IMPORTANT FOR STABILITY)
        # =====================================================
        if self.normalize:
            hs = self.hit_stats

            new_hits = []

            for h in hits:

                xo, yo, xe, ye, doca, xm, xr, yr, z, det = h

                # =========================
                # uRWell
                # =========================
                if det == 0:
                    xo = (xo - hs["xo_mean"]) / (hs["xo_std"])
                    yo = (yo - hs["yo_mean"]) / (hs["yo_std"])
                    xe = (xe - hs["xe_mean"]) / (hs["xe_std"])
                    ye = (ye - hs["ye_mean"]) / (hs["ye_std"])

                # =========================
                # DC
                # =========================
                else:
                    doca = (doca - hs["doca_mean"]) / (hs["doca_std"])
                    xm   = (xm   - hs["xm_mean"])   / (hs["xm_std"])
                    xr   = (xr   - hs["xr_mean"])   / (hs["xr_std"])
                    yr   = (yr   - hs["yr_mean"])   / (hs["yr_std"])

                # =========================
                # GLOBAL Z
                # =========================
                z = (z - hs["z_mean"]) / (hs["z_std"])

                new_hits.append([
                    xo, yo, xe, ye,
                    doca, xm, xr, yr,
                    z,
                    det
                ])

            hits = np.array(new_hits, dtype=np.float32)

            # ---------------------
            # state normalization
            # ---------------------
            for i, k in enumerate(["x", "y", "tx", "ty", "Q"]):
                m, s = self.state_stats[k]
                state[i] = (state[i] - m) / s

        return (
            torch.tensor(hits, dtype=torch.float32),
            torch.tensor(state, dtype=torch.float32)
        )


# =========================================================
# Collate function
# =========================================================
def collate_fn(batch):

    hits, states = zip(*batch)

    max_len = max(h.shape[0] for h in hits)

    padded_hits = []
    mask = []

    for h in hits:
        L = h.shape[0]
        pad = max_len - L

        if pad > 0:
            h = torch.cat([h, torch.zeros(pad, h.shape[1])], dim=0)

        padded_hits.append(h)
        mask.append([False] * L + [True] * pad)

    return (
        torch.stack(padded_hits),
        torch.stack(states),
        torch.tensor(mask, dtype=torch.bool)
    )


# =========================================================
# Model (UNCHANGED ARCH, only assumes det_id semantics)
# =========================================================
class TrackTransformer(pl.LightningModule):

    def __init__(self,
                 input_dim=9,
                 hidden_dim=32,
                 nhead=4,
                 num_layers=2,
                 lr=1e-3):

        super().__init__()
        self.lr = lr

        self.embedding = nn.Linear(input_dim, hidden_dim)

        # det_id:
        # 0 = uRWell
        # 1 = DC
        self.det_emb = nn.Embedding(2, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5)
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, x, mask):

        if mask is None:
            mask = torch.zeros(x.shape[0], x.shape[1],
                               dtype=torch.bool,
                               device=x.device)

        feat = x[..., :9]
        det_id = x[..., 9].long()

        x_emb = self.embedding(feat)
        x_emb = x_emb + self.det_emb(det_id)

        x = self.transformer(x_emb, src_key_padding_mask=mask)

        valid = ~mask
        length = valid.sum(dim=1, keepdim=True).clamp(min=1)

        x = x * valid.unsqueeze(-1)
        x = x.sum(dim=1) / length

        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        loss = self.loss_fn(self(x, mask), y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        loss = self.loss_fn(self(x, mask), y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class TrackTransformerWrapper(nn.Module):
    """
    Inference wrapper:
    - automatically creates padding mask
    - compatible with TorchScript / DJL export
    - supports DC + uRWell unified input
    """

    def __init__(self, core_model: nn.Module):
        super().__init__()
        self.core = core_model

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : [B, N, 10]
            Each hit:
            [doca, xm, xr, yr, xo, yo, xe, ye, z, det_id]
        """

        # IMPORTANT: all-valid mask (no padding inference logic here)
        mask = torch.zeros(
            x.shape[0],
            x.shape[1],
            dtype=torch.bool,
            device=x.device
        )

        return self.core(x, mask)

# =========================================================
# Logging callback
# =========================================================
class LossTracker(Callback):

    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        if "train_loss" in trainer.callback_metrics:
            self.train_losses.append(
                trainer.callback_metrics["train_loss"].item()
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        if "val_loss" in trainer.callback_metrics:
            self.val_losses.append(
                trainer.callback_metrics["val_loss"].item()
            )