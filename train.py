import os
from pathlib import Path
import time
import argparse
import json
import random
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from trainer import *
from data import *
from plotter import Plotter
import numpy as np


def parse_args():
    """
    Parse command-line arguments for training or inference.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including device selection, input files,
        model hyperparameters, and output options.
    """
    parser = argparse.ArgumentParser(description="Transformer Autoencoder Training")

    # Device selection
    parser.add_argument("--device", type=str, choices=["cpu", "gpu", "auto"], default="auto",
                        help="Choose device: cpu, gpu, or auto (default: auto)")

    # Input data files
    parser.add_argument("inputs", type=str, nargs="*", default=["hitsTracks.csv"],
                        help="One or more input CSV files")

    # Training hyperparameters
    parser.add_argument("--max_epochs", type=int, default=120,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for DataLoader")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for optimizer")

    # Model architecture parameters
    parser.add_argument("--hidden_dim", type=int, default=32,
                        help="Hidden dimension (d_model) of the transformer")
    parser.add_argument("--nhead", type=int, default=4,
                        help="Number of attention heads in the transformer")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of transformer encoder layers")

    # Output and runtime options
    parser.add_argument("--outdir", type=str, default="outputs/local",
                        help="Directory to save models and plots")
    parser.add_argument("--end_name", type=str, default="",
                        help="Optional suffix appended to output files")
    parser.add_argument("--no_train", action="store_true",
                        help="Skip training and only run inference using a saved model")
    parser.add_argument("--outbending", action="store_true",
        help="Use outbending (default: inbending")
    parser.add_argument("--enable_progress_bar", action="store_true",
                        help="Enable progress bar during training (default: disabled)")
    return parser.parse_args()


def set_seed(seed=42):
    """
    Fix random seeds to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


# --------------------------------------------------
def main():
    """
    Main entry point for training or evaluating the Transformer-based
    masked autoencoder / regression model.

    Workflow
    --------
    1. Parse command-line arguments and set random seeds.
    2. Load input CSV files and collect hit-level and track-state-level data.
    3. Compute or load normalization statistics.
    4. Construct datasets and data loaders.
    5. Train the transformer model (unless --no_train is specified).
    6. Run inference on the validation set.
    7. Denormalize predictions and generate diagnostic plots.
    """
    set_seed(42)

    args = parse_args()

    inputs = args.inputs if args.inputs else ["hitsTracks.csv"]
    outDir = args.outdir
    maxEpochs = args.max_epochs
    batchSize = args.batch_size
    end_name = args.end_name
    doTraining = not args.no_train
    inbending = not args.outbending

    # Ensure output directory exists
    os.makedirs(outDir, exist_ok=True)

    # --------------------------------------------------
    # Load data from CSV files
    print("\n\nLoading data...")
    startT_data = time.time()

    hits_ur_all = []  # List of hit arrays, each with shape [num_hits, 5]
    hits_dc_all = []     # List of hit arrays, each with shape [num_hits, 5]
    states_all = []   # List of state vectors, each with shape [5]

    for fname in inputs:
        print(f"Loading data from {fname} ...")
        ur_hits, dc_hits, states = read_tracks_with_hits(fname)
        hits_ur_all.extend(ur_hits)
        hits_dc_all.extend(dc_hits)
        states_all.extend(states)

    states_all = np.array(states_all, dtype=np.float32)

    # Paths for normalization statistics
    hit_stats_out_path = os.path.join(outDir, "hit_stats.json")
    state_stats_out_path = os.path.join(outDir, "state_stats.json")

    # --------------------------------------------------
    # Load or compute normalization statistics
    BASE_DIR = Path(__file__).resolve().parent
    NETS_DIR = BASE_DIR / "nets"

    if args.no_train:
        print("\n=== Inference mode: loading normalization stats from nets/ ===")

        if inbending:
            hit_stats_path = NETS_DIR / "hit_stats_inbending.json"
            state_stats_path = NETS_DIR / "state_stats_inbending.json"
        else:
            hit_stats_path = NETS_DIR / "hit_stats_outbending.json"
            state_stats_path = NETS_DIR / "state_stats_outbending.json"

        for p in (hit_stats_path, state_stats_path):
            if not p.exists():
                raise FileNotFoundError(f"Normalization stats not found: {p}")

        with open(hit_stats_path, "r") as f:
            hit_stats = json.load(f)

        with open(state_stats_path, "r") as f:
            state_stats = json.load(f)

        print("Loaded normalization stats from nets/")
    else:
        # Training mode: compute normalization statistics from data
        print("\n=== Automatically computing normalization statistics ===")
        all_ur = np.vstack(hits_ur_all)
        all_dc = np.vstack(hits_dc_all)


        # --------------------------------------------------
        # GLOBAL Z
        # --------------------------------------------------
        z_all = np.concatenate([all_ur[:, 4], all_dc[:, 4]])

        z_stats = {
            "z_mean": float(z_all.mean()),
            "z_std": float(z_all.std())
        }

        # ---------------- uRWell ----------------
        ur_stats = {
            "xo_mean": float(all_ur[:, 0].mean()),
            "xo_std": float(all_ur[:, 0].std()),
            "yo_mean": float(all_ur[:, 1].mean()),
            "yo_std": float(all_ur[:, 1].std()),
            "xe_mean": float(all_ur[:, 2].mean()),
            "xe_std": float(all_ur[:, 2].std()),
            "ye_mean": float(all_ur[:, 3].mean()),
            "ye_std": float(all_ur[:, 3].std()),
        }

        print("\n[uRWell]")
        for k, v in ur_stats.items():
            print(f"{k}: {v:.6g}")

        # ---------------- DC ----------------
        dc_stats = {
            "doca_mean": float(all_dc[:, 0].mean()),
            "doca_std": float(all_dc[:, 0].std()),
            "xm_mean": float(all_dc[:, 1].mean()),
            "xm_std": float(all_dc[:, 1].std()),
            "xr_mean": float(all_dc[:, 2].mean()),
            "xr_std": float(all_dc[:, 2].std()),
            "yr_mean": float(all_dc[:, 3].mean()),
            "yr_std": float(all_dc[:, 3].std()),
        }

        print("\n[DC]")
        for k, v in dc_stats.items():
            print(f"{k}: {v:.6g}")

        # Compute state-level statistics
        state_names = ["x", "y", "tx", "ty", "Q"]
        state_stats = {}

        print("\n=== State statistics ===")
        for i, name in enumerate(state_names):
            vals = states_all[:, i]
            mean, std = float(vals.mean()), float(vals.std())
            state_stats[name] = (mean, std)
            print(f"{name:>3s}: mean={mean:.6g}, std={std:.6g}")

        # save
        hit_stats = {**ur_stats, **dc_stats, **z_stats}
        with open(hit_stats_out_path, "w") as f:
            json.dump(hit_stats, f, indent=2)

        with open(state_stats_out_path, "w") as f:
            json.dump(state_stats, f, indent=2)

        print(f"\nSaved stats to {args.outdir}")

    # --------------------------------------------------
    # Initialize dataset with automatic normalization
    dataset = TrackDataset(
        ur_hits_list=hits_ur_all,
        dc_hits_list=hits_dc_all,
        states=states_all,
        normalize=True,
        hit_stats=hit_stats,
        state_stats=state_stats
    )

    # Split into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    print("\n\nTrain size:", train_size)
    print("Validation size:", val_size)

    train_loader = DataLoader(train_set, batch_size=batchSize,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batchSize,
                            shuffle=False, collate_fn=collate_fn)

    # Inspect one batch
    hits_sample, state_sample, mask_sample = next(iter(train_loader))
    print("hits_sample shape :", hits_sample.shape)
    print("state_sample shape:", state_sample.shape)
    print("mask_sample shape :", mask_sample.shape)

    endT_data = time.time()
    print(f"Loading data took {endT_data - startT_data:.2f}s\n\n")

    # --------------------------------------------------
    # Plotting utility
    plotter = Plotter(print_dir=outDir, end_name=end_name)

    # Sanity check for transformer dimensions
    if args.hidden_dim % args.nhead != 0:
        raise ValueError(
            f"d_model ({args.hidden_dim}) must be divisible by nhead ({args.nhead})"
        )

    # Initialize model and loss tracker
    model = TrackTransformer(
        hidden_dim=args.hidden_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        lr=args.lr
    )

    loss_tracker = LossTracker()

    # --------------------------------------------------
    # Training phase
    if doTraining:
        # Select accelerator and devices
        if args.device == "cpu":
            accelerator, devices = "cpu", 1
        elif args.device == "gpu":
            if torch.cuda.is_available():
                accelerator, devices = "gpu", 1
            else:
                print("GPU not available. Falling back to CPU.")
                accelerator, devices = "cpu", 1
        elif args.device == "auto":
            if torch.cuda.is_available():
                accelerator, devices = "gpu", "auto"
            else:
                accelerator, devices = "cpu", 1
        else:
            raise ValueError(f"Unknown device option: {args.device}")

        print(f"Using accelerator={accelerator}, devices={devices}")

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            strategy="auto",
            max_epochs=maxEpochs,
            enable_progress_bar=args.enable_progress_bar,
            log_every_n_steps=1000,
            enable_checkpointing=False,
            check_val_every_n_epoch=1,
            num_sanity_val_steps=0,
            logger=False,
            callbacks=[loss_tracker]
        )

        print("\n\nTraining...")
        startT_train = time.time()
        trainer.fit(model, train_loader, val_loader)
        endT_train = time.time()
        print(f"Training took {(endT_train - startT_train) / 60:.2f} minutes\n\n")

        # Plot training and validation losses
        plotter.plotTrainLoss(loss_tracker)

        # --------------------------------------------------
        # Save trained model as TorchScript
        model.to("cpu")
        model.eval()

        # Wrap model so that forward() automatically creates a full False mask
        wrapper_model = TrackTransformerWrapper(model)
        wrapper_model.eval()

        torchscript_model = torch.jit.script(wrapper_model)
        torchscript_model.save(f"{outDir}/transformer_{end_name}.pt")

    # --------------------------------------------------
    # Inference phase
    if doTraining:
        model_file = Path(outDir) / f"transformer_{end_name}.pt"
    else:
        if inbending:
            model_file = NETS_DIR / "transformer_default_inbending.pt"
        else:
            model_file = NETS_DIR / "transformer_default_outbending.pt"

    model_file = model_file.resolve()
    print("Loading model from:", model_file)

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    model = torch.jit.load(str(model_file))
    model.eval()

    val_loader2 = DataLoader(val_set, batch_size=1, shuffle=False)

    all_preds = []
    all_targets = []

    startT_test = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        for hits_batch, state_batch in val_loader2:
            hits_batch = hits_batch.to(device)
            state_batch = state_batch.to(device)

            y_pred = model(hits_batch)
            y_true = state_batch

            all_preds.append(y_pred.cpu())
            all_targets.append(y_true.cpu())

    endT_test = time.time()
    print(f"Test with {len(val_loader2.dataset)} samples took {endT_test - startT_test:.2f}s\n\n")

    # Concatenate predictions over the entire validation set
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # --------------------------------------------------
    # Denormalize predictions and targets
    def denormalize_state(states, stats):
        """
        Convert normalized state variables back to track-representation units.
        """
        result = states.copy()
        for i, key in enumerate(["x", "y", "tx", "ty", "Q"]):
            mean, std = stats[key]
            result[:, i] = result[:, i] * std + mean
        return result

    all_preds_denorm = denormalize_state(all_preds, state_stats)
    all_targets_denorm = denormalize_state(all_targets, state_stats)

    # Generate comparison plots
    plotter.plot_diff(all_targets_denorm, all_preds_denorm)
    plotter.plot_pred_target(all_targets_denorm, all_preds_denorm)


if __name__ == "__main__":
    main()
