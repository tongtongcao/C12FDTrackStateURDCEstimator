import os
import matplotlib

# Use non-interactive backend for batch / server environments
matplotlib.use('Agg')

from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import norm
import pandas as pd

# --------------------------------------------------
# Global matplotlib style configuration
plt.rcParams.update({
    'font.size': 15,
    'legend.edgecolor': 'white',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.major.size': 15,
    'xtick.minor.size': 10,
    'ytick.major.size': 15,
    'ytick.minor.size': 10,
    'xtick.major.width': 3,
    'xtick.minor.width': 3,
    'ytick.major.width': 3,
    'ytick.minor.width': 3,
    'axes.linewidth': 3,
    'figure.max_open_warning': 200,
    'lines.linewidth': 5
})


class Plotter:
    """
    Utility class for producing training and evaluation plots
    for track reconstruction models.
    """

    def __init__(self, print_dir='', end_name=''):
        """
        Parameters
        ----------
        print_dir : str
            Directory where plots and CSV files will be saved.
        end_name : str
            Optional suffix appended to output file names
            (useful for distinguishing different runs).
        """
        self.print_dir = print_dir
        self.end_name = end_name

    # --------------------------------------------------
    def plotTrainLoss(self, tracker):
        """
        Plot training and validation loss curves on a logarithmic scale.

        Parameters
        ----------
        tracker : object
            An object with attributes:
              - train_losses : list of float
              - val_losses   : list of float
            Typically provided by a PyTorch Lightning callback.
        """
        train_losses = tracker.train_losses
        val_losses = tracker.val_losses

        plt.figure(figsize=(20, 20))
        plt.plot(train_losses, label='Train', color='royalblue')
        plt.plot(val_losses, label='Validation', color='firebrick')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()

        outname = f"{self.print_dir}/loss_{self.end_name}.png"
        plt.savefig(outname)
        plt.close()

    # --------------------------------------------------
    def plot_diff(self, y_true, y_pred):
        """
        Plot distributions of prediction residuals (y_pred - y_true)
        for each track parameter, together with Gaussian fits.

        Fixed fit ranges and plot ranges are used for consistency
        across different runs.

        Parameters
        ----------
        y_true : np.ndarray, shape [N, 5]
            Ground-truth track states: [x, y, tx, ty, Q].
        y_pred : np.ndarray, shape [N, 5]
            Predicted track states.
        """

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"

        # --------------------------------------------------
        # Predefined fit ranges and display ranges for each variable
        fit_ranges = {
            'x': (-0.6, 0.6),
            'y': (-2.0, 2.0),
            'tx': (-0.005, 0.005),
            'ty': (-0.01, 0.01),
            'Q': (-0.02, 0.02),
        }
        plot_ranges = {
            'x': (-1.5, 1.5),
            'y': (-5, 5),
            'tx': (-0.01, 0.01),
            'ty': (-0.05, 0.05),
            'Q': (-0.05, 0.05),
        }

        names = ["x", "y", "tx", "ty", "Q"]
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        fit_results = []

        for i, name in enumerate(names):
            # Residuals
            diff = y_pred[:, i] - y_true[:, i]
            diff = diff[np.isfinite(diff)]

            fit_min, fit_max = fit_ranges[name]
            plot_min, plot_max = plot_ranges[name]

            # Select values within fit range
            mask_fit = (diff >= fit_min) & (diff <= fit_max)
            diff_fit = diff[mask_fit]

            # Gaussian fit
            mu, sigma = norm.fit(diff_fit)
            fit_results.append((name, mu, sigma))

            # Histogram of residuals
            counts, bins, _ = axes[i].hist(
                diff,
                bins=60,
                range=(plot_min, plot_max),
                color='royalblue',
                alpha=0.7
            )

            # Gaussian curve scaled to histogram
            x_fit = np.linspace(plot_min, plot_max, 400)
            y_fit = norm.pdf(x_fit, mu, sigma) * len(diff_fit) * (bins[1] - bins[0])
            axes[i].plot(
                x_fit,
                y_fit,
                'r--',
                lw=3,
                label=f"μ={mu:+.3e}\nσ={sigma:.3e}"
            )

            axes[i].set_title(f"{name} residual")
            axes[i].set_xlabel("Prediction − Truth")
            axes[i].set_ylabel("Counts")
            axes[i].set_xlim(plot_min, plot_max)
            axes[i].legend(loc='upper left')

        plt.tight_layout()
        outname = os.path.join(self.print_dir, f"track_diff_{self.end_name}.png")
        plt.savefig(outname, dpi=200)
        plt.close()

        # --------------------------------------------------
        # Save Gaussian fit results to CSV
        df = pd.DataFrame(fit_results, columns=['Variable', 'Mean (μ)', 'Sigma (σ)'])
        csv_path = os.path.join(self.print_dir, f"track_diff_fit_{self.end_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved fit results to {csv_path}")

    # --------------------------------------------------
    def plot_pred_target(self, targets, preds):
        """
        Scatter plots of predicted vs. true values for each
        track parameter.

        Parameters
        ----------
        targets : np.ndarray, shape [N, 5]
            Ground-truth track states.
        preds : np.ndarray, shape [N, 5]
            Predicted track states.
        """
        components = ["x", "y", "tx", "ty", "Q"]
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, comp in enumerate(components):
            ax = axes[i]
            true_vals = targets[:, i]
            pred_vals = preds[:, i]

            # Scatter plot
            ax.scatter(true_vals, pred_vals, s=10, alpha=0.5, color='royalblue')

            # Ideal y = x reference line
            min_val = min(true_vals.min(), pred_vals.min())
            max_val = max(true_vals.max(), pred_vals.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

            ax.set_xlabel(f"True {comp}")
            ax.set_ylabel(f"Predicted {comp}")
            ax.set_title(f"{comp}: Predicted vs True")
            ax.grid(True)

        plt.tight_layout()
        outname = f"{self.print_dir}/track_pred_vs_true_{self.end_name}.png"
        plt.savefig(outname)
        plt.close()
