import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.gridspec as gridspec

# Plot the pixesl scatter of true FDF vs predicted FDF
def scatter_plot(gt_flat, pred_flat, filename, title="Scatter"):
    pairs = np.vstack([gt_flat.flatten(), pred_flat.flatten()]).T
    plt.rcParams['font.size'] = '12'
    plt.rcParams['font.family'] = 'Liberation Sans'
    plt.rcParams['font.weight'] = 'bold'
    plt.figure(figsize=(4, 4))
    plt.scatter(pairs[:, 0], pairs[:, 1], \
                facecolors=(0.5, 0.8, 1, 0.4), edgecolors='blue', linewidths=1.5, s=30)
    mn, mx = pairs.min(), pairs.max()
    plt.plot([mn, mx], [mn, mx], 'k--', label='Ideal')
    plt.xlabel("True", fontsize=12, fontweight='bold')
    plt.ylabel("Predicted", fontsize=12, fontweight='bold')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)

# Plot heatmap comparison of original and reconstructed FDF
def heatmap_pair(gt_img, pred_img, filename):
    vmin, vmax = min(gt_img.min(), pred_img.min()), max(gt_img.max(), pred_img.max())
    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])

    ax0 = plt.subplot(gs[0])
    sns.heatmap(gt_img, cmap='viridis', vmin=vmin, vmax=vmax, cbar=False, ax=ax0)
    ax0.invert_yaxis()
    ax0.set_title("Original", fontsize=12, fontweight='bold')

    ax1 = plt.subplot(gs[1])
    sns.heatmap(pred_img, cmap='viridis', vmin=vmin, vmax=vmax, cbar=False, ax=ax1)
    ax1.invert_yaxis()
    ax1.set_title("Reconstructed", fontsize=12, fontweight='bold')

    cax = plt.subplot(gs[2])
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cax)
    plt.tight_layout()
    plt.savefig(filename)
