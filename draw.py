import matplotlib.pyplot as plt
import numpy as np

# --------------------------
# Step 1: Data Extraction & Preparation
# --------------------------
# Extract iteration and AUC data from training logs
fold_data = {
    "Fold 1": {
        "iterations": [0, 200, 400, 600, 671],
        "auc": [0.6743267, 0.7018071, 0.7026670, 0.7028689, 0.7029509]
    },
    "Fold 2": {
        "iterations": [0, 200, 400, 600, 800, 1000, 1051],
        "auc": [0.6777811, 0.7003444, 0.7012239, 0.7015609, 0.7016698, 0.7017887, 0.7017958]
    },
    "Fold 3": {
        "iterations": [0, 200, 400, 600, 800, 1000, 907],
        "auc": [0.6743363, 0.7005262, 0.7016584, 0.7020776, 0.7021902, 0.7022057, 0.7022433]
    },
    "Fold 4": {
        "iterations": [0, 200, 400, 600, 800, 1000, 957],
        "auc": [0.6786501, 0.7019023, 0.7030329, 0.7033354, 0.7034208, 0.7034459, 0.7034608]
    },
    "Fold 5": {
        "iterations": [0, 200, 400, 600, 800, 1000, 1030],
        "auc": [0.6788318, 0.7020959, 0.7031103, 0.7034619, 0.7035549, 0.7035730, 0.7035936]
    }
}

# Best performance info for each fold
best_info = {
    "Fold 1": {"iter": 671, "auc": 0.7029509},
    "Fold 2": {"iter": 1051, "auc": 0.7017958},
    "Fold 3": {"iter": 907, "auc": 0.7022433},
    "Fold 4": {"iter": 957, "auc": 0.7034608},
    "Fold 5": {"iter": 1030, "auc": 0.7035936}
}

# Color and line style configuration
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
styles = ["-", "--", "-.", ":", "-"]

# --------------------------
# Step 2: Plot 1 - AUC Convergence Trend Curve
# --------------------------
fig, ax = plt.subplots(figsize=(12, 8))

# Plot each fold's curve and mark best points
for i, (fold_name, data) in enumerate(fold_data.items()):
    its = np.array(data["iterations"])
    aucs = np.array(data["auc"])
    
    # Plot iteration-AUC curve
    ax.plot(
        its, aucs,
        color=colors[i],
        linestyle=styles[i],
        linewidth=2,
        label=f"{fold_name}",
        alpha=0.8
    )
    
    # Mark best performance point
    best_iter = best_info[fold_name]["iter"]
    best_auc = best_info[fold_name]["auc"]
    ax.scatter(
        best_iter, best_auc,
        color=colors[i],
        s=100,
        zorder=5,
        edgecolors="black",
        linewidth=1.5,
        label=f"{fold_name} (Best)"
    )
    
    # Annotate best performance info
    ax.annotate(
        f"({best_iter}, {best_auc:.6f})",
        xy=(best_iter, best_auc),
        xytext=(best_iter + 20, best_auc + 0.0002),
        fontsize=9,
        color=colors[i],
        bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.2)
    )

# Set axis labels and title
ax.set_xlabel("Iterations", fontsize=14, fontweight="bold")
ax.set_ylabel("AUC Score", fontsize=14, fontweight="bold")
ax.set_title("CatBoost 5-Fold Cross Validation - AUC Convergence Trend", fontsize=16, fontweight="bold", pad=20)

# Add grid
ax.grid(True, alpha=0.3, linestyle="--")

# Add legend
ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

# Set axis limits
ax.set_xlim(0, 1100)
ax.set_ylim(0.67, 0.71)

# Adjust layout
plt.tight_layout()

# Save and show plot
plt.savefig("catboost_new_auc_convergence_curve.png", dpi=300, bbox_inches="tight")
plt.show()

# --------------------------
# Step 3: Plot 2 - Best AUC Bar Chart
# --------------------------
fig, ax = plt.subplots(figsize=(10, 6))

# Extract data for bar chart
folds = list(best_info.keys())
best_aucs = [best_info[fold]["auc"] for fold in folds]
best_iters = [best_info[fold]["iter"] for fold in folds]

# Plot bar chart
bars = ax.bar(
    folds,
    best_aucs,
    color=colors,
    alpha=0.8,
    edgecolor="black",
    linewidth=1.5
)

# Annotate bar chart with values
for bar, auc, iter_num in zip(bars, best_aucs, best_iters):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.,
        height + 0.0002,
        f"AUC: {auc:.6f}\nIter: {iter_num}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

# Set axis labels and title
ax.set_xlabel("Cross Validation Fold", fontsize=14, fontweight="bold")
ax.set_ylabel("Best AUC Score", fontsize=14, fontweight="bold")
ax.set_title("CatBoost 5-Fold Cross Validation - Best AUC Performance Comparison", fontsize=16, fontweight="bold", pad=20)

# Add grid
ax.grid(True, axis="y", alpha=0.3, linestyle="--")

# Set y-axis limit
ax.set_ylim(0.70, 0.705)

# Adjust layout
plt.tight_layout()

# Save and show plot
plt.savefig("catboost_new_best_auc_bar_chart.png", dpi=300, bbox_inches="tight")
plt.show()