import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_benchmark_artifacts(artifact_dir="benchmark_artifacts"):
    artifact_dir = Path(artifact_dir)
    results_path = artifact_dir / "benchmark_results.npz"
    summary_path = artifact_dir / "benchmark_summary.json"
    metrics_path = artifact_dir / "model_classification_metrics.json"

    if not results_path.exists():
        raise FileNotFoundError(
            f"Run crystallographic_resnet_models.ipynb first. Missing: {results_path}"
        )

    results = np.load(results_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else []
    return results, summary, metrics


def print_summary(summary):
    if not summary:
        print("No benchmark summary JSON found.")
        return

    print("Benchmark summary")
    for row in summary:
        print(
            f"{row['model']}: "
            f"accuracy={row['accuracy']*100:.2f}%, "
            f"NLL={row['nll']:.4f}, "
            f"mean entropy={row['mean_predictive_entropy']:.4f}, "
            f"mean MI={row['mean_mutual_information']:.4f}"
        )


def print_model_metrics(metrics):
    if not metrics:
        print("No model classification metrics JSON found.")
        return

    try:
        import pandas as pd
        from IPython.display import display

        rows = [
            {
                "Model type": row["model_type"],
                "Accuracy (%)": row["accuracy_percent"],
                "Macro Precision (%)": row["macro_precision_percent"],
                "Macro Recall (%)": row["macro_recall_percent"],
                "Macro F1 (%)": row["macro_f1_percent"],
            }
            for row in metrics
        ]
        df = pd.DataFrame(rows)
        display(df.style.format({
            "Accuracy (%)": "{:.2f}",
            "Macro Precision (%)": "{:.2f}",
            "Macro Recall (%)": "{:.2f}",
            "Macro F1 (%)": "{:.2f}",
        }))
    except ImportError:
        header = (
            f"{'Model type':34s} {'Accuracy (%)':>13s} {'Macro Precision (%)':>20s} "
            f"{'Macro Recall (%)':>17s} {'Macro F1 (%)':>13s}"
        )
        print(header)
        print("-" * len(header))
        for row in metrics:
            print(
                f"{row['model_type']:34s} "
                f"{row['accuracy_percent']:13.2f} "
                f"{row['macro_precision_percent']:20.2f} "
                f"{row['macro_recall_percent']:17.2f} "
                f"{row['macro_f1_percent']:13.2f}"
            )


def print_model_metrics_latex(metrics):
    if not metrics:
        print("No model classification metrics JSON found.")
        return

    print(r"\begin{tabular}{lcccc}")
    print(r"\hline")
    print(r"Model type & Accuracy (\%) & Macro Precision (\%) & Macro Recall (\%) & Macro F1 (\%) \\")
    print(r"\hline")
    for row in metrics:
        print(
            f"{row['model_type']} & "
            f"{row['accuracy_percent']:.2f} & "
            f"{row['macro_precision_percent']:.2f} & "
            f"{row['macro_recall_percent']:.2f} & "
            f"{row['macro_f1_percent']:.2f} \\\\"
        )
    print(r"\hline")
    print(r"\end{tabular}")


def plot_training_history(results, output_path="training_loss_accuracy.png"):
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(results["epochs"], results["loss_cls_hist"], color="tab:blue", marker="o", linewidth=2, markersize=5, label="Data-driven")
    ax.plot(results["epochs"], results["loss_crystal_hist"], color="tab:orange", marker="s", linewidth=2, markersize=5, label="Crystallographic constrained")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.grid(True, alpha=0.6)
    ax.legend(frameon=True, loc="upper right")
    ax.text(-0.13, 1.03, "(a)", transform=ax.transAxes, fontsize=16, fontweight="bold")

    ax = axes[1]
    ax.plot(results["epochs"], results["acc_cls_hist"] * 100, color="tab:blue", marker="o", linewidth=2, markersize=5, label="Data-driven")
    ax.plot(results["epochs"], results["acc_crystal_hist"] * 100, color="tab:orange", marker="s", linewidth=2, markersize=5, label="Crystallographic constrained")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, alpha=0.6)
    ax.legend(frameon=True, loc="lower right")
    ax.text(-0.13, 1.03, "(b)", transform=ax.transAxes, fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_noise_robustness(
    results,
    output_path="noise_robustness_accuracy.png",
    train_thermal_mu=98.012,
    train_poisson_scale=98.0,
):
    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 20,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
    })

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    ax.plot(results["thermal_mu_sweep"], results["acc_cls_thermal"], linestyle="--", marker="s", linewidth=3, markersize=9, label="Data-driven")
    ax.plot(results["thermal_mu_sweep"], results["acc_crystal_thermal"], linestyle="-.", marker="o", linewidth=3, markersize=9, label="Crystallographic constrained")
    ax.axvline(x=train_thermal_mu, color="gray", linestyle=":", linewidth=3, label="Training point")
    ax.set_xlabel("Thermal noise mean")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="upper right")
    ax.text(-0.08, 1.01, "(a)", transform=ax.transAxes, fontsize=26, fontweight="bold")

    ax = axes[1]
    ax.plot(results["poisson_sweep"], results["acc_cls_poisson"], linestyle="--", marker="s", linewidth=3, markersize=9, label="Data-driven")
    ax.plot(results["poisson_sweep"], results["acc_crystal_poisson"], linestyle="-.", marker="o", linewidth=3, markersize=9, label="Crystallographic constrained")
    ax.axvline(x=train_poisson_scale, color="gray", linestyle=":", linewidth=3, label="Training point")
    ax.set_xlabel("Poisson scale")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    ax.text(-0.08, 1.01, "(b)", transform=ax.transAxes, fontsize=26, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_uncertainty(results, output_path="uncertainty_predictive_entropy_mi.png"):
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    series = [
        ("Data-driven", "data", "tab:blue"),
        ("Crystallographic constrained", "crystal", "tab:orange"),
    ]

    for label, prefix, color in series:
        axes[0].hist(results[f"{prefix}_predictive_entropy"], bins=40, density=True, histtype="stepfilled", alpha=0.22, color=color)
        axes[0].hist(results[f"{prefix}_predictive_entropy"], bins=40, density=True, histtype="step", linewidth=1.6, color=color, label=label)

        axes[1].hist(results[f"{prefix}_mutual_information"], bins=40, density=True, histtype="stepfilled", alpha=0.22, color=color)
        axes[1].hist(results[f"{prefix}_mutual_information"], bins=40, density=True, histtype="step", linewidth=1.6, color=color, label=label)

    axes[0].set_xlabel("Predictive Entropy")
    axes[0].set_ylabel("Density")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(frameon=True, loc="upper right")
    axes[0].text(-0.13, 1.03, "(a)", transform=axes[0].transAxes, fontsize=12, fontweight="bold")

    axes[1].set_xlabel("Mutual Information")
    axes[1].set_ylabel("Density")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend(frameon=True, loc="upper right")
    axes[1].text(-0.13, 1.03, "(b)", transform=axes[1].transAxes, fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_all(results, summary=None, metrics=None):
    if summary is not None:
        print_summary(summary)
    if metrics is not None:
        print_model_metrics(metrics)
    plot_training_history(results)
    plot_noise_robustness(results)
    plot_uncertainty(results)


def plot_experimental_prediction_comparison(
    crystal_predictions,
    data_predictions,
    output_path="experimental_burgers_prediction_comparison.png",
):
    labels = [Path(pred["image_path"]).stem.replace("expr_111refl", "img") for pred in crystal_predictions]
    x = np.arange(len(labels))

    crystal_conf = np.asarray([pred["confidence"] for pred in crystal_predictions])
    data_conf = np.asarray([pred["confidence"] for pred in data_predictions])
    crystal_b = [str(pred["predicted_burgers"]) for pred in crystal_predictions]
    data_b = [str(pred["predicted_burgers"]) for pred in data_predictions]

    fig, ax = plt.subplots(figsize=(11, 4.8))
    width = 0.36

    ax.bar(
        x - width / 2,
        crystal_conf,
        width,
        color="tab:orange",
        alpha=0.85,
        label="Crystallographic constrained",
    )
    ax.bar(
        x + width / 2,
        data_conf,
        width,
        color="tab:blue",
        alpha=0.85,
        label="Data-driven",
    )

    for i, (cb, db) in enumerate(zip(crystal_b, data_b)):
        ax.text(
            x[i] - width / 2,
            crystal_conf[i] + 0.025,
            cb,
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=8,
        )
        ax.text(
            x[i] + width / 2,
            data_conf[i] + 0.025,
            db,
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Prediction Probability")
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_experimental_topk_heatmaps(
    crystal_predictions,
    data_predictions,
    output_path="experimental_burgers_topk_heatmaps.png",
):
    image_labels = [Path(pred["image_path"]).stem.replace("expr_111refl", "img") for pred in crystal_predictions]
    burgers_labels = sorted(
        {
            str(cand["burgers"])
            for pred in crystal_predictions + data_predictions
            for cand in pred["topk"]
        }
    )
    b_to_idx = {label: i for i, label in enumerate(burgers_labels)}

    crystal_matrix = np.zeros((len(burgers_labels), len(image_labels)))
    data_matrix = np.zeros((len(burgers_labels), len(image_labels)))

    for j, pred in enumerate(crystal_predictions):
        for cand in pred["topk"]:
            crystal_matrix[b_to_idx[str(cand["burgers"])], j] = cand["probability"]

    for j, pred in enumerate(data_predictions):
        for cand in pred["topk"]:
            data_matrix[b_to_idx[str(cand["burgers"])], j] = cand["probability"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)

    for ax, matrix, title in [
        (axes[0], crystal_matrix, "Crystallographic constrained"),
        (axes[1], data_matrix, "Data-driven"),
    ]:
        im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(image_labels)))
        ax.set_xticklabels(image_labels, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(burgers_labels)))
        ax.set_yticklabels(burgers_labels)
        ax.set_xlabel("Experimental image")

        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                if matrix[y, x] > 0:
                    ax.text(x, y, f"{matrix[y, x]:.2f}", ha="center", va="center", color="white", fontsize=8)

    axes[0].set_ylabel("Burgers vector")
    fig.colorbar(im, ax=axes, label="Probability", shrink=0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
