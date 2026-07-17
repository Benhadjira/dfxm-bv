import os
import re
import random
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import ConcatDataset, Dataset, DataLoader, random_split, Subset
import matplotlib.pyplot as plt


# ======================================================
# Reproducibility
# ======================================================
def set_deterministic(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================
# Parse labels from filename
# ======================================================
def extract_vector(filename, label):
    m = re.search(rf"{label}_(-?\d-?\d-?\d)", filename)
    if m:
        nums = re.findall(r"-?\d", m.group(1))
        return np.array(list(map(int, nums)), dtype=np.float32)
    return None


# ======================================================
# Noise model
# ======================================================
def add_poisson_and_student_t_noise(
    img,
    poisson_scale=98.0,
    thermal_mu=98.012,
    nu=13.067,
    rho=3.095,
):
    img = img.astype(np.float32)
    img_nonneg = np.clip(img, a_min=0.0, a_max=None)

    lam = img_nonneg * poisson_scale
    img_poisson = np.random.poisson(lam).astype(np.float32) / poisson_scale

    t_sample = np.random.standard_t(df=nu, size=img_poisson.shape).astype(np.float32)
    thermal = thermal_mu + rho * t_sample

    return (img_poisson + thermal).astype(np.float32)


# ======================================================
# Dataset
# ======================================================
class NpyImageDataset(Dataset):
    def __init__(
        self,
        image_dir,
        poisson_scale=98.0,
        thermal_mu=98.012,
        nu=13.067,
        rho=3.095,
        apply_noise=True,
        class_map=None,
        class_to_burgers=None,
    ):
        self.image_dir = image_dir
        self.poisson_scale = float(poisson_scale)
        self.thermal_mu = float(thermal_mu)
        self.nu = float(nu)
        self.rho = float(rho)
        self.apply_noise = apply_noise

        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".npy")])
        if len(self.image_files) == 0:
            raise ValueError(f"No .npy files found in {image_dir}")

        self.labels = []
        for f in self.image_files:
            b = extract_vector(f, "b")
            n = extract_vector(f, "n")
            if b is None or n is None:
                raise ValueError(f"Missing vectors in filename: {f}")
            self.labels.append({"b": b, "n": n})

        if class_map is None or class_to_burgers is None:
            b_vecs = [tuple(lbl["b"]) for lbl in self.labels]
            self.class_map, self.class_to_burgers, self.num_classes = self._make_class_map(b_vecs)
        else:
            self.class_map = class_map
            self.class_to_burgers = class_to_burgers
            self.num_classes = len(class_to_burgers)

        for lbl in self.labels:
            key = tuple(lbl["b"])
            if key not in self.class_map:
                raise ValueError(f"Burgers vector {key} not found in shared class_map.")
            lbl["class"] = self.class_map[key]

        self.class_b_tensor = torch.tensor(
            np.stack([self.class_to_burgers[i] for i in range(self.num_classes)], axis=0),
            dtype=torch.float32,
        )
        self.class_b_tensor = F.normalize(self.class_b_tensor, dim=1)

    def _make_class_map(self, b_vecs):
        uniq = sorted(set(b_vecs))
        class_map = {}
        class_to_burgers = []
        seen = set()
        idx = 0

        for v in uniq:
            if v in seen:
                continue

            neg_v = tuple(-x for x in v)

            class_map[v] = idx
            if neg_v in uniq:
                class_map[neg_v] = idx

            class_to_burgers.append(np.array(v, dtype=np.float32))

            seen.add(v)
            seen.add(neg_v)
            idx += 1

        return class_map, class_to_burgers, idx

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        path = os.path.join(self.image_dir, filename)
        img = np.load(path).astype(np.float32)

        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)

        if self.apply_noise:
            img = add_poisson_and_student_t_noise(
                img=img,
                poisson_scale=self.poisson_scale,
                thermal_mu=self.thermal_mu,
                nu=self.nu,
                rho=self.rho,
            )

        img = np.clip(img, a_min=1e-5, a_max=None)
        img = np.log(img)

        t = torch.tensor(img, dtype=torch.float32)

        t = t - t.min()
        if t.max() > 0:
            t = t / t.max()

        lbl = self.labels[idx]

        b = torch.tensor(lbl["b"], dtype=torch.float32)
        n = torch.tensor(lbl["n"], dtype=torch.float32)

        b = F.normalize(b, dim=0)
        n = F.normalize(n, dim=0)

        dots = torch.abs(torch.matmul(self.class_b_tensor, n))
        valid_mask = (dots < 1e-5).float()

        valid_mask[lbl["class"]] = 1.0

        return {
            "image": t,
            "b": b,
            "n": n,
            "class": torch.tensor(lbl["class"], dtype=torch.long),
            "valid_mask": valid_mask,
        }


# ======================================================
# Models
# ======================================================
class ResNet18Gray(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        base = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )

        base.conv1 = nn.Conv2d(
            1,
            base.conv1.out_channels,
            kernel_size=base.conv1.kernel_size,
            stride=base.conv1.stride,
            padding=base.conv1.padding,
            bias=False,
        )

        if pretrained:
            base.conv1.weight.data = base.conv1.weight.data.mean(dim=1, keepdim=True)

        self.features = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(base.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ResNetCrystal(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        base = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )

        base.conv1 = nn.Conv2d(
            1,
            base.conv1.out_channels,
            kernel_size=base.conv1.kernel_size,
            stride=base.conv1.stride,
            padding=base.conv1.padding,
            bias=False,
        )

        if pretrained:
            base.conv1.weight.data = base.conv1.weight.data.mean(dim=1, keepdim=True)

        self.features = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(base.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ======================================================
# Strong crystallographic tools
# ======================================================
def apply_crystal_mask(logits, valid_mask):
    return logits.masked_fill(valid_mask == 0, -1e9)


def crystallographic_loss_v2(
    logits,
    targets,
    valid_mask,
    alpha_invalid=0.2,
    beta_margin=0.5,
    margin=2.0,
    label_smoothing=0.05,
):
    eps = 1e-12
    invalid_mask = 1.0 - valid_mask

    masked_logits = apply_crystal_mask(logits, valid_mask)
    log_probs_valid = F.log_softmax(masked_logits, dim=1)

    num_valid = valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)

    with torch.no_grad():
        valid_smooth = valid_mask / num_valid

        target_dist = label_smoothing * valid_smooth
        target_dist.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing)

        target_dist = target_dist * valid_mask
        target_dist = target_dist / target_dist.sum(dim=1, keepdim=True).clamp_min(eps)

    masked_ce = -(target_dist * log_probs_valid).sum(dim=1).mean()

    probs = F.softmax(logits, dim=1)
    invalid_prob = (probs * invalid_mask).sum(dim=1)
    invalid_penalty = invalid_prob.mean()

    target_logits = logits.gather(1, targets.unsqueeze(1))

    margin_terms = torch.clamp(
        margin - (target_logits - logits),
        min=0.0,
    )

    margin_terms = margin_terms * invalid_mask

    invalid_count = invalid_mask.sum(dim=1).clamp_min(1.0)
    margin_penalty = (margin_terms.sum(dim=1) / invalid_count).mean()

    loss = masked_ce + alpha_invalid * invalid_penalty + beta_margin * margin_penalty

    return loss, masked_ce.detach(), invalid_penalty.detach(), margin_penalty.detach()


# ======================================================
# Training / evaluation
# ======================================================
def train_epoch_cls(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["class"].to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.numel()

    return total_loss / len(loader), correct / total


@torch.no_grad()
def eval_epoch_cls(model, loader, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["class"].to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.numel()

    return total_loss / len(loader), correct / total


def train_epoch_crystal(
    model,
    loader,
    optimizer,
    device,
    alpha_invalid=0.2,
    beta_margin=0.5,
    margin=2.0,
    label_smoothing=0.05,
):
    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_inv = 0.0
    total_margin = 0.0
    correct = 0
    total = 0

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["class"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        optimizer.zero_grad()

        logits = model(x)

        loss, ce_term, inv_term, margin_term = crystallographic_loss_v2(
            logits=logits,
            targets=y,
            valid_mask=valid_mask,
            alpha_invalid=alpha_invalid,
            beta_margin=beta_margin,
            margin=margin,
            label_smoothing=label_smoothing,
        )

        loss.backward()
        optimizer.step()

        masked_logits = apply_crystal_mask(logits, valid_mask)

        total_loss += loss.item()
        total_ce += ce_term.item()
        total_inv += inv_term.item()
        total_margin += margin_term.item()

        correct += (masked_logits.argmax(dim=1) == y).sum().item()
        total += y.numel()

    return (
        total_loss / len(loader),
        correct / total,
        total_ce / len(loader),
        total_inv / len(loader),
        total_margin / len(loader),
    )


@torch.no_grad()
def eval_epoch_crystal(
    model,
    loader,
    device,
    alpha_invalid=0.2,
    beta_margin=0.5,
    margin=2.0,
    label_smoothing=0.05,
):
    model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_inv = 0.0
    total_margin = 0.0
    correct = 0
    total = 0

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["class"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        logits = model(x)

        loss, ce_term, inv_term, margin_term = crystallographic_loss_v2(
            logits=logits,
            targets=y,
            valid_mask=valid_mask,
            alpha_invalid=alpha_invalid,
            beta_margin=beta_margin,
            margin=margin,
            label_smoothing=label_smoothing,
        )

        masked_logits = apply_crystal_mask(logits, valid_mask)

        total_loss += loss.item()
        total_ce += ce_term.item()
        total_inv += inv_term.item()
        total_margin += margin_term.item()

        correct += (masked_logits.argmax(dim=1) == y).sum().item()
        total += y.numel()

    return (
        total_loss / len(loader),
        correct / total,
        total_ce / len(loader),
        total_inv / len(loader),
        total_margin / len(loader),
    )


@torch.no_grad()
def collect_predictive_uncertainty(
    model,
    loader,
    device,
    crystal=False,
    mc_passes=8,
    eps=1e-12,
):
    """Estimate uncertainty from repeated noisy test-time predictions.

    Predictive entropy uses the mean probability over repeated passes.
    Mutual information is H(E[p]) - E[H(p)], so it is high when the
    model's predictions change across stochastic noise realizations.
    """
    was_training = model.training
    model.eval()

    pass_probs = []
    targets = None

    for _ in range(mc_passes):
        probs_this_pass = []
        targets_this_pass = []

        for batch in loader:
            x = batch["image"].to(device)
            y = batch["class"].to(device)
            logits = model(x)

            if crystal:
                valid_mask = batch["valid_mask"].to(device)
                logits = apply_crystal_mask(logits, valid_mask)

            probs_this_pass.append(F.softmax(logits, dim=1).cpu())
            targets_this_pass.append(y.cpu())

        pass_probs.append(torch.cat(probs_this_pass, dim=0))
        if targets is None:
            targets = torch.cat(targets_this_pass, dim=0)

    probs = torch.stack(pass_probs, dim=0)
    mean_probs = probs.mean(dim=0)
    predictive_entropy = -(mean_probs * (mean_probs + eps).log()).sum(dim=1)
    expected_entropy = -(probs * (probs + eps).log()).sum(dim=2).mean(dim=0)
    mutual_information = predictive_entropy - expected_entropy

    pred = mean_probs.argmax(dim=1)
    confidence = mean_probs.max(dim=1).values
    accuracy = (pred == targets).float().mean().item()
    nll = -(mean_probs[torch.arange(targets.numel()), targets] + eps).log().mean().item()

    if was_training:
        model.train()

    return {
        "predictive_entropy": predictive_entropy.numpy(),
        "mutual_information": mutual_information.numpy(),
        "confidence": confidence.numpy(),
        "accuracy": accuracy,
        "nll": nll,
    }


def summarize_uncertainty(name, uncertainty):
    pe = uncertainty["predictive_entropy"]
    mi = uncertainty["mutual_information"]
    conf = uncertainty["confidence"]
    return {
        "model": name,
        "accuracy": uncertainty["accuracy"],
        "nll": uncertainty["nll"],
        "mean_confidence": float(np.mean(conf)),
        "mean_predictive_entropy": float(np.mean(pe)),
        "median_predictive_entropy": float(np.median(pe)),
        "mean_mutual_information": float(np.mean(mi)),
        "median_mutual_information": float(np.median(mi)),
    }


def print_benchmark_table(rows):
    print("\n=== Benchmark summary at training noise point ===")
    header = (
        f"{'Model':34s} {'Acc (%)':>8s} {'NLL':>8s} {'Conf':>8s} "
        f"{'Entropy':>10s} {'MI':>10s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['model']:34s} "
            f"{row['accuracy'] * 100:8.2f} "
            f"{row['nll']:8.4f} "
            f"{row['mean_confidence']:8.4f} "
            f"{row['mean_predictive_entropy']:10.4f} "
            f"{row['mean_mutual_information']:10.4f}"
        )


@torch.no_grad()
def collect_class_predictions(model, loader, device, crystal=False):
    model.eval()

    y_true = []
    y_pred = []

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["class"].to(device)
        logits = model(x)

        if crystal:
            valid_mask = batch["valid_mask"].to(device)
            logits = apply_crystal_mask(logits, valid_mask)

        y_true.append(y.cpu().numpy())
        y_pred.append(logits.argmax(dim=1).cpu().numpy())

    return np.concatenate(y_true), np.concatenate(y_pred)


def classification_metrics(y_true, y_pred, labels):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    accuracy = float((y_true == y_pred).mean()) if y_true.size else 0.0
    precisions = []
    recalls = []
    f1s = []

    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
    }


def make_model_metrics_table(data_y_true, data_y_pred, crystal_y_true, crystal_y_pred, num_classes):
    labels = np.arange(num_classes)
    rows = []

    for model_type, y_true, y_pred in [
        ("Data-driven", data_y_true, data_y_pred),
        ("Crystallographic constrained", crystal_y_true, crystal_y_pred),
    ]:
        metrics = classification_metrics(y_true, y_pred, labels)
        rows.append(
            {
                "model_type": model_type,
                "accuracy_percent": 100.0 * metrics["accuracy"],
                "macro_precision_percent": 100.0 * metrics["macro_precision"],
                "macro_recall_percent": 100.0 * metrics["macro_recall"],
                "macro_f1_percent": 100.0 * metrics["macro_f1"],
            }
        )

    return rows


def print_model_metrics_table(rows):
    print("\n=== Test-set classification metrics ===")
    header = (
        f"{'Model type':34s} {'Accuracy (%)':>13s} {'Macro Precision (%)':>20s} "
        f"{'Macro Recall (%)':>17s} {'Macro F1 (%)':>13s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['model_type']:34s} "
            f"{row['accuracy_percent']:13.2f} "
            f"{row['macro_precision_percent']:20.2f} "
            f"{row['macro_recall_percent']:17.2f} "
            f"{row['macro_f1_percent']:13.2f}"
        )


def plot_uncertainty_distributions(data_uncertainty, crystal_uncertainty):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    series = [
        ("Data-driven", data_uncertainty, "tab:blue"),
        ("Crystallographic constrained", crystal_uncertainty, "tab:orange"),
    ]

    for label, uncertainty, color in series:
        axes[0].hist(
            uncertainty["predictive_entropy"],
            bins=40,
            density=True,
            histtype="stepfilled",
            alpha=0.22,
            color=color,
        )
        axes[0].hist(
            uncertainty["predictive_entropy"],
            bins=40,
            density=True,
            histtype="step",
            linewidth=2,
            color=color,
            label=label,
        )

        axes[1].hist(
            uncertainty["mutual_information"],
            bins=40,
            density=True,
            histtype="stepfilled",
            alpha=0.22,
            color=color,
        )
        axes[1].hist(
            uncertainty["mutual_information"],
            bins=40,
            density=True,
            histtype="step",
            linewidth=2,
            color=color,
            label=label,
        )

    axes[0].set_xlabel("Predictive Entropy")
    axes[0].set_ylabel("Density")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(frameon=True, loc="upper right")
    axes[0].text(-0.13, 1.03, "(a)", transform=axes[0].transAxes, fontsize=14, fontweight="bold")

    axes[1].set_xlabel("Mutual Information")
    axes[1].set_ylabel("Density")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend(frameon=True, loc="upper right")
    axes[1].text(-0.13, 1.03, "(b)", transform=axes[1].transAxes, fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig("uncertainty_predictive_entropy_mi.png", dpi=300, bbox_inches="tight")
    plt.show()




from dataclasses import dataclass, field


@dataclass
class BenchmarkConfig:
    data_path: str = "/data/projects/engage_id03/int_fcc_002/"
    artifact_dir: str = "benchmark_artifacts"
    seed: int = 42
    batch_size: int = 32
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-5
    uncertainty_mc_passes: int = 8
    nu: float = 13.067
    rho: float = 3.095
    train_thermal_mu: float = 98.012
    train_poisson_scale: float = 98.0
    alpha_invalid: float = 0.2
    beta_margin: float = 0.5
    margin: float = 2.0
    label_smoothing: float = 0.05
    pretrained: bool = True
    num_workers: int = 0
    thermal_mu_sweep: tuple = (50, 70, 85, 98.012, 120, 180, 250, 300, 400, 500)
    poisson_sweep: tuple = (50, 70, 85, 98, 120, 180, 250, 300, 400, 500)


def _make_dataset(config, poisson_scale, thermal_mu, class_map=None, class_to_burgers=None):
    return NpyImageDataset(
        image_dir=config.data_path,
        poisson_scale=poisson_scale,
        thermal_mu=thermal_mu,
        nu=config.nu,
        rho=config.rho,
        apply_noise=True,
        class_map=class_map,
        class_to_burgers=class_to_burgers,
    )


def preprocess_experimental_image(image_path):
    img = np.load(image_path).astype(np.float32)

    if img.ndim == 2:
        img = np.expand_dims(img, axis=0)

    img = np.clip(img, a_min=1e-5, a_max=None)
    img = np.log(img)

    t = torch.tensor(img, dtype=torch.float32)
    t = t - t.min()
    if t.max() > 0:
        t = t / t.max()

    return t


def make_valid_mask_from_normal(class_to_burgers, normal, eps=1e-5):
    class_b_tensor = torch.tensor(
        np.asarray(class_to_burgers, dtype=np.float32),
        dtype=torch.float32,
    )
    class_b_tensor = F.normalize(class_b_tensor, dim=1)

    n = torch.tensor(normal, dtype=torch.float32)
    n = F.normalize(n, dim=0)

    dots = torch.abs(torch.matmul(class_b_tensor, n))
    return (dots < eps).float()


def load_class_to_burgers(config=None, artifact_dir=None):
    config = config or BenchmarkConfig()
    artifact_dir = Path(artifact_dir or config.artifact_dir)
    results_path = artifact_dir / "benchmark_results.npz"

    if results_path.exists():
        results = np.load(results_path)
        if "class_to_burgers" in results:
            return results["class_to_burgers"].astype(np.float32)

    base_ds = _make_dataset(
        config,
        poisson_scale=config.train_poisson_scale,
        thermal_mu=config.train_thermal_mu,
    )
    return np.asarray(base_ds.class_to_burgers, dtype=np.float32)


@torch.no_grad()
def predict_burgers_vectors(
    image_paths,
    normal=(1, -1, -1),
    config=None,
    artifact_dir=None,
    checkpoint_name=None,
    model_type="crystallographic",
    topk=3,
):
    config = config or BenchmarkConfig()
    device = get_device()
    artifact_dir = Path(artifact_dir or config.artifact_dir)
    if checkpoint_name is None:
        checkpoint_name = (
            "crystallographic_resnet18.pt"
            if model_type == "crystallographic"
            else "data_driven_resnet18.pt"
        )
    checkpoint_path = artifact_dir / checkpoint_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}. Run crystallographic_resnet_models.ipynb first."
        )

    class_to_burgers = load_class_to_burgers(config=config, artifact_dir=artifact_dir)
    if model_type == "crystallographic":
        model = ResNetCrystal(num_classes=len(class_to_burgers), pretrained=False).to(device)
    elif model_type == "data":
        model = ResNet18Gray(num_classes=len(class_to_burgers), pretrained=False).to(device)
    else:
        raise ValueError("model_type must be 'crystallographic' or 'data'.")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    valid_mask = make_valid_mask_from_normal(class_to_burgers, normal).to(device)
    if model_type == "crystallographic" and valid_mask.sum().item() == 0:
        raise ValueError(f"No crystallographically valid Burgers classes for normal={normal}.")

    predictions = []
    for image_path in image_paths:
        image_path = Path(image_path)
        x = preprocess_experimental_image(image_path).unsqueeze(0).to(device)
        logits = model(x)
        if model_type == "crystallographic":
            logits = apply_crystal_mask(logits, valid_mask.unsqueeze(0))
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()

        k = min(topk, probs.numel())
        values, indices = torch.topk(probs, k=k)
        pred_idx = int(indices[0].item())
        predictions.append(
            {
                "image_path": str(image_path),
                "model_type": model_type,
                "normal": tuple(normal),
                "predicted_class": pred_idx,
                "predicted_burgers": class_to_burgers[pred_idx].astype(int).tolist(),
                "confidence": float(values[0].item()),
                "topk": [
                    {
                        "class": int(idx.item()),
                        "burgers": class_to_burgers[int(idx.item())].astype(int).tolist(),
                        "probability": float(val.item()),
                    }
                    for val, idx in zip(values, indices)
                ],
            }
        )

    return predictions


def _make_shared_burgers_class_map(data_paths):
    burgers_vectors = []
    for data_path in data_paths:
        dataset = NpyImageDataset(image_dir=str(data_path), apply_noise=False)
        burgers_vectors.extend(tuple(label["b"]) for label in dataset.labels)

    unique_vectors = sorted(set(burgers_vectors))
    class_map = {}
    class_to_burgers = []
    seen = set()

    for vector in unique_vectors:
        if vector in seen:
            continue

        negative = tuple(-value for value in vector)
        class_index = len(class_to_burgers)
        class_map[vector] = class_index
        if negative in unique_vectors:
            class_map[negative] = class_index

        class_to_burgers.append(np.asarray(vector, dtype=np.float32))
        seen.add(vector)
        seen.add(negative)

    return class_map, class_to_burgers


def _make_multi_path_dataset(
    data_paths,
    config,
    class_map,
    class_to_burgers,
):
    datasets = [
        NpyImageDataset(
            image_dir=str(data_path),
            poisson_scale=config.train_poisson_scale,
            thermal_mu=config.train_thermal_mu,
            nu=config.nu,
            rho=config.rho,
            apply_noise=True,
            class_map=class_map,
            class_to_burgers=class_to_burgers,
        )
        for data_path in data_paths
    ]
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def _confusion_matrix_counts(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_class, predicted_class in zip(y_true, y_pred):
        matrix[int(true_class), int(predicted_class)] += 1
    return matrix


def _row_normalize_confusion_matrix(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix),
        where=row_sums > 0,
    ) * 100.0


def run_crystallographic_only_benchmark(data_paths, config=None):
    """Train and evaluate only the crystallographic-constrained model.

    ``data_paths`` may contain one reflection dataset or multiple reflection
    datasets. All folders share one Burgers-vector class map before they are
    concatenated and split into training and test subsets.
    """
    config = config or BenchmarkConfig()
    data_paths = [Path(path) for path in data_paths]
    if not data_paths:
        raise ValueError("data_paths must contain at least one dataset folder.")

    for data_path in data_paths:
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset folder does not exist: {data_path}")

    set_deterministic(config.seed)
    device = get_device()
    print(f"Using device: {device}")
    print("Datasets:")
    for data_path in data_paths:
        print(f"  {data_path}")

    artifact_dir = Path(config.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    class_map, class_to_burgers = _make_shared_burgers_class_map(data_paths)
    num_classes = len(class_to_burgers)
    print(f"num_classes = {num_classes}")

    base_dataset = _make_multi_path_dataset(
        data_paths,
        config,
        class_map,
        class_to_burgers,
    )
    n_train = int(0.8 * len(base_dataset))
    n_test = len(base_dataset) - n_train
    if n_train == 0 or n_test == 0:
        raise ValueError(
            f"Dataset is too small for an 80/20 split: {len(base_dataset)} samples."
        )

    split_generator = torch.Generator().manual_seed(config.seed)
    train_subset, test_subset = random_split(
        base_dataset,
        [n_train, n_test],
        generator=split_generator,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = ResNetCrystal(
        num_classes=num_classes,
        pretrained=config.pretrained,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
        "train_cross_entropy": [],
        "train_invalid_penalty": [],
        "train_margin_penalty": [],
    }

    for epoch in range(config.epochs):
        train_loss, train_accuracy, ce, invalid, margin_term = train_epoch_crystal(
            model,
            train_loader,
            optimizer,
            device,
            alpha_invalid=config.alpha_invalid,
            beta_margin=config.beta_margin,
            margin=config.margin,
            label_smoothing=config.label_smoothing,
        )
        test_loss, test_accuracy, _, _, _ = eval_epoch_crystal(
            model,
            test_loader,
            device,
            alpha_invalid=config.alpha_invalid,
            beta_margin=config.beta_margin,
            margin=config.margin,
            label_smoothing=config.label_smoothing,
        )
        scheduler.step(test_accuracy)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)
        history["train_cross_entropy"].append(ce)
        history["train_invalid_penalty"].append(invalid)
        history["train_margin_penalty"].append(margin_term)

        print(
            f"Epoch {epoch + 1:03d}/{config.epochs:03d} | "
            f"train loss={train_loss:.4f} acc={train_accuracy * 100:6.2f}% | "
            f"test loss={test_loss:.4f} acc={test_accuracy * 100:6.2f}%"
        )

    y_true, y_pred = collect_class_predictions(
        model,
        test_loader,
        device,
        crystal=True,
    )
    metrics = classification_metrics(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
    )
    metrics_row = {
        "model_type": "Crystallographic constrained",
        "accuracy_percent": metrics["accuracy"] * 100.0,
        "macro_precision_percent": metrics["macro_precision"] * 100.0,
        "macro_recall_percent": metrics["macro_recall"] * 100.0,
        "macro_f1_percent": metrics["macro_f1"] * 100.0,
    }

    confusion_counts = _confusion_matrix_counts(y_true, y_pred, num_classes)
    confusion_percent = _row_normalize_confusion_matrix(confusion_counts)

    np.savez(
        artifact_dir / "cc_only_results.npz",
        epochs=np.arange(1, config.epochs + 1),
        train_loss=np.asarray(history["train_loss"]),
        train_accuracy=np.asarray(history["train_accuracy"]),
        test_loss=np.asarray(history["test_loss"]),
        test_accuracy=np.asarray(history["test_accuracy"]),
        train_cross_entropy=np.asarray(history["train_cross_entropy"]),
        train_invalid_penalty=np.asarray(history["train_invalid_penalty"]),
        train_margin_penalty=np.asarray(history["train_margin_penalty"]),
        y_true=y_true,
        y_pred=y_pred,
        confusion_counts=confusion_counts,
        confusion_percent=confusion_percent,
        class_to_burgers=np.asarray(class_to_burgers, dtype=np.float32),
    )
    with (artifact_dir / "cc_only_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics_row, file, indent=2)
    with (artifact_dir / "cc_only_config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "data_paths": [str(path) for path in data_paths],
                "train_samples": n_train,
                "test_samples": n_test,
                "num_classes": num_classes,
            },
            file,
            indent=2,
        )

    torch.save(model.state_dict(), artifact_dir / "crystallographic_resnet18.pt")
    print_model_metrics_table([metrics_row])
    print(f"Saved CC-only artifacts to: {artifact_dir.resolve()}")

    return {
        "artifact_dir": artifact_dir,
        "model": model,
        "metrics": metrics_row,
        "history": history,
        "y_true": y_true,
        "y_pred": y_pred,
        "confusion_counts": confusion_counts,
        "confusion_percent": confusion_percent,
        "class_to_burgers": np.asarray(class_to_burgers, dtype=np.float32),
    }


def run_benchmark(config=None):
    config = config or BenchmarkConfig()
    set_deterministic(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    artifact_dir = Path(config.artifact_dir)
    artifact_dir.mkdir(exist_ok=True)

    base_ds = _make_dataset(
        config,
        poisson_scale=config.train_poisson_scale,
        thermal_mu=config.train_thermal_mu,
    )
    print(f"num_classes = {base_ds.num_classes}")

    n_train = int(0.8 * len(base_ds))
    n_test = len(base_ds) - n_train
    train_subset, test_subset = random_split(
        base_ds,
        [n_train, n_test],
        generator=torch.Generator().manual_seed(config.seed),
    )

    train_indices = train_subset.indices
    test_indices = test_subset.indices
    shared_class_map = base_ds.class_map
    shared_class_to_burgers = base_ds.class_to_burgers

    train_ds = Subset(
        _make_dataset(
            config,
            poisson_scale=config.train_poisson_scale,
            thermal_mu=config.train_thermal_mu,
            class_map=shared_class_map,
            class_to_burgers=shared_class_to_burgers,
        ),
        train_indices,
    )
    test_ds_same = Subset(
        _make_dataset(
            config,
            poisson_scale=config.train_poisson_scale,
            thermal_mu=config.train_thermal_mu,
            class_map=shared_class_map,
            class_to_burgers=shared_class_to_burgers,
        ),
        test_indices,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    test_loader_same = DataLoader(
        test_ds_same,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model_cls = ResNet18Gray(num_classes=base_ds.num_classes, pretrained=config.pretrained).to(device)
    model_crystal = ResNetCrystal(num_classes=base_ds.num_classes, pretrained=config.pretrained).to(device)

    opt_cls = torch.optim.Adam(model_cls.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    opt_crystal = torch.optim.Adam(model_crystal.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_cls, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )
    scheduler_crystal = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_crystal, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )

    history = {
        "loss_cls_hist": [],
        "loss_crystal_hist": [],
        "acc_cls_hist": [],
        "acc_crystal_hist": [],
        "test_loss_cls_hist": [],
        "test_loss_crystal_hist": [],
        "test_acc_cls_hist": [],
        "test_acc_crystal_hist": [],
    }

    print("\n=== Training at thermal_mu=98.012 and poisson_scale=98 ===")
    for ep in range(1, config.epochs + 1):
        tr_loss_cls, tr_acc_cls = train_epoch_cls(model_cls, train_loader, opt_cls, device)
        te_loss_cls, te_acc_cls = eval_epoch_cls(model_cls, test_loader_same, device)

        tr_loss_cr, tr_acc_cr, tr_ce_cr, tr_inv_cr, tr_margin_cr = train_epoch_crystal(
            model_crystal,
            train_loader,
            opt_crystal,
            device,
            alpha_invalid=config.alpha_invalid,
            beta_margin=config.beta_margin,
            margin=config.margin,
            label_smoothing=config.label_smoothing,
        )
        te_loss_cr, te_acc_cr, te_ce_cr, te_inv_cr, te_margin_cr = eval_epoch_crystal(
            model_crystal,
            test_loader_same,
            device,
            alpha_invalid=config.alpha_invalid,
            beta_margin=config.beta_margin,
            margin=config.margin,
            label_smoothing=config.label_smoothing,
        )

        scheduler_cls.step(te_acc_cls)
        scheduler_crystal.step(te_acc_cr)

        history["loss_cls_hist"].append(tr_loss_cls)
        history["loss_crystal_hist"].append(tr_loss_cr)
        history["acc_cls_hist"].append(tr_acc_cls)
        history["acc_crystal_hist"].append(tr_acc_cr)
        history["test_loss_cls_hist"].append(te_loss_cls)
        history["test_loss_crystal_hist"].append(te_loss_cr)
        history["test_acc_cls_hist"].append(te_acc_cls)
        history["test_acc_crystal_hist"].append(te_acc_cr)

        print(
            f"Epoch {ep:02d} | "
            f"Data train acc={tr_acc_cls*100:6.2f}% | "
            f"Data test acc={te_acc_cls*100:6.2f}% | "
            f"Crystal train acc={tr_acc_cr*100:6.2f}% | "
            f"Crystal test acc={te_acc_cr*100:6.2f}% | "
            f"Crystal CE={te_ce_cr:.4f} | "
            f"Invalid={te_inv_cr:.4f} | "
            f"Margin={te_margin_cr:.4f}"
        )

    acc_cls_thermal = []
    acc_crystal_thermal = []
    print("\n=== Thermal mean sweep ===")
    for th_mu in config.thermal_mu_sweep:
        test_ds = Subset(
            _make_dataset(
                config,
                poisson_scale=config.train_poisson_scale,
                thermal_mu=th_mu,
                class_map=shared_class_map,
                class_to_burgers=shared_class_to_burgers,
            ),
            test_indices,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        _, a_cls = eval_epoch_cls(model_cls, test_loader, device)
        _, a_cr, _, _, _ = eval_epoch_crystal(
            model_crystal,
            test_loader,
            device,
            alpha_invalid=config.alpha_invalid,
            beta_margin=config.beta_margin,
            margin=config.margin,
            label_smoothing=config.label_smoothing,
        )
        acc_cls_thermal.append(a_cls)
        acc_crystal_thermal.append(a_cr)
        print(
            f"Thermal mu={th_mu:7.3f} | "
            f"Data-driven={a_cls*100:6.2f}% | "
            f"Crystallographic constrained={a_cr*100:6.2f}%"
        )

    acc_cls_poisson = []
    acc_crystal_poisson = []
    print("\n=== Poisson sweep ===")
    for ps in config.poisson_sweep:
        test_ds = Subset(
            _make_dataset(
                config,
                poisson_scale=ps,
                thermal_mu=config.train_thermal_mu,
                class_map=shared_class_map,
                class_to_burgers=shared_class_to_burgers,
            ),
            test_indices,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        _, a_cls = eval_epoch_cls(model_cls, test_loader, device)
        _, a_cr, _, _, _ = eval_epoch_crystal(
            model_crystal,
            test_loader,
            device,
            alpha_invalid=config.alpha_invalid,
            beta_margin=config.beta_margin,
            margin=config.margin,
            label_smoothing=config.label_smoothing,
        )
        acc_cls_poisson.append(a_cls)
        acc_crystal_poisson.append(a_cr)
        print(
            f"Poisson scale={ps:6.1f} | "
            f"Data-driven={a_cls*100:6.2f}% | "
            f"Crystallographic constrained={a_cr*100:6.2f}%"
        )

    print("\n=== Uncertainty benchmark at training noise point ===")
    data_uncertainty = collect_predictive_uncertainty(
        model_cls,
        test_loader_same,
        device,
        crystal=False,
        mc_passes=config.uncertainty_mc_passes,
    )
    crystal_uncertainty = collect_predictive_uncertainty(
        model_crystal,
        test_loader_same,
        device,
        crystal=True,
        mc_passes=config.uncertainty_mc_passes,
    )

    benchmark_rows = [
        summarize_uncertainty("Data-driven", data_uncertainty),
        summarize_uncertainty("Crystallographic constrained", crystal_uncertainty),
    ]
    print_benchmark_table(benchmark_rows)

    data_y_true, data_y_pred = collect_class_predictions(
        model_cls,
        test_loader_same,
        device,
        crystal=False,
    )
    crystal_y_true, crystal_y_pred = collect_class_predictions(
        model_crystal,
        test_loader_same,
        device,
        crystal=True,
    )
    model_metrics_rows = make_model_metrics_table(
        data_y_true,
        data_y_pred,
        crystal_y_true,
        crystal_y_pred,
        num_classes=base_ds.num_classes,
    )
    print_model_metrics_table(model_metrics_rows)

    np.savez(
        artifact_dir / "benchmark_results.npz",
        epochs=np.arange(1, config.epochs + 1),
        loss_cls_hist=np.asarray(history["loss_cls_hist"]),
        loss_crystal_hist=np.asarray(history["loss_crystal_hist"]),
        acc_cls_hist=np.asarray(history["acc_cls_hist"]),
        acc_crystal_hist=np.asarray(history["acc_crystal_hist"]),
        test_loss_cls_hist=np.asarray(history["test_loss_cls_hist"]),
        test_loss_crystal_hist=np.asarray(history["test_loss_crystal_hist"]),
        test_acc_cls_hist=np.asarray(history["test_acc_cls_hist"]),
        test_acc_crystal_hist=np.asarray(history["test_acc_crystal_hist"]),
        thermal_mu_sweep=np.asarray(config.thermal_mu_sweep, dtype=float),
        acc_cls_thermal=np.asarray(acc_cls_thermal),
        acc_crystal_thermal=np.asarray(acc_crystal_thermal),
        poisson_sweep=np.asarray(config.poisson_sweep, dtype=float),
        acc_cls_poisson=np.asarray(acc_cls_poisson),
        acc_crystal_poisson=np.asarray(acc_crystal_poisson),
        data_predictive_entropy=data_uncertainty["predictive_entropy"],
        crystal_predictive_entropy=crystal_uncertainty["predictive_entropy"],
        data_mutual_information=data_uncertainty["mutual_information"],
        crystal_mutual_information=crystal_uncertainty["mutual_information"],
        data_confidence=data_uncertainty["confidence"],
        crystal_confidence=crystal_uncertainty["confidence"],
        class_to_burgers=np.asarray(shared_class_to_burgers, dtype=np.float32),
        data_y_true=data_y_true,
        data_y_pred=data_y_pred,
        crystal_y_true=crystal_y_true,
        crystal_y_pred=crystal_y_pred,
    )

    with (artifact_dir / "benchmark_summary.json").open("w", encoding="utf-8") as f:
        json.dump(benchmark_rows, f, indent=2)
    with (artifact_dir / "model_classification_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(model_metrics_rows, f, indent=2)

    torch.save(model_cls.state_dict(), artifact_dir / "data_driven_resnet18.pt")
    torch.save(model_crystal.state_dict(), artifact_dir / "crystallographic_resnet18.pt")

    print(f"\nSaved benchmark artifacts to: {artifact_dir.resolve()}")
    return {
        "artifact_dir": artifact_dir,
        "summary": benchmark_rows,
        "model_metrics": model_metrics_rows,
        "data_model": model_cls,
        "crystallographic_model": model_crystal,
    }
