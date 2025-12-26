import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from diffpair_dataset import PairDiffDataset, PairedTransform
from eval_diff_cam import build_model_from_ckpt


@torch.no_grad()
def collect_probs_labels(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    imagenet_norm: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    probs_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if imagenet_norm:
            x = (x - mean) / std
        logits = model(x)
        prob_signal = F.softmax(logits, dim=1)[:, 1]
        probs_list.append(prob_signal.detach().cpu())
        labels_list.append(y.detach().cpu())

    probs = torch.cat(probs_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return probs, labels


def evaluate_thresholds(probs: torch.Tensor, labels: torch.Tensor, steps: int = 1001) -> Dict[str, Any]:
    best = {
        "threshold": 0.5,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "confusion": [0, 0, 0, 0],  # tn, fp, fn, tp
    }
    thresholds = torch.linspace(0.0, 1.0, steps=steps)
    y = labels.long()
    n = float(y.numel())
    for t in thresholds.tolist():
        pred = (probs >= t).long()
        tp = int(((pred == 1) & (y == 1)).sum().item())
        tn = int(((pred == 0) & (y == 0)).sum().item())
        fp = int(((pred == 1) & (y == 0)).sum().item())
        fn = int(((pred == 0) & (y == 1)).sum().item())
        acc = (tp + tn) / max(n, 1.0)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        if acc > best["accuracy"]:
            best = {
                "threshold": t,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "confusion": [tn, fp, fn, tp],
            }
    # Baseline at 0.5
    t = 0.5
    pred = (probs >= t).long()
    tp = int(((pred == 1) & (y == 1)).sum().item())
    tn = int(((pred == 0) & (y == 0)).sum().item())
    fp = int(((pred == 1) & (y == 0)).sum().item())
    fn = int(((pred == 0) & (y == 1)).sum().item())
    acc = (tp + tn) / max(n, 1.0)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {"best": best, "baseline@0.5": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion": [tn, fp, fn, tp]}}


def main():
    ap = argparse.ArgumentParser(description="Search threshold to maximize accuracy on a dataset")
    ap.add_argument("--data_root", type=str, default="test_split")
    ap.add_argument("--ckpt", type=str, default="outputs_convnextv2/best_model.pth")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=1001, help="Number of thresholds in [0,1] to scan")
    ap.add_argument("--out", type=str, default="threshold_search.json")
    ap.add_argument("--imagenet_norm", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    transform = PairedTransform(image_size=args.image_size, augment=False)
    ds = PairDiffDataset(root=args.data_root, classes=("noise", "signal"), transform=transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model
    model, model_src, _ = build_model_from_ckpt(args.ckpt, num_classes=2)
    model.to(device)

    # Collect probabilities
    probs, labels = collect_probs_labels(model, loader, device, imagenet_norm=args.imagenet_norm)

    # Search thresholds
    res = evaluate_thresholds(probs, labels, steps=args.steps)
    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": model_src, "total": int(labels.numel()), **res}, f, ensure_ascii=False, indent=2)
    print(json.dumps({"out": str(out_path), "total": int(labels.numel()), **res}, ensure_ascii=False))


if __name__ == "__main__":
    main()

