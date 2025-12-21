#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from torch.utils.data import DataLoader
from diffpair_dataset import PairedTransform, PairDiffDataset
from eval_threshold_search import collect_probs_labels, evaluate_thresholds

import os
import tempfile
import mmap

PAYLOAD_MARKER = b"__DML_PAYLOAD_START__\n"



B64_MARKER = b"__DML_PAYLOAD_B64_START__\n"

import base64

DEFAULT_THRESHOLDS = [0.7, 0.075, 0.975]  # [resnet, swin, vit]



def _extract_payload_to_temp(script_path: Path) -> Optional[Path]:
    # Try base64 payload first to avoid Python treating file as a zipapp
    try:
        with open(script_path, 'rb') as f:
            data = f.read()
        idx_b64 = data.find(B64_MARKER)
        if idx_b64 != -1:
            pos = idx_b64 + len(B64_MARKER)
            # Skip optional whitespace
            while pos < len(data) and data[pos:pos+1] in (b' ', b'\t', b'\r', b'\n'):
                pos += 1
            b64_bytes = data[pos:]
            allowed = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r\t "
            b64_only = bytes(ch for ch in b64_bytes if ch in allowed)
            tmp = tempfile.NamedTemporaryFile(prefix='best_dml_', suffix='.pth', delete=False)
            tmp.write(base64.b64decode(b64_only))
            tmp.flush(); tmp.close()
            return Path(tmp.name)
    except Exception:
        pass
    # Fallback: raw binary payload after marker (may conflict with zipapp detection)
    try:
        with open(script_path, 'rb') as f:
            data = f.read()
        idx = data.find(PAYLOAD_MARKER)
        if idx == -1:
            return None
        start = idx + len(PAYLOAD_MARKER)
        tmp = tempfile.NamedTemporaryFile(prefix='best_dml_', suffix='.pth', delete=False)
        tmp.write(data[start:])
        tmp.flush(); tmp.close()
        return Path(tmp.name)
    except Exception:
        return None
import os
from typing import Sequence, Tuple
from PIL import Image
import torch

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _list_image_files(folder: str):
    files = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in IMG_EXTS:
            files.append(path)
    return files


PREFIX_SAMPLE = "sample_"
PREFIX_CONTROL = "control_"


def _pair_by_suffix(paths: Sequence[str]):
    pairs = {}
    for p in paths:
        fname = os.path.basename(p)
        if fname.startswith(PREFIX_SAMPLE):
            key = fname[len(PREFIX_SAMPLE):]
            pairs.setdefault(key, {})['sample'] = p
        elif fname.startswith(PREFIX_CONTROL):
            key = fname[len(PREFIX_CONTROL):]
            pairs.setdefault(key, {})['control'] = p
    out = []
    for key, d in pairs.items():
        if 'sample' in d and 'control' in d:
            out.append((d['sample'], d['control'], key))
    out.sort(key=lambda x: x[2])
    return out


def _build_recursive_pairs(root: str):
    all_pairs = []
    for dirpath, dirnames, filenames in os.walk(root):
        files = [os.path.join(dirpath, f) for f in filenames if os.path.splitext(f)[1].lower() in IMG_EXTS]
        if not files:
            continue
        pairs = _pair_by_suffix(files)
        if pairs:
            # Stable order by directory then suffix
            pairs.sort(key=lambda t: (dirpath, t[2]))
            all_pairs.extend(pairs)
    return all_pairs


class PairDiffUnlabeledFromList(torch.utils.data.Dataset):
    def __init__(self, pairs: Sequence[Tuple[str, str, str]], transform: PairedTransform):
        self.samples = list(pairs)  # (sample_path, control_path, suffix)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sp, cp, suffix = self.samples[idx]
        img_s = Image.open(sp).convert('RGB')
        img_c = Image.open(cp).convert('RGB')
        tsf_s, tsf_c = self.transform(img_s, img_c)
        diff = torch.abs(tsf_s - tsf_c)
        return diff, suffix, sp, cp


def _build_model_from_src(model_src: str, num_classes: int = 2):
    used = None
    model = None
    if isinstance(model_src, str) and model_src.startswith("timm:"):
        name = model_src.split(":", 1)[1]
        try:
            import timm

            model = timm.create_model(name, pretrained=False, num_classes=num_classes)
            used = f"timm:{name}"
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Failed to rebuild model via timm: {e}")
    else:
        # Fallback: torchvision convnext_tiny (matches eval_diff_cam default)
        from torchvision.models import convnext_tiny
        from torch import nn

        model = convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        used = "torchvision:convnext_tiny"

    return model, used


def main():
    ap = argparse.ArgumentParser(description="Infer noise/signal on auto-paired images within a folder using best_dml_all.pth")
    ap.add_argument("path", nargs="?", default=None, help="Target path (folder or one image); if provided, overrides --data_root.")
    ap.add_argument("--data_root", type=str, default=".", help="Target folder (default: current directory). Must contain sample_*.png and control_*.png pairs.")
    ap.add_argument("--ckpt", type=str, default=None, help="Path to best_dml_all.pth. If omitted, searches next to this script, CWD, then outputs_dml_e3_pl1.")
    ap.add_argument("--thresholds", type=float, nargs="*", default=None, help="Optional per-model thresholds (3 values). If omitted, thresholds are tuned per model on --tune_root when available, otherwise 0.5.")
    ap.add_argument("--tune_root", type=str, default="", help="Labeled dataset root with subfolders noise/signal to tune thresholds. Empty by default (uses built-in defaults).")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap on number of pairs (0 = no cap)")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--out_tsv", type=str, default="pred_from_best_dml.tsv")
    ap.add_argument("--out_json", type=str, default="pred_from_best_dml.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve target root from positional path or --data_root
    data_root = args.path if args.path is not None else args.data_root
    pr = Path(data_root)
    if pr.is_file():
        data_root = str(pr.parent)

    # Load aggregated DML checkpoint
    script_dir = Path(__file__).resolve().parent
    candidate_paths = []
    if args.ckpt:
        candidate_paths.append(Path(args.ckpt))
    candidate_paths.extend([
        script_dir / "best_dml_all.pth",
        Path.cwd() / "best_dml_all.pth",
        Path("outputs_dml_e3_pl1/best_dml_all.pth"),
    ])
    ckpt_path = None
    for c in candidate_paths:
        if c.exists():
            ckpt_path = c
            break
    if ckpt_path is None:
        # Try to extract appended payload from this script
        extracted = _extract_payload_to_temp(Path(__file__).resolve())
        if extracted is not None and extracted.exists():
            ckpt_path = extracted
        else:
            raise FileNotFoundError("Could not locate best_dml_all.pth. Provide --ckpt, place it next to the script, or bundle it after the __DML_PAYLOAD_START__ marker in this file.")
    ckpt: Dict[str, Any] = torch.load(str(ckpt_path), map_location="cpu")

    # Expect these keys from train_mutual_dml.py
    states = [
        ckpt.get("model_state_resnet"),
        ckpt.get("model_state_swin"),
        ckpt.get("model_state_vit"),
    ]
    model_srcs = ckpt.get("model_srcs", [])
    imagenet_norm_flags = ckpt.get("imagenet_norm_flags", [False, False, False])

    if any(s is None for s in states):
        raise RuntimeError("best_dml_all.pth does not contain expected per-model state dicts")
    if not isinstance(model_srcs, list) or len(model_srcs) != 3:
        raise RuntimeError("Missing or invalid model_srcs in checkpoint")
    if not isinstance(imagenet_norm_flags, list) or len(imagenet_norm_flags) != 3:
        imagenet_norm_flags = [False, False, False]

    # Build models
    models = []
    infos: List[Dict[str, Any]] = []
    for i in range(3):
        m, used = _build_model_from_src(model_srcs[i], num_classes=2)
        missing, unexpected = m.load_state_dict(states[i], strict=False)
        models.append(m.to(device))
        infos.append({
            "model_src": used,
            "missing": len(missing),
            "unexpected": len(unexpected),
            "imagenet_norm": bool(imagenet_norm_flags[i]),
        })

    # Dataset and loader (unlabeled folder for inference)
    transform = PairedTransform(image_size=args.image_size, augment=False)
    limit = None if args.limit is None or args.limit <= 0 else int(args.limit)
    pairs = _build_recursive_pairs(data_root)
    if limit is not None:
        pairs = pairs[:limit]
    ds = PairDiffUnlabeledFromList(pairs, transform=transform)
    if len(ds) == 0:
        raise RuntimeError("No valid pairs found. Ensure files are named sample_*.png and control_*.png with matching suffixes.")
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Thresholds: use provided values, otherwise tune per model on labeled dataset if available
    thresholds: List[float]
    if args.thresholds and len(args.thresholds) > 0:
        if len(args.thresholds) != 3:
            raise ValueError("Provide exactly 3 thresholds or none")
        thresholds = [float(t) for t in args.thresholds]
    else:
        tune_root = (args.tune_root or "").strip()
        if tune_root and Path(tune_root).is_dir():
            # Build labeled dataset for threshold tuning
            ds_tune = PairDiffDataset(root=tune_root, classes=("noise", "signal"), transform=transform)
            if len(ds_tune) > 0:
                loader_tune = DataLoader(ds_tune, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
                tuned: List[float] = []
                for i, m in enumerate(models):
                    probs_l, labels_l = collect_probs_labels(m, loader_tune, device, imagenet_norm=bool(imagenet_norm_flags[i]))
                    stats = evaluate_thresholds(probs_l, labels_l, steps=401)
                    tuned.append(float(stats["best"]["threshold"]))
                thresholds = tuned
            else:
                thresholds = DEFAULT_THRESHOLDS
        else:
            thresholds = DEFAULT_THRESHOLDS

    # Prepare normalization tensors if needed
    mean = std = None
    if any(imagenet_norm_flags):
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # Iterate over loader once, classify batch-by-batch, and stream results
    results: List[Dict[str, Any]] = []
    sample_index = 0
    for m in models:
        m.eval()
    with torch.no_grad():
        for batch in loader:
            diffs, suffixes, sample_paths, control_paths = batch
            diffs = diffs.to(device, non_blocking=True)
            batch_probs: List[torch.Tensor] = []
            for model_idx, model in enumerate(models):
                x = diffs
                if imagenet_norm_flags[model_idx]:
                    x = (x - mean) / std  # type: ignore[operator]
                logits = model(x)
                prob_signal = torch.softmax(logits, dim=1)[:, 1]
                batch_probs.append(prob_signal.detach().cpu())

            batch_size = len(suffixes)
            for j in range(batch_size):
                suffix = suffixes[j]
                sp = sample_paths[j]
                cp = control_paths[j]
                per_model_info: List[Dict[str, Any]] = []
                ensemble_flag = 0
                for model_idx, probs_tensor in enumerate(batch_probs):
                    prob_val = float(probs_tensor[j].item())
                    thr = thresholds[model_idx]
                    pred_val = 1 if prob_val >= thr else 0
                    per_model_info.append({
                        "model_src": infos[model_idx]["model_src"],
                        "threshold": thr,
                        "prob_signal": prob_val,
                        "pred": "signal" if pred_val == 1 else "noise",
                    })
                    ensemble_flag = 1 if (ensemble_flag == 1 or pred_val == 1) else 0

                ens_label = "signal" if ensemble_flag == 1 else "noise"
                item = {
                    "index": sample_index + j,
                    "suffix": suffix,
                    "sample_path": sp,
                    "control_path": cp,
                    "per_model": per_model_info,
                    "ensemble_pred": ens_label,
                }
                print(f"{suffix}\t{ens_label}", flush=True)
                results.append(item)

            sample_index += batch_size

    # Save TSV: suffix<TAB>ensemble_pred
    tsv_path = Path(args.out_tsv)
    with open(tsv_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(f"{item['suffix']}\t{item['ensemble_pred']}\n")

    # Optional JSON for detailed inspection
    if args.out_json:
        out = {
            "data_root": data_root,
            "ckpt": str(ckpt_path),
            "models": infos,
            "thresholds": thresholds,
            "count": len(results),
            "results": results,
        }
        out_path = Path(args.out_json)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(json.dumps({"out_tsv": str(tsv_path), "out_json": str(out_path), "count": len(results), "thresholds": thresholds}, ensure_ascii=False))
    else:
        print(json.dumps({"out_tsv": str(tsv_path), "count": len(results), "thresholds": thresholds}, ensure_ascii=False))


if __name__ == "__main__":
    main()
