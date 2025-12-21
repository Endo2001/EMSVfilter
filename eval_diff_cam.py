import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from diffpair_dataset import PairDiffDataset, PairedTransform


def build_model_from_ckpt(ckpt_path: str, num_classes: int = 2):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta_src = ckpt.get("model_src", "torchvision:convnext_tiny")
    model = None
    used = None
    if isinstance(meta_src, str) and meta_src.startswith("timm:"):
        name = meta_src.split(":", 1)[1]
        try:
            import timm

            model = timm.create_model(name, pretrained=False, num_classes=num_classes)
            used = f"timm:{name}"
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Failed to rebuild model via timm: {e}")
    else:
        # torchvision fallback
        from torchvision.models import convnext_tiny

        model = convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        used = "torchvision:convnext_tiny"

    model.load_state_dict(ckpt["model_state"], strict=False)
    return model, used, ckpt


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += int(y.size(0))
    acc = correct / max(total, 1)
    return {"total": total, "correct": correct, "acc": acc}


class GradCAM:
    def __init__(self, model: nn.Module, target_module: nn.Module, device: torch.device):
        self.model = model
        self.target_module = target_module
        self.device = device
        self.activations = None
        self.gradients = None
        self._h1 = target_module.register_forward_hook(self._forward_hook)
        self._h2 = target_module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        # grad_out is a tuple; take gradient wrt output activation
        self.gradients = grad_out[0].detach()

    def remove(self):
        self._h1.remove()
        self._h2.remove()

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[torch.Tensor, int]:
        """Returns CAM (H,W) tensor in [0,1] and the used class index."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)
        A = self.activations  # [B,C,h,w]
        G = self.gradients    # [B,C,h,w]
        weights = G.mean(dim=(2, 3), keepdim=True)
        cam = (A * weights).sum(dim=1)  # [B,h,w]
        cam = torch.relu(cam)
        # Normalize to [0,1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.squeeze(0).detach(), class_idx


def pick_target_module(model: nn.Module) -> nn.Module:
    # Prefer attribute-based selection
    for attr in ["stages", "features"]:
        if hasattr(model, attr):
            m = getattr(model, attr)
            # Try to get the last child module
            if isinstance(m, (nn.Sequential, list, tuple)) and len(m) > 0:
                return m[-1]
            # Otherwise, return it as-is (hook should still work)
            return m
    # Fallback: find the last Conv2d in the model
    last_conv = None
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            last_conv = mod
    if last_conv is None:
        raise RuntimeError("Could not find a target module for Grad-CAM")
    return last_conv


def save_cam_overlay(diff_tensor: torch.Tensor, cam: torch.Tensor, out_path: Path):
    import numpy as np
    from PIL import Image
    import matplotlib
    import matplotlib.cm as cm

    diff = diff_tensor.detach().cpu().clamp(0, 1).numpy()  # [3,H,W]
    diff = np.transpose(diff, (1, 2, 0))  # [H,W,3]
    H, W, _ = diff.shape

    cam_np = cam.detach().cpu().numpy()
    # Resize CAM to match image
    cam_img = Image.fromarray((cam_np * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    cam_arr = np.array(cam_img).astype(np.float32) / 255.0
    # Apply colormap
    colormap = cm.get_cmap('jet')
    heat = colormap(cam_arr)[..., :3]  # drop alpha
    heat = (heat * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat)

    # Overlay with the diff image (convert diff to uint8)
    base = (diff * 255).astype(np.uint8)
    base_img = Image.fromarray(base)
    overlay = Image.blend(base_img.convert('RGBA'), heat_img.convert('RGBA'), alpha=0.5)
    overlay.convert('RGB').save(out_path)


def main():
    ap = argparse.ArgumentParser(description="Evaluate on test_split and export Grad-CAM overlays")
    ap.add_argument("--data_root", type=str, default="test_split")
    ap.add_argument("--ckpt", type=str, default="outputs_es/best_model.pth")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_cams", type=int, default=50, help="Max number of CAM images to export")
    ap.add_argument("--out_dir", type=str, default="eval_outputs")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    transform = PairedTransform(image_size=args.image_size, augment=False)
    ds = PairDiffDataset(root=args.data_root, classes=("noize", "signal"), transform=transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model
    model, model_src, ckpt = build_model_from_ckpt(args.ckpt, num_classes=2)
    model.to(device)

    # Evaluate
    metrics = evaluate(model, loader, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"model": model_src, **metrics}, f, ensure_ascii=False, indent=2)

    # Grad-CAM on a subset
    target_module = pick_target_module(model)
    cammer = GradCAM(model, target_module, device)
    model.eval()

    saved = 0
    for idx in range(len(ds)):
        if saved >= args.max_cams:
            break
        x, y = ds[idx]
        x = x.unsqueeze(0).to(device)
        cam, cls_idx = cammer(x)
        # Name: use class label and basename suffix
        sample_path, control_path, label = ds.samples[idx]
        base = os.path.basename(sample_path)
        # e.g., sample_XYZ.png -> XYZ.png
        if base.startswith("sample_"):
            suffix = base[len("sample_"):]
        elif base.startswith("control_"):
            suffix = base[len("control_"):]
        else:
            suffix = base
        cls_name = "signal" if label == 1 else "noize"
        out_path = out_dir / f"cam_{cls_name}_{suffix}"
        save_cam_overlay(x.squeeze(0).detach().cpu(), cam, out_path)
        saved += 1

    print(json.dumps({"model": model_src, **metrics, "cams_saved": saved}, ensure_ascii=False))


if __name__ == "__main__":
    main()

