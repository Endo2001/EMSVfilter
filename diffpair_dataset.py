import os
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


PREFIX_SAMPLE = "sample_"
PREFIX_CONTROL = "control_"


def _list_image_files(folder: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in exts:
            files.append(path)
    return files


def _pair_by_suffix(paths: Sequence[str]) -> List[Tuple[str, str]]:
    pairs: Dict[str, Dict[str, str]] = {}
    for p in paths:
        fname = os.path.basename(p)
        if fname.startswith(PREFIX_SAMPLE):
            key = fname[len(PREFIX_SAMPLE) :]
            pairs.setdefault(key, {})["sample"] = p
        elif fname.startswith(PREFIX_CONTROL):
            key = fname[len(PREFIX_CONTROL) :]
            pairs.setdefault(key, {})["control"] = p
    out: List[Tuple[str, str]] = []
    for key, d in pairs.items():
        if "sample" in d and "control" in d:
            out.append((d["sample"], d["control"]))
    return out


class PairDiffDataset(Dataset):
    """
    Builds pairs within each class subfolder (signal/noize), matching
    files that share the same suffix after `sample_` or `control_`.
    Returns an absolute difference image (3-channel) and label {0,1}.
    """

    def __init__(
        self,
        root: str,
        classes: Sequence[str] = ("noize", "signal"),
        transform: Optional[Callable] = None,
        diff_mode: str = "abs",  # "abs" or "signed"
        limit_per_class: Optional[int] = None,
    ) -> None:
        self.root = root
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform
        self.diff_mode = diff_mode

        samples: List[Tuple[str, str, int]] = []
        for cls in self.classes:
            folder = os.path.join(root, cls)
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Class folder not found: {folder}")
            files = _list_image_files(folder)
            pairs = _pair_by_suffix(files)
            if limit_per_class is not None:
                pairs = pairs[:limit_per_class]
            label = self.class_to_idx[cls]
            for sample_path, control_path in pairs:
                samples.append((sample_path, control_path, label))

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample_path, control_path, label = self.samples[index]
        img_s = Image.open(sample_path).convert("RGB")
        img_c = Image.open(control_path).convert("RGB")

        if self.transform is not None:
            img_s, img_c = self.transform(img_s, img_c)

        # assume tensors in [0,1]; compute absolute difference
        if not isinstance(img_s, torch.Tensor) or not isinstance(img_c, torch.Tensor):
            raise TypeError("Transform must return tensors for both images")

        if self.diff_mode == "signed":
            diff = img_s - img_c
        else:
            diff = torch.abs(img_s - img_c)
        y = torch.tensor(label, dtype=torch.long)
        return diff, y


class PairedTransform:
    """
    Applies the same deterministic resizing/cropping to both images.
    Minimal augmentation to preserve difference structure.
    """

    def __init__(self, image_size: int = 224, augment: bool = False):
        from torchvision import transforms as T

        self.image_size = image_size
        # Use deterministic resize + center crop
        base = [
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ]
        self.base = T.Compose(base)
        self.augment = augment
        self.hflip = T.RandomHorizontalFlip(p=0.5) if augment else None

    def __call__(self, img_a: Image.Image, img_b: Image.Image):
        # Shared flip decision if augmenting
        if self.augment and self.hflip is not None:
            # Make a single random decision
            do_flip = torch.rand(()) < 0.5
            if do_flip:
                img_a = img_a.transpose(Image.FLIP_LEFT_RIGHT)
                img_b = img_b.transpose(Image.FLIP_LEFT_RIGHT)

        ta = self.base(img_a)
        tb = self.base(img_b)
        return ta, tb


def make_splits(
    dataset: PairDiffDataset,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    """Stratified split without sklearn.

    Groups by label, shuffles each group with the given seed, and splits
    by the provided ratio.
    """
    import random

    label_to_indices: Dict[int, List[int]] = {}
    for i, (_, _, y) in enumerate(dataset.samples):
        label_to_indices.setdefault(int(y), []).append(i)

    rng = random.Random(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for _, idxs in label_to_indices.items():
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_ratio)) if len(idxs) > 1 else 1
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    return train_idx, val_idx
