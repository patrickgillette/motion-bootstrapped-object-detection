import os
import json
import random
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import functional as F
from PIL import Image


def list_samples(dataset_root: str) -> List[str]:
    boxed_dir = os.path.join(dataset_root, "boxed")
    flat_dir = os.path.join(dataset_root, "flat")
    labels_dir = os.path.join(dataset_root, "labels")

    if not (os.path.isdir(boxed_dir) and os.path.isdir(flat_dir) and os.path.isdir(labels_dir)):
        raise RuntimeError(f"Expected {dataset_root}/boxed, {dataset_root}/flat, {dataset_root}/labels")

    ids: List[str] = []
    for fn in os.listdir(boxed_dir):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        base = os.path.splitext(fn)[0]

        flat_jpg = os.path.join(flat_dir, base + ".jpg")
        flat_png = os.path.join(flat_dir, base + ".png")
        lbl_path = os.path.join(labels_dir, base + ".txt")

        if os.path.exists(lbl_path) and (os.path.exists(flat_jpg) or os.path.exists(flat_png)):
            ids.append(base)

    ids.sort()
    if not ids:
        raise RuntimeError("No usable samples found. Make sure boxed/, flat/, labels/ have matching basenames.")
    return ids


def read_box(lbl_path: str):
    with open(lbl_path, "r", encoding="utf-8") as f:
        line = f.read().strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) != 4:
        raise ValueError(f"Bad label file format: {lbl_path}")
    return list(map(float, parts))

def find_image_by_base(flat_dir: str, base: str) -> str:
    candidates = [
        os.path.join(flat_dir, base + ".jpg"),
        os.path.join(flat_dir, base + ".jpeg"),
        os.path.join(flat_dir, base + ".png"),
        os.path.join(flat_dir, base + ".JPG"),
        os.path.join(flat_dir, base + ".JPEG"),
        os.path.join(flat_dir, base + ".PNG"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    base_lower = base.lower()
    for fn in os.listdir(flat_dir):
        stem, ext = os.path.splitext(fn)
        if stem.lower() == base_lower and ext.lower() in [".jpg", ".jpeg", ".png"]:
            return os.path.join(flat_dir, fn)

    raise FileNotFoundError(f"Could not find image for id={base} in {flat_dir}")

def make_splits(dataset_root: str, seed: int = 1337, val_frac: float = 0.2):
    splits_dir = os.path.join(dataset_root, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    train_json = os.path.join(splits_dir, "train.json")
    val_json = os.path.join(splits_dir, "val.json")

    if os.path.exists(train_json) and os.path.exists(val_json):
        with open(train_json, "r", encoding="utf-8") as f:
            train_ids = json.load(f)
        with open(val_json, "r", encoding="utf-8") as f:
            val_ids = json.load(f)
        return train_ids, val_ids

    ids = list_samples(dataset_root)
    rng = random.Random(seed)
    rng.shuffle(ids)

    n_val = max(1, int(len(ids) * val_frac))
    val_ids = sorted(ids[:n_val])
    train_ids = sorted(ids[n_val:])

    with open(train_json, "w", encoding="utf-8") as f:
        json.dump(train_ids, f, indent=2)
    with open(val_json, "w", encoding="utf-8") as f:
        json.dump(val_ids, f, indent=2)

    return train_ids, val_ids


class MotionBoxDataset(Dataset):
    def __init__(self, dataset_root: str, ids: List[str], train: bool):
        self.dataset_root = dataset_root
        self.ids = ids
        self.train = train

        self.flat_dir = os.path.join(dataset_root, "flat")
        self.labels_dir = os.path.join(dataset_root, "labels")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        base = self.ids[idx]

        img_path = find_image_by_base(self.flat_dir, base)

        lbl_path = os.path.join(self.labels_dir, base + ".txt")

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        box = read_box(lbl_path)

        if box is None:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            x1, y1, x2, y2 = box
            x1 = max(0.0, min(x1, w - 1))
            y1 = max(0.0, min(y1, h - 1))
            x2 = max(0.0, min(x2, w - 1))
            y2 = max(0.0, min(y2, h - 1))
            boxes = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
            labels = torch.tensor([1], dtype=torch.int64)

        target: Dict[str, Any] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "iscrowd": torch.zeros((labels.shape[0],), dtype=torch.int64),
        }
        if boxes.shape[0] > 0:
            area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        else:
            area = torch.zeros((0,), dtype=torch.float32)
        target["area"] = area

        if self.train and random.random() < 0.5:
            img = F.hflip(img)
            if boxes.shape[0] > 0:
                boxes_flipped = boxes.clone()
                boxes_flipped[:, 0] = w - boxes[:, 2]
                boxes_flipped[:, 2] = w - boxes[:, 0]
                target["boxes"] = boxes_flipped

        img_tensor = F.to_tensor(img)
        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes: int = 2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model


def train_one_epoch(model, optimizer, loader, device):
    model.train()
    total = 0.0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
    return total / max(1, len(loader))


@torch.no_grad()
def eval_loss(model, loader, device):
    model.train()
    total = 0.0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        total += float(loss.item())
    return total / max(1, len(loader))


def main():
    dataset_root = "dataset"
    seed = 1337
    val_frac = 0.2

    epochs = 12
    batch_size = 2
    lr = 1e-4
    weight_decay = 1e-4
    num_workers = 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    train_ids, val_ids = make_splits(dataset_root, seed=seed, val_frac=val_frac)
    print(f"samples: train={len(train_ids)} val={len(val_ids)}")

    train_ds = MotionBoxDataset(dataset_root, train_ids, train=True)
    val_ds = MotionBoxDataset(dataset_root, val_ids, train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn)

    model = build_model(num_classes=2).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    os.makedirs("models", exist_ok=True)
    best_path = os.path.join("models", "frcnn_best.pth")

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, optimizer, train_loader, device)
        va = eval_loss(model, val_loader, device)
        print(f"epoch {epoch}/{epochs} train_loss={tr:.4f} val_loss={va:.4f}")

        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), best_path)
            print("  saved:", best_path)

    print("done. best val loss:", best_val)


if __name__ == "__main__":
    main()
