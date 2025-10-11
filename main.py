import os
import glob
import time
import argparse
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

# ---------------- Dataset ----------------
class RunwayDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.augment = augment

        self.aug_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05)
        ]) if augment else None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise RuntimeError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        raw = cv2.imread(mask_path)
        if raw is None:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        else:
            red = raw[:, :, 2]
            mask = (red > 127).astype(np.uint8)

        h, w = img.shape[:2]
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h or pad_w:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                                     borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")

        pil_img = Image.fromarray(img)

        if self.augment and self.aug_tf:
            pil_img = self.aug_tf(pil_img)

        if self.transform:
            img_t = self.transform(pil_img)
        else:
            img_t = transforms.ToTensor()(pil_img)

        mask_t = torch.from_numpy(mask).long()
        return {
            "image": img_t,
            "mask": mask_t
        }

# ---------------- Model (Smaller, efficient UNet) ----------------
class UNet(nn.Module):
    def __init__(self, n_classes=2, base_ch=16):
        super().__init__()
        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.enc1 = block(3, base_ch)
        self.enc2 = block(base_ch, base_ch*2)
        self.enc3 = block(base_ch*2, base_ch*4)
        self.enc4 = block(base_ch*4, base_ch*8)
        self.bottleneck = block(base_ch*8, base_ch*16)

        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, 2)
        self.dec4 = block(base_ch*16, base_ch*8)
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, 2)
        self.dec3 = block(base_ch*8, base_ch*4)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, 2)
        self.dec2 = block(base_ch*4, base_ch*2)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, 2)
        self.dec1 = block(base_ch*2, base_ch)

        self.final = nn.Conv2d(base_ch, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.final(d1)

# ---------------- Combined Loss ----------------
class CombinedLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight
        
    def forward(self, logits, targets):
        # Cross Entropy
        ce = F.cross_entropy(logits, targets, weight=self.weight)
        
        # Soft Dice for foreground
        probs = torch.softmax(logits, dim=1)[:, 1]
        tgt = (targets == 1).float()
        inter = (probs * tgt).sum()
        union = probs.sum() + tgt.sum()
        dice = (2 * inter + 1e-6) / (union + 1e-6)
        
        # Focal component
        ce_all = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_all)
        focal = ((1 - pt) ** 1.5 * ce_all).mean()
        
        return 0.4 * ce + 0.3 * (1 - dice) + 0.3 * focal

# ---------------- Helper Functions ----------------
def has_runway_pixels(mask_path):
    m = cv2.imread(mask_path)
    if m is None:
        return False
    red = m[:, :, 2]
    return (red > 127).any()

def estimate_class_weights(mask_paths, sample_limit=300):
    fg = 0
    bg = 0
    for p in mask_paths[:sample_limit]:
        m = cv2.imread(p)
        if m is None:
            continue
        red = m[:, :, 2]
        binm = (red > 127)
        fg += int(binm.sum())
        bg += int(binm.size - binm.sum())
    if fg == 0:
        return torch.tensor([0.1, 0.9], dtype=torch.float32)
    total = fg + bg
    w_fg = min(0.95, bg / total)
    w_bg = max(0.05, fg / total)
    return torch.tensor([w_bg, w_fg], dtype=torch.float32)

def save_checkpoint(model, processed_images, best_iou, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "processed_images": processed_images,
        "best_val_iou": best_iou
    }, path)

def load_checkpoint(model, path, device):
    if not os.path.exists(path):
        return [], 0.0
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    processed = ckpt.get("processed_images", [])
    best_iou = ckpt.get("best_val_iou", 0.0)
    print(f"✅ Loaded checkpoint: {len(processed)} processed, best IoU={best_iou:.4f}")
    return processed, best_iou

def batch_iou(preds, targets):
    intersection = ((preds == 1) & (targets == 1)).sum()
    union = ((preds == 1) | (targets == 1)).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def compute_dice(preds, targets):
    intersection = ((preds == 1) & (targets == 1)).sum()
    total = (preds == 1).sum() + (targets == 1).sum()
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return (2 * intersection) / total

# ---------------- Training ----------------
def train_chunk(
    model,
    all_images,
    all_masks,
    device,
    chunk_size=1000,
    epochs=50,
    batch_size=4,
    lr=1e-3,
    ckpt_path="runway_checkpoint.pth",
    best_model_path="best_runway_model.pth",
    use_oversampling=True
):
    processed_images, best_val_iou = load_checkpoint(model, ckpt_path, device)

    processed_set = set(processed_images)
    remaining = [(i, p) for i, p in enumerate(all_images) if p not in processed_set]

    if not remaining:
        print("✅ All images processed")
        return

    take = min(chunk_size, len(remaining))
    print(f"🔍 Next chunk: {take} images ({len(remaining)} remaining)")
    sel_indices = [idx for idx, _ in remaining[:take]]
    chunk_img_paths = [all_images[i] for i in sel_indices]
    chunk_mask_paths = [all_masks[i] for i in sel_indices]

    split = int(0.85 * len(chunk_img_paths))
    train_imgs = chunk_img_paths[:split]
    train_masks = chunk_mask_paths[:split]
    val_imgs = chunk_img_paths[split:]
    val_masks = chunk_mask_paths[split:]

    train_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_ds = RunwayDataset(train_imgs, train_masks, transform=train_tf, augment=True)
    val_ds = RunwayDataset(val_imgs, val_masks, transform=val_tf, augment=False)

    # Oversampling
    if use_oversampling:
        train_has_rwy = [has_runway_pixels(mp) for mp in train_masks]
        pos_count = sum(train_has_rwy)
        
        if pos_count > 0 and pos_count < len(train_imgs):
            per_sample_weight = [1.0 if h else 0.25 for h in train_has_rwy]
            sampler = WeightedRandomSampler(per_sample_weight, num_samples=len(train_imgs), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                      num_workers=0, pin_memory=True)
            print(f"✅ Oversampling: {pos_count}/{len(train_imgs)} positive samples")
        else:
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                      num_workers=0, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Class weights
    weights = estimate_class_weights(train_masks, sample_limit=300)
    print(f"⚖️  Class weights [bg, fg]: {weights.tolist()}")
    weights = weights.to(device)

    # Loss, optimizer, scheduler
    criterion = CombinedLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        
        for b_idx, batch in enumerate(train_loader):
            imgs = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        val_iou_sum = 0.0
        val_dice_sum = 0.0
        true_fg = 0
        pred_fg = 0
        total_px = 0
        
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"]
                logits = model(imgs)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                masks_np = masks.numpy()
                
                val_iou_sum += batch_iou(preds, masks_np)
                val_dice_sum += compute_dice(preds, masks_np)
                
                true_fg += (masks_np == 1).sum()
                pred_fg += (preds == 1).sum()
                total_px += preds.size
        
        val_iou = val_iou_sum / len(val_loader)
        val_dice = val_dice_sum / len(val_loader)
        
        fg_true_pct = (true_fg / total_px * 100) if total_px else 0
        fg_pred_pct = (pred_fg / total_px * 100) if total_px else 0
        
        print(f"📊 Epoch {epoch}/{epochs} | Loss {avg_loss:.4f} | IoU {val_iou:.4f} | Dice {val_dice:.4f} | "
              f"FG% true:{fg_true_pct:.2f} pred:{fg_pred_pct:.2f} | LR {optimizer.param_groups[0]['lr']:.6f} | {time.time()-t0:.1f}s")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), best_model_path)
            print(f"🏆 New best IoU: {best_val_iou:.4f}")

        if epoch % 5 == 0:
            save_checkpoint(model, processed_images, best_val_iou, ckpt_path)

    processed_images.extend(chunk_img_paths)
    save_checkpoint(model, processed_images, best_val_iou, ckpt_path)
    print(f"✅ Chunk complete. Total processed: {len(processed_images)}")

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default=r"1920x1080\1920x1080\train")
    parser.add_argument("--mask_dir", default=r"labels\labels\areas\train_labels_1920x1080")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint", default="runway_checkpoint.pth")
    parser.add_argument("--best_model", default="best_runway_model.pth")
    parser.add_argument("--no_oversample", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.reset:
        for f in [args.checkpoint, args.best_model]:
            if os.path.exists(f):
                os.remove(f)
                print(f"♻️  Deleted {f}")

    all_image_paths = sorted(glob.glob(os.path.join(args.img_dir, "*.png")))
    all_mask_paths = [os.path.join(args.mask_dir, os.path.basename(p)) for p in all_image_paths]

    print(f"📁 Total images: {len(all_image_paths)}")

    # Sanity check
    sample = all_mask_paths[:100]
    nonempty = sum(1 for mp in sample if has_runway_pixels(mp))
    print(f"🧪 Sample check: {nonempty}/{len(sample)} masks have runway")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    model = UNet(n_classes=2, base_ch=16)

    train_chunk(
        model,
        all_image_paths,
        all_mask_paths,
        device,
        chunk_size=args.chunk_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        ckpt_path=args.checkpoint,
        best_model_path=args.best_model,
        use_oversampling=not args.no_oversample
    )

    print("✅ Training complete")

if __name__ == "__main__":
    main()