import os
import glob
import time
import argparse
import cv2
import numpy as np
from PIL import Image
import random
from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler

# ---------------- Optimized Dataset with Caching ----------------
class RunwayDatasetOptimized(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, augment=False, cache_masks=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.augment = augment
        self.mask_cache = {}
        
        # Pre-cache masks (they're small)
        if cache_masks:
            print("🗂️  Caching masks in memory...")
            for idx, mask_path in enumerate(tqdm(mask_paths, desc="Caching masks")):
                raw = cv2.imread(mask_path)
                if raw is None:
                    self.mask_cache[idx] = None
                else:
                    red = raw[:, :, 2]
                    self.mask_cache[idx] = (red > 127).astype(np.uint8)
        
        # Albumentations for GPU-friendly augmentation
        self.aug_tf = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3)
        ]) if augment else None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read image
        img_bgr = cv2.imread(self.image_paths[idx])
        if img_bgr is None:
            raise RuntimeError(f"Image not found: {self.image_paths[idx]}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Get mask from cache or load
        if idx in self.mask_cache:
            mask = self.mask_cache[idx]
            if mask is None:
                mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            else:
                mask = mask.copy()
        else:
            raw = cv2.imread(self.mask_paths[idx])
            if raw is None:
                mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            else:
                red = raw[:, :, 2]
                mask = (red > 127).astype(np.uint8)

        # Padding to multiple of 32 (better for deeper networks)
        h, w = img.shape[:2]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h or pad_w:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                                     borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")

        # Apply augmentations (Albumentations is much faster)
        if self.augment and self.aug_tf:
            augmented = self.aug_tf(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # Normalize and convert to tensor
        if self.transform:
            img = self.transform(image=img)['image']
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        mask = torch.from_numpy(mask).long()
        return {"image": img, "mask": mask, "path": self.image_paths[idx]}


# ---------------- Enhanced UNet with Attention ----------------
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ResNetUNetAttention(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Encoder
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.enc1 = resnet.layer1  # 256
        self.enc2 = resnet.layer2  # 512
        self.enc3 = resnet.layer3  # 1024
        self.enc4 = resnet.layer4  # 2048

        # Center with ASPP-like multi-scale
        self.center = nn.Sequential(
            nn.Conv2d(2048, 1024, 3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

        # Attention gates
        self.att4 = AttentionBlock(F_g=512, F_l=1024, F_int=512)
        self.att3 = AttentionBlock(F_g=256, F_l=512, F_int=256)
        self.att2 = AttentionBlock(F_g=128, F_l=256, F_int=128)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = self._dec_block(512 + 1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = self._dec_block(256 + 512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = self._dec_block(128 + 256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.low_level_proj = nn.Conv2d(256, 64, 1)
        self.dec1 = self._dec_block(64 + 64, 64)

        self.final = nn.Conv2d(64, n_classes, 1)

    def _dec_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        s = self.stem(x)
        e1 = self.enc1(s)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        center = self.center(e4)

        # Decoder with attention
        d4 = self.up4(center)
        if d4.shape[2:] != e3.shape[2:]:
            d4 = F.interpolate(d4, size=e3.shape[2:], mode="bilinear", align_corners=False)
        e3_att = self.att4(d4, e3)
        d4 = torch.cat([d4, e3_att], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        if d3.shape[2:] != e2.shape[2:]:
            d3 = F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        e2_att = self.att3(d3, e2)
        d3 = torch.cat([d3, e2_att], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        if d2.shape[2:] != e1.shape[2:]:
            d2 = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        e1_att = self.att2(d2, e1)
        d2 = torch.cat([d2, e1_att], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        low = self.low_level_proj(e1)
        low = F.interpolate(low, size=d1.shape[2:], mode="bilinear", align_corners=False)
        d1 = torch.cat([d1, low], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out


# ---------------- Enhanced Loss ----------------
class EnhancedLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight
        
    def forward(self, logits, targets):
        # Cross Entropy with label smoothing
        ce = F.cross_entropy(logits, targets, weight=self.weight, label_smoothing=0.1)
        
        # Dice Loss
        probs = torch.softmax(logits, dim=1)[:, 1]
        tgt = (targets == 1).float()
        inter = (probs * tgt).sum()
        union = probs.sum() + tgt.sum()
        dice = (2 * inter + 1e-7) / (union + 1e-7)
        
        # Boundary Loss (helps with edge precision)
        laplacian = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                                 dtype=torch.float32, device=logits.device).view(1, 1, 3, 3)
        tgt_4d = tgt.unsqueeze(1)
        boundaries = F.conv2d(tgt_4d, laplacian, padding=1).abs()
        boundaries = (boundaries > 0).float()
        boundary_loss = F.binary_cross_entropy(probs.unsqueeze(1), tgt_4d, reduction='none')
        boundary_loss = (boundary_loss * (1 + 2 * boundaries)).mean()
        
        return 0.4 * ce + 0.4 * (1 - dice) + 0.2 * boundary_loss


# ---------------- Helper Functions ----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # will set True later if we enable speed mode

def has_runway_pixels(mask_path):
    m = cv2.imread(mask_path)
    if m is None:
        return False
    red = m[:, :, 2]
    return (red > 127).any()

def estimate_class_weights(mask_paths, sample_limit=500):
    fg, bg = 0, 0
    for p in tqdm(mask_paths[:sample_limit], desc="Estimating class weights"):
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
    w_fg = min(0.98, bg / total)
    w_bg = max(0.02, fg / total)
    return torch.tensor([w_bg, w_fg], dtype=torch.float32)

def batch_iou(preds, targets):
    intersection = ((preds == 1) & (targets == 1)).sum()
    union = ((preds == 1) | (targets == 1)).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)

def compute_dice(preds, targets):
    intersection = ((preds == 1) & (targets == 1)).sum()
    total = (preds == 1).sum() + (targets == 1).sum()
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(2 * intersection) / float(total)


# ---------------- Optimized Training ----------------
def train_all(
    model,
    all_images,
    all_masks,
    device,
    epochs=50,
    batch_size=32,  # Increased from 8
    lr=1e-3,
    num_workers=8,  # CRITICAL: Use parallel data loading
    ckpt_path="runway_resnet_checkpoint.pth",
    best_model_path="best_runway_resnet_model.pth"
):
    set_seed(42)
    
    # Stratified split
    has_runway = [has_runway_pixels(mp) for mp in tqdm(all_masks, desc="Checking runways")]
    pos_idxs = [i for i, h in enumerate(has_runway) if h]
    neg_idxs = [i for i, h in enumerate(has_runway) if not h]
    
    print(f"📊 Dataset: {len(pos_idxs)} with runway, {len(neg_idxs)} without")
    
    random.shuffle(pos_idxs)
    random.shuffle(neg_idxs)
    
    pos_split = int(0.85 * len(pos_idxs))
    neg_split = int(0.85 * len(neg_idxs))
    
    train_idxs = pos_idxs[:pos_split] + neg_idxs[:neg_split]
    val_idxs = pos_idxs[pos_split:] + neg_idxs[neg_split:]
    
    random.shuffle(train_idxs)
    random.shuffle(val_idxs)
    
    print(f"✓ Train: {len(train_idxs)}, Val: {len(val_idxs)}")
    
    train_imgs = [all_images[i] for i in train_idxs]
    train_masks = [all_masks[i] for i in train_idxs]
    val_imgs = [all_images[i] for i in val_idxs]
    val_masks = [all_masks[i] for i in val_idxs]

    # Albumentations transforms (GPU-friendly)
    train_tf = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    val_tf = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    train_ds = RunwayDatasetOptimized(train_imgs, train_masks, transform=train_tf, augment=True)
    val_ds = RunwayDatasetOptimized(val_imgs, val_masks, transform=val_tf, augment=False)

    # Smarter sampling (3:1 instead of 5:1)
    train_has_rwy = [has_runway_pixels(mp) for mp in train_masks]
    per_sample_weight = [3.0 if h else 1.0 for h in train_has_rwy]
    sampler = WeightedRandomSampler(per_sample_weight, num_samples=len(train_imgs), replacement=True)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    weights = estimate_class_weights(train_masks, sample_limit=500).to(device)
    print(f"⚖️  Weights: {weights.tolist()}")

    criterion = EnhancedLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Mixed Precision Training (CRITICAL for speed)
    scaler = GradScaler()
    
    model.to(device)
    # enable channels_last and cudnn benchmark for speed on fixed-size inputs
    try:
        model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    best_val_iou = 0.0
    start_epoch = 1
    
    if os.path.exists(ckpt_path):
        print(f"📂 Loading checkpoint...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        ckpt_state = checkpoint.get('model_state_dict', checkpoint)
        model_state = model.state_dict()
        matched, skipped = [], []
        # copy only matching params (name & shape)
        for k, v in ckpt_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                matched.append(k)
            else:
                skipped.append(k)
        model.load_state_dict(model_state)
        print(f"✅ Loaded {len(matched)} params from checkpoint, skipped {len(skipped)} incompatible keys.")
        # try loading optimizer & scaler safely
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✅ Optimizer state loaded.")
            except Exception:
                print("⚠️ Optimizer state incompatible, starting optimizer fresh.")
        if 'scaler' in checkpoint:
            try:
                scaler.load_state_dict(checkpoint['scaler'])
                print("✅ AMP scaler loaded.")
            except Exception:
                print("⚠️ AMP scaler not loaded (incompatible).")
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_iou = checkpoint.get('best_val_iou', 0.0)
        print(f"✅ Resumed from epoch {start_epoch}, best IoU: {best_val_iou:.4f}")

    print(f"🚀 Training with batch_size={batch_size}, workers={num_workers}, mixed precision=True")
    
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            imgs = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # Slightly faster
            
            # Mixed precision forward pass
            with autocast():
                logits = model(imgs)
                loss = criterion(logits, masks)
            
            # Mixed precision backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        val_iou_sum = 0.0
        val_dice_sum = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                imgs = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"]
                
                with autocast():
                    logits = model(imgs)
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                masks_np = masks.numpy()
                val_iou_sum += batch_iou(preds, masks_np)
                val_dice_sum += compute_dice(preds, masks_np)
        
        val_iou = val_iou_sum / len(val_loader)
        val_dice = val_dice_sum / len(val_loader)
        elapsed = time.time() - t0

        print(f"📊 Epoch {epoch} | Loss {avg_loss:.4f} | IoU {val_iou:.4f} | "
              f"Dice {val_dice:.4f} | Time {elapsed:.1f}s | LR {optimizer.param_groups[0]['lr']:.6f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), best_model_path)
            print(f"🏆 New best IoU: {best_val_iou:.4f}")

        # Save checkpoint
        if epoch % 5 == 0 or epoch == epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'best_val_iou': best_val_iou
            }, ckpt_path)

    print(f"✅ Training complete. Best IoU: {best_val_iou:.4f}")
    return best_val_iou


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default=r"1920x1080\1920x1080\train")
    parser.add_argument("--mask_dir", default=r"labels\labels\areas\train_labels_1920x1080")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--checkpoint", default="runway_resnet_checkpoint.pth")
    parser.add_argument("--best_model", default="best_runway_resnet_model.pth")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    if torch.cuda.is_available():
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    model = ResNetUNetAttention(n_classes=2)
    print("🔥 Created UNet with ResNet-50 + Attention")

    train_all(
        model,
        all_image_paths,
        all_mask_paths,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        ckpt_path=args.checkpoint,
        best_model_path=args.best_model
    )

if __name__ == "__main__":
    main()