import os
import glob
import time
import argparse
import cv2
import numpy as np
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler
from copy import deepcopy


# ============ PRIORITY: ACCURACY FIRST ============
# Using proven architecture and training strategy
# ==================================================


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RunwayDataset(Dataset):
    """Dataset with verified mask loading"""
    def __init__(self, image_paths, mask_paths, transform=None, augment=False, cache_masks=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.augment = augment
        self.mask_cache = {}

        # Cache masks
        if cache_masks:
            print("🗂️  Caching masks...")
            for idx, mask_path in enumerate(tqdm(mask_paths, desc="Caching")):
                self.mask_cache[idx] = self._load_mask(mask_path)

        # Simple, proven augmentations
        self.aug_tf = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.95, 1.05), translate_percent=0.03, rotate=(-10, 10), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        ]) if augment else None

    def _load_mask(self, mask_path):
        """Load mask - extract red channel with value 128"""
        if not os.path.exists(mask_path):
            return None
        
        mask = cv2.imread(mask_path)
        if mask is None:
            return None
        
        # Extract red channel (BGR format, index 2)
        if len(mask.shape) == 3:
            red = mask[:, :, 2]
        else:
            red = mask
        
        # Convert 128 to 1, 0 stays 0
        binary_mask = (red >= 128).astype(np.uint8)
        return binary_mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img = cv2.imread(self.image_paths[idx])
        if img is None:
            raise RuntimeError(f"Cannot read: {self.image_paths[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load mask
        if idx in self.mask_cache and self.mask_cache[idx] is not None:
            mask = self.mask_cache[idx].copy()
        else:
            mask_data = self._load_mask(self.mask_paths[idx])
            if mask_data is None:
                mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            else:
                mask = mask_data

        # Apply augmentations BEFORE resize
        if self.augment and self.aug_tf:
            augmented = self.aug_tf(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # Normalize and convert
        if self.transform:
            # Ensure mask is int64 for cross_entropy
            mask = mask.astype('int64')
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask'].long()  # <--- Ensure long type here
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return {"image": img, "mask": mask}


class DoubleConv(nn.Module):
    """Double convolution block"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ResNetUNet(nn.Module):
    """Classic UNet with ResNet34 encoder - with flexible size handling"""
    def __init__(self, n_classes=2):
        super().__init__()
        
        # ResNet34 encoder
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool1 = resnet.maxpool
        self.encoder2 = resnet.layer1  # 64
        self.encoder3 = resnet.layer2  # 128
        self.encoder4 = resnet.layer3  # 256
        self.encoder5 = resnet.layer4  # 512
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder
        self.upconv5 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.decoder5 = DoubleConv(512 + 512, 512)
        
        self.upconv4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.decoder4 = DoubleConv(256 + 256, 256)
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder3 = DoubleConv(128 + 128, 128)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder2 = DoubleConv(64 + 64, 64)
        
        self.upconv1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.decoder1 = DoubleConv(64 + 64, 64)
        
        self.final = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        input_size = x.shape[2:]
        
        # Encoder
        enc1 = self.encoder1(x)      # 64, H/2, W/2
        enc1_pool = self.pool1(enc1) # 64, H/4, W/4
        enc2 = self.encoder2(enc1_pool)  # 64, H/4, W/4
        enc3 = self.encoder3(enc2)   # 128, H/8, W/8
        enc4 = self.encoder4(enc3)   # 256, H/16, W/16
        enc5 = self.encoder5(enc4)   # 512, H/32, W/32
        
        # Bottleneck
        bottleneck = self.bottleneck(enc5)  # 1024, H/32, W/32
        
        # Decoder with size matching
        dec5 = self.upconv5(bottleneck)
        dec5 = self._match_size(dec5, enc5)
        dec5 = torch.cat([dec5, enc5], dim=1)
        dec5 = self.decoder5(dec5)
        
        dec4 = self.upconv4(dec5)
        dec4 = self._match_size(dec4, enc4)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = self._match_size(dec3, enc3)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = self._match_size(dec2, enc2)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = self._match_size(dec1, enc1)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        out = self.final(dec1)
        
        # Resize to original input size if needed
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        
        return out
    
    def _match_size(self, x, target):
        """Match spatial dimensions using interpolation"""
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)
        return x


class CombinedLoss(nn.Module):
    """Simple but effective loss combination"""
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight

    def forward(self, logits, targets):
        # Weighted Cross Entropy
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight)
        
        # Dice Loss
        probs = F.softmax(logits, dim=1)[:, 1]
        targets_flat = (targets == 1).float()
        
        intersection = (probs * targets_flat).sum()
        union = probs.sum() + targets_flat.sum()
        dice_loss = 1 - (2 * intersection + 1) / (union + 1)
        
        return 0.5 * ce_loss + 0.5 * dice_loss


def compute_metrics(preds, targets):
    """Compute IoU and Dice"""
    intersection = ((preds == 1) & (targets == 1)).sum()
    union = ((preds == 1) | (targets == 1)).sum()
    
    if union == 0:
        iou = 1.0 if intersection == 0 else 0.0
    else:
        iou = float(intersection) / float(union)
    
    total = (preds == 1).sum() + (targets == 1).sum()
    if total == 0:
        dice = 1.0 if intersection == 0 else 0.0
    else:
        dice = float(2 * intersection) / float(total)
    
    return iou, dice


def has_runway(mask_path):
    """Check if mask has runway pixels"""
    if not os.path.exists(mask_path):
        return False
    mask = cv2.imread(mask_path)
    if mask is None:
        return False
    if len(mask.shape) == 3:
        red = mask[:, :, 2]
    else:
        red = mask
    return (red >= 128).sum() > 100


def train_model(
    model,
    all_images,
    all_masks,
    device,
    epochs=80,
    batch_size=8,
    lr=2e-4,
    num_workers=4,
    checkpoint_path="runway_ckpt.pth",
    best_path="runway_best.pth",
    resume=False
):
    set_seed(42)
    
    # Verify mask loading
    print("\n🔍 Verifying masks...")
    sample_mask = cv2.imread(all_masks[0])
    red_ch = sample_mask[:, :, 2]
    runway_pixels = (red_ch >= 128).sum()
    total_pixels = red_ch.size
    print(f"   Sample mask: {runway_pixels}/{total_pixels} runway pixels ({100*runway_pixels/total_pixels:.2f}%)")
    
    # Split data
    has_rwy = [has_runway(mp) for mp in tqdm(all_masks, desc="Analyzing")]
    pos_idx = [i for i, h in enumerate(has_rwy) if h]
    neg_idx = [i for i, h in enumerate(has_rwy) if not h]
    
    print(f"📊 Positive: {len(pos_idx)}, Negative: {len(neg_idx)}")
    
    # Ensure we have positive samples
    if len(pos_idx) == 0:
        raise RuntimeError("No positive samples found! Check mask loading.")
    
    # Stratified split
    random.shuffle(pos_idx)
    random.shuffle(neg_idx)
    
    split_pos = int(0.85 * len(pos_idx))
    split_neg = int(0.85 * len(neg_idx))
    
    train_idx = pos_idx[:split_pos] + neg_idx[:split_neg]
    val_idx = pos_idx[split_pos:] + neg_idx[split_neg:]
    
    random.shuffle(train_idx)
    random.shuffle(val_idx)
    
    train_imgs = [all_images[i] for i in train_idx]
    train_masks = [all_masks[i] for i in train_idx]
    val_imgs = [all_images[i] for i in val_idx]
    val_masks = [all_masks[i] for i in val_idx]
    
    print(f"✓ Train: {len(train_idx)}, Val: {len(val_idx)}")
    
    # Transforms - NO RESIZE, keep original 1920x1080
    train_tf = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    val_tf = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    train_ds = RunwayDataset(train_imgs, train_masks, transform=train_tf, augment=True)
    val_ds = RunwayDataset(val_imgs, val_masks, transform=val_tf, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    # Calculate class weights properly
    print("\n⚖️  Calculating class weights from training data...")
    total_bg = 0
    total_fg = 0
    for mp in tqdm(train_masks[:500], desc="Sampling"):
        m = cv2.imread(mp)
        if m is not None:
            red = m[:, :, 2] if len(m.shape) == 3 else m
            fg = (red >= 128).sum()
            total_fg += fg
            total_bg += (red.size - fg)
    
    # Inverse frequency weighting
    total = total_bg + total_fg
    weight_bg = total / (2 * total_bg) if total_bg > 0 else 1.0
    weight_fg = total / (2 * total_fg) if total_fg > 0 else 1.0
    weights = torch.tensor([weight_bg, weight_fg], dtype=torch.float32).to(device)
    print(f"   Weights: BG={weight_bg:.4f}, FG={weight_fg:.4f}")
    
    criterion = CombinedLoss(weight=weights)
    
    # Separate learning rates
    encoder_params = []
    for name, param in model.named_parameters():
        if 'encoder' in name:
            encoder_params.append(param)
    encoder_ids = [id(p) for p in encoder_params]
    decoder_params = [p for p in model.parameters() if id(p) not in encoder_ids]
    
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': lr * 0.1, 'weight_decay': 1e-4},
        {'params': decoder_params, 'lr': lr, 'weight_decay': 1e-4}
    ])
    
    # Cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    scaler = GradScaler()
    model.to(device)
    
    best_iou = 0.0
    start_epoch = 1
    patience = 0
    max_patience = 15
    
    if resume and os.path.exists(checkpoint_path):
        print(f"\n📂 Resuming...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_iou = ckpt['best_iou']
        print(f"   Resumed from epoch {start_epoch}, best IoU: {best_iou:.4f}")
    
    print(f"\n🚀 Training Configuration:")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {lr}")
    print(f"   Device: {device}")
    print(f"   Image size: 1920x1080 (FULL RESOLUTION)")
    print()
    
    for epoch in range(start_epoch, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            imgs = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast():
                logits = model(imgs)
                loss = criterion(logits, masks)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{train_loss/(pbar.n+1):.4f}'})
        
        scheduler.step()
        avg_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_ious, val_dices = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                imgs = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].cpu().numpy()
                
                with autocast():
                    logits = model(imgs)
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                for pred, mask in zip(preds, masks):
                    iou, dice = compute_metrics(pred, mask)
                    val_ious.append(iou)
                    val_dices.append(dice)
        
        avg_iou = np.mean(val_ious)
        avg_dice = np.mean(val_dices)
        
        print(f"\n📊 Epoch {epoch}:")
        print(f"   Loss:  {avg_loss:.4f}")
        print(f"   IoU:   {avg_iou:.4f}")
        print(f"   Dice:  {avg_dice:.4f}")
        print(f"   LR:    {optimizer.param_groups[1]['lr']:.6f}\n")
        
        # Save best
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), best_path)
            print(f"🏆 New best IoU: {best_iou:.4f}\n")
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print(f"⏹️  Early stopping at epoch {epoch}")
                break
        
        # Save checkpoint
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_iou': best_iou,
            }, checkpoint_path)
    
    print(f"✅ Training complete! Best IoU: {best_iou:.4f}")
    return best_iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default=r"1920x1080\1920x1080\train")
    parser.add_argument("--mask_dir", default=r"labels\labels\areas\train_labels_1920x1080")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)  # Reduced for full resolution
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    
    checkpoint_path = "runway_ckpt.pth"
    best_path = "runway_best.pth"
    
    if args.reset:
        for f in [checkpoint_path, best_path]:
            if os.path.exists(f):
                os.remove(f)
                print(f"♻️  Deleted {f}")
    
    all_images = sorted(glob.glob(os.path.join(args.img_dir, "*.png")))
    all_masks = [os.path.join(args.mask_dir, os.path.basename(p)) for p in all_images]
    
    print(f"📁 Found {len(all_images)} images")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"💾 GPU: {props.name} ({props.total_memory / 1e9:.1f}GB)")
    
    model = ResNetUNet(n_classes=2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🔥 ResNet34-UNet created ({total_params/1e6:.1f}M parameters)")
    
    train_model(
        model,
        all_images,
        all_masks,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        checkpoint_path=checkpoint_path,
        best_path=best_path,
        resume=args.resume
    )


if __name__ == "__main__":
    main()