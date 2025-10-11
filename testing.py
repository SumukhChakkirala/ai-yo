import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from main import UNet, RunwayDataset  # import from training file
from training import ResNetUNet

import matplotlib.pyplot as plt

class RunwayTestDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_bgr = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Pad to multiple of 16 (same as training)
        h, w = img.shape[:2]
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h or pad_w:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                                     borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

        pil_img = Image.fromarray(img)
        
        if self.transform:
            img_t = self.transform(pil_img)
        else:
            img_t = torch.from_numpy(img.transpose(2,0,1)).float() / 255.0

        return {"image": img_t, "path": self.image_paths[idx], "original_shape": (h, w)}

# ...existing code...

def test_model(model_path, image_path, device):
    print("🚀 Loading model for testing...")
    model = ResNetUNet(n_classes=2)  # match training base_ch
    
    # Load and inspect checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Try to load with correct key
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # If checkpoint is directly the state dict (not a dictionary with keys)
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    # ... rest of your function

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = RunwayTestDataset([image_path], transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    os.makedirs("test_results", exist_ok=True)

    with torch.no_grad():
        for batch in test_loader:
            image = batch["image"].to(device)
            path = batch["path"][0]
            orig_h, orig_w = batch["original_shape"]

            output = model(image)
            pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0]

            # Crop back to original size (remove padding)
            pred_mask = pred_mask[:orig_h, :orig_w]

            # Save mask overlay
            orig_img = cv2.imread(path)
            overlay = np.zeros_like(orig_img)
            overlay[pred_mask == 1] = [0, 0, 255]  # Red in BGR

            filename = os.path.basename(path)
            
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.title(filename)
            plt.axis("off")
            plt.show()

            print(f"✅ Saved prediction for {filename}")

    print("✅ Testing complete.")



# ...existing code...

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "best_runway_resnet_model.pth"
    image_path = r"C:\Users\User1\Downloads\ai-yo\1920x1080\1920x1080\train\PHNL08R2_3FNLImage6.png"

    test_model(model_path, image_path, device)
