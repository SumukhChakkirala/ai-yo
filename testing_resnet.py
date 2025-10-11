import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from training_resnet import ResNetUNet, RunwayDataset

class RunwayTestDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_bgr = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Pad to multiple of 32 (to match training)
        h, w = img.shape[:2]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h or pad_w:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                                     borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

        if self.transform:
            img_t = self.transform(Image.fromarray(img))
        else:
            img_t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        return {"image": img_t, "path": self.image_paths[idx], "original_shape": (h, w)}

def test_model(model_path, image_path, device):
    print("🚀 Loading model for testing...")
    model = ResNetUNet(n_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    # Try to load with correct key
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

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
            pred_mask = pred_mask[:orig_h, :orig_w]

            orig_img = cv2.imread(path)
            overlay = orig_img.copy()
            overlay[pred_mask == 1] = [0, 0, 255]  # Red in BGR

            alpha = 0.4
            blended = cv2.addWeighted(orig_img, 1.0, overlay, alpha, 0)

            filename = os.path.basename(path)
            out_path = os.path.join("test_results", f"pred_{filename}")
            cv2.imwrite(out_path, blended)
            print(f"✅ Saved prediction overlay to {out_path}")

            # Optionally show
            plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
            plt.title(filename)
            plt.axis("off")
            plt.show()

    print("✅ Testing complete.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "runway_best.pth"  # or your best model path
    image_path = r"C:\Users\User1\Downloads\ai-yo\1920x1080\1920x1080\train\ETOU25_1_6LDImage1.png"
    test_model(model_path, image_path, device)