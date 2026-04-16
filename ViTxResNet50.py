# =============================================================================
# train_and_eval_combined.py (UPDATED WITH CELEBDF)
# =============================================================================

import os, random, time, warnings
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision import transforms, models
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, classification_report, roc_curve,
)

import timm
from pathlib import Path
import kagglehub

warnings.filterwarnings('ignore')

# Download CelebDF dataset from Kaggle (with cache check)
def download_celebdf():
    try:
        # Check if already cached
        cache_path = Path(r"C:\Users\Dell\.cache\kagglehub\datasets\pranabr0y\celebdf-v2image-dataset")
        if cache_path.exists():
            print(f"[INFO] CelebDF found in cache: {cache_path}")
            return str(cache_path)
        
        print("[INFO] Downloading CelebDF-v2 dataset from Kaggle...")
        import time
        for attempt in range(3):
            try:
                path = kagglehub.dataset_download("pranabr0y/celebdf-v2image-dataset")
                print(f"[SUCCESS] CelebDF downloaded to: {path}")
                return path
            except Exception as e:
                if "in use by another process" in str(e):
                    print(f"[RETRY] File lock detected, retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    raise
        return None
    except Exception as e:
        print(f"[WARNING] Failed to download CelebDF: {e}")
        return None


# --------------------------
# SETUP
# --------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Create models directory for checkpoints
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMG_SIZE    = 224
BATCH_SIZE  = 128
NUM_EPOCHS  = 10
LR          = 5e-4
NUM_WORKERS = 4





# --------------------------
# TRANSFORMS
# --------------------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3,0.3,0.2,0.05),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# --------------------------
# DATASET (UPDATED)
# --------------------------
class CombinedDeepfakeDataset(Dataset):
    def __init__(self, datasets_info, transform=None):
        self.samples = []
        self.transform = transform
        self.dataset_counts = defaultdict(int)

        for root_path, dataset_name in datasets_info:
            root = Path(root_path)

            print(f"\n[CHECK] {dataset_name}")
            print(f"Path: {root}")
            print(f"Exists: {root.exists()}")
            if root.exists():
                subdirs = [d.name for d in root.iterdir() if d.is_dir()]
                print(f"Subdirectories: {subdirs}")

            if not root.exists():
                print(f"[ERROR] Path not found → {root}")
                continue

            count, skipped = 0, 0

            for img_path in root.rglob("*"):
                # ✅ ensure it's a file
                if not img_path.is_file():
                    continue

                # ✅ allow only image formats
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue

                parent = img_path.parent.name.lower()

                # ✅ BEST: direct folder-based labeling
                if parent == "fake":
                    label = 1
                elif parent == "real":
                    label = 0
                else:
                    # ✅ fallback for other datasets (including CelebDF)
                    path_str = str(img_path).lower()

                    # Check for CelebDF-specific structure
                    if "youtube_v2" in path_str or "deepfake" in path_str or "fake" in path_str or "synthesis" in path_str:
                        label = 1
                    elif "youtube" in path_str or "real" in path_str or "original" in path_str:
                        label = 0
                    else:
                        skipped += 1
                        continue

                self.samples.append((img_path, label))
                count += 1

            self.dataset_counts[dataset_name] = count
            print(f"[DATA] {dataset_name}: {count} | Skipped: {skipped}")

        print(f"\n[INFO] TOTAL SAMPLES: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label
# --------------------------
# DATASET PATHS (UPDATED)
# --------------------------
# ✅ These will be initialized in main() to avoid re-execution in worker processes

# --------------------------
# DATA LOADERS
# --------------------------
def get_data_loaders(datasets_info):
    dataset = CombinedDeepfakeDataset(datasets_info, transform=train_transform)
    total_images = len(dataset)

    val_len = test_len = int(0.025 * total_images)
    train_len = total_images - val_len - test_len

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(SEED),
    )

    # 🔥 Reduce training data to HALF
    half_train_len = len(train_ds) // 4
    train_ds, _ = random_split(
        train_ds,
       [half_train_len, len(train_ds) - half_train_len],
       generator=torch.Generator().manual_seed(SEED),
   )

    # ✅ FIX for your earlier crash
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=6,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )

    print(f"\n[DATA] TOTAL: {total_images:,}")
    print(f"Train (HALF): {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    print("\n[DATASET BREAKDOWN]")
    for k, v in dataset.dataset_counts.items():
        print(f"{k:30}: {v}")

    return train_loader, val_loader, test_loader

# --------------------------
# MODEL (UNCHANGED)
# --------------------------
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim_a, dim_b, proj_dim=512, num_heads=8):
        super().__init__()
        self.proj_a = nn.Linear(dim_a, proj_dim)
        self.proj_b = nn.Linear(dim_b, proj_dim)
        self.attn_ab = nn.MultiheadAttention(proj_dim, num_heads, batch_first=True)
        self.attn_ba = nn.MultiheadAttention(proj_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, a, b):
        pa, pb = self.proj_a(a).unsqueeze(1), self.proj_b(b).unsqueeze(1)
        ab,_ = self.attn_ab(pa,pb,pb)
        ba,_ = self.attn_ba(pb,pa,pa)
        return self.norm(ab.squeeze(1)+ba.squeeze(1))

class ViTResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.resnet = nn.Sequential(*list(models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).children())[:-1])

        self.cross = CrossAttentionBlock(self.vit.num_features, 2048)
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        v = self.vit(x)
        r = self.resnet(x).flatten(1)
        return self.fc(self.cross(v,r))

# --------------------------
# TRAIN / EVAL SAME AS BEFORE
# --------------------------
def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0
    preds, labels_all = [], []

    for imgs, labels in tqdm(loader, desc="Train"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out = model(imgs)
            loss = criterion(out, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds += out.argmax(1).cpu().tolist()
        labels_all += labels.cpu().tolist()

    return total_loss/len(loader), accuracy_score(labels_all, preds)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    preds, labels_all, probs = [], [], []
    total_loss = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out = model(imgs)

        loss = criterion(out, labels)
        total_loss += loss.item()

        p = torch.softmax(out,1)[:,1]
        preds += out.argmax(1).cpu().tolist()
        probs += p.cpu().tolist()
        labels_all += labels.cpu().tolist()

    return total_loss/len(loader), accuracy_score(labels_all,preds), roc_auc_score(labels_all,probs)

# --------------------------
# MAIN
# --------------------------
def main():
    # Initialize datasets inside main() to avoid multiprocessing re-execution
    print(f"[INFO] Model checkpoint directory: {MODEL_DIR.absolute()}")
    print(f"[INFO] Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    
    celebdf_path = download_celebdf()
    
    datasets_info = [
        # ✅ Working datasets
        (r"C:\Users\Dell\.cache\kagglehub\datasets\xhlulu\140k-real-and-fake-faces\versions\2\real_vs_fake\real-vs-fake\train", "140K"),

        (r"C:\Users\Dell\.cache\kagglehub\datasets\ayushmandatta1\deepdetect-2025\versions\1\ddata\train", "DeepDetect-2025"),

        (r"C:\Users\Dell\.cache\kagglehub\datasets\aryansingh16\deepfake-dataset\versions\1\real_vs_fake\real-vs-fake\train", "aryansingh"),

        (r"C:\Users\Dell\.cache\kagglehub\datasets\manjilkarki\deepfake-and-real-images\versions\1\Dataset\Train", "deepfake-and-real-images"),
    ]

    # ✅ Add CelebDF only if successfully downloaded
    if celebdf_path:
        datasets_info.append((celebdf_path, "CelebDF-v2"))
    
    train_loader, val_loader, test_loader = get_data_loaders(datasets_info)

    model = ViTResNet50().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # Model checkpointing
    best_val_acc = 0.0
    best_model_path = None
    training_history = []

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        vl_loss, vl_acc, vl_auc = evaluate(model, val_loader, criterion)

        print(f"Train Acc: {tr_acc:.4f} | Val Acc: {vl_acc:.4f} | AUC: {vl_auc:.4f}")
        
        # Track metrics
        training_history.append({
            'epoch': epoch + 1,
            'train_acc': tr_acc,
            'train_loss': tr_loss,
            'val_acc': vl_acc,
            'val_loss': vl_loss,
            'val_auc': vl_auc
        })

        # Save best model based on validation accuracy
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_model_path = MODEL_DIR / f"best_model_epoch{epoch+1}_acc{vl_acc:.4f}.pth"
            
            # Remove previous best model
            for old_model in MODEL_DIR.glob("best_model_*.pth"):
                old_model.unlink()
            
            # Save new best model
            torch.save(model.state_dict(), best_model_path)
            print(f"[CHECKPOINT] Best model saved: {best_model_path.name}")

    # Load and test best model
    print(f"\n[INFO] Loading best model: {best_model_path.name}")
    best_model = ViTResNet50().to(DEVICE)
    best_model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    
    print("\n[FINAL TEST - BEST MODEL]")
    ts_loss, ts_acc, ts_auc = evaluate(best_model, test_loader, criterion)
    print(f"Test Acc: {ts_acc:.4f} | Test AUC: {ts_auc:.4f}")
    print(f"\n[SUCCESS] Best model checkpoint saved at: {best_model_path}")

if __name__ == "__main__":
    main()