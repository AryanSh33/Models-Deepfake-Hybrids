# =============================================================================
# train_and_eval_combined.py
# Combined Deepfake Training (4 datasets)
# =============================================================================
import os, random, time, warnings
from pathlib import Path
from collections import defaultdict
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
warnings.filterwarnings('ignore')
# --------------------------
# SETUP
# --------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
IMG_SIZE    = 224
BATCH_SIZE  = 128
NUM_EPOCHS  = 10
LR          = 3e-4
NUM_WORKERS = 4
# --------------------------
# TRANSFORMS
# --------------------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
# --------------------------
# DATASET
# --------------------------
class CombinedDeepfakeDataset(Dataset):
    """Combines all 4 deepfake datasets with different folder structures."""
    def __init__(self, datasets_info, transform=None):
        """
        datasets_info: list of tuples (root_path, dataset_name)
        """
        self.samples = []
        self.transform = transform
        self.dataset_counts = defaultdict(int)
        for root_path, dataset_name in datasets_info:
            root = Path(root_path)
            count = 0
            # Collect images recursively
            for img_path in root.rglob("*.*"):
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue
                # Determine label based on folder names
                parts = [p.lower() for p in img_path.parts]
                if "fake" in parts:
                    label = 1
                elif "real" in parts:
                    label = 0
                else:
                    continue  # Skip unknown
                self.samples.append((img_path, label))
                count += 1
            self.dataset_counts[dataset_name] = count
            print(f"[DATA] Dataset {dataset_name}: {count} images")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
# --------------------------
# DATASET PATHS (4 Datasets)
# --------------------------
datasets_info = [
    # Dataset 1: 140K Real and Fake Faces
    (r"C:\Users\Dell\.cache\kagglehub\datasets\xhlulu\140k-real-and-fake-faces\versions\2\real_vs_fake\real-vs-fake\train", "140K"),
    # Dataset 2: DeepDetect-2025
    (r"C:\Users\Dell\.cache\kagglehub\datasets\ayushmandatta1\deepdetect-2025\versions\1\ddata\train", "DeepDetect-2025"),
    # Dataset 3: DeepFake Dataset (aryansingh16)
    (r"C:\Users\Dell\.cache\kagglehub\datasets\aryansingh16\deepfake-dataset\versions\1\real_vs_fake\real-vs-fake\train", "aryansingh"),
    # Dataset 4: DeepFake and Real Images (manjilkarki)
    (r"C:\Users\Dell\.cache\kagglehub\datasets\manjilkarki\deepfake-and-real-images\versions\1\Dataset\Train", "deepfake-and-real-images"),
]
# --------------------------
# DATA LOADERS
# --------------------------
def get_data_loaders():
    dataset = CombinedDeepfakeDataset(datasets_info, transform=train_transform)
    total_images = len(dataset)
    # Split 80% train, 10% val, 10% test
    val_len = test_len = int(0.1 * total_images)
    train_len = total_images - val_len - test_len
    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"\n[DATA] Combined total images: {total_images:,}")
    print(f"[DATA] Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    print("\n[DATA] Images per dataset:")
    for k, v in dataset.dataset_counts.items():
        print(f"  - {k:30}: {v}")
    return train_loader, val_loader, test_loader
# --------------------------
# MODEL
# --------------------------
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim_a, dim_b, proj_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.proj_a  = nn.Linear(dim_a, proj_dim)
        self.proj_b  = nn.Linear(dim_b, proj_dim)
        self.attn_ab = nn.MultiheadAttention(proj_dim, num_heads, batch_first=True, dropout=dropout)
        self.attn_ba = nn.MultiheadAttention(proj_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm_a  = nn.LayerNorm(proj_dim)
        self.norm_b  = nn.LayerNorm(proj_dim)
        self.ffn = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim * 4),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(proj_dim * 4, proj_dim),
        )
        self.norm_out = nn.LayerNorm(proj_dim)
    def forward(self, a, b):
        pa = self.proj_a(a).unsqueeze(1)
        pb = self.proj_b(b).unsqueeze(1)
        ab, _ = self.attn_ab(pa, pb, pb)
        ba, _ = self.attn_ba(pb, pa, pa)
        ra  = self.norm_a(ab.squeeze(1) + pa.squeeze(1))
        rb  = self.norm_b(ba.squeeze(1) + pb.squeeze(1))
        merged = torch.cat([ra, rb], dim=-1)
        out = self.norm_out(self.ffn(merged) + ra + rb)
        return out
class AdaptiveGate(nn.Module):
    def __init__(self, in_dim, num_streams=2):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, num_streams),
            nn.Softmax(dim=-1),
        )
    def forward(self, x):
        return self.gate(x)
class ViTResNet50(nn.Module):
    def __init__(self, num_classes=2, proj_dim=512, freeze_vit_blocks=6):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224',
                                     pretrained=True, num_classes=0,
                                     drop_rate=0.1, drop_path_rate=0.1)
        vit_dim = self.vit.num_features
        for i, blk in enumerate(self.vit.blocks):
            if i < freeze_vit_blocks:
                for p in blk.parameters():
                    p.requires_grad = False
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        res_dim = 2048
        self.cross_attn = CrossAttentionBlock(vit_dim, res_dim, proj_dim=proj_dim)
        self.gate        = AdaptiveGate(in_dim=proj_dim)
        self.vit_proj    = nn.Sequential(nn.Linear(vit_dim, proj_dim), nn.LayerNorm(proj_dim))
        self.res_proj    = nn.Sequential(nn.Linear(res_dim, proj_dim), nn.LayerNorm(proj_dim))
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 64),  nn.GELU(),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        v     = self.vit(x)
        r     = self.resnet(x).flatten(1)
        fused = self.cross_attn(v, r)
        w     = self.gate(fused)
        gated = w[:, 0:1] * self.vit_proj(v) + w[:, 1:2] * self.res_proj(r)
        out   = fused + gated
        return self.classifier(out)
# --------------------------
# UTILITIES
# --------------------------
class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count
class EarlyStopping:
    """Stops training if the monitored metric does not improve for `patience` epochs."""
    def __init__(self, patience=3, min_delta=1e-4, mode='max'):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.counter   = 0
        self.best      = None
        self.stop      = False
    def step(self, metric):
        improved = (
            self.best is None or
            (self.mode == 'max' and metric > self.best + self.min_delta) or
            (self.mode == 'min' and metric < self.best - self.min_delta)
        )
        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return improved
# --------------------------
# TRAINING LOOP
# --------------------------
def train_one_epoch(model, loader, optimizer, criterion, scaler, scheduler):
    """
    Runs one full pass over the training set.
    Returns: (avg_loss, accuracy)
    """
    model.train()
    loss_m = AverageMeter()
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc="  Train", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            logits = model(imgs)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        loss_m.update(loss.item(), imgs.size(0))
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        pbar.set_postfix(loss=f"{loss_m.avg:.4f}")
    acc = accuracy_score(all_labels, all_preds)
    return loss_m.avg, acc
# --------------------------
# VALIDATION / EVAL LOOP
# --------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, desc="  Val"):
    """
    Runs inference on loader and returns full metrics.
    Returns: (avg_loss, accuracy, roc_auc, f1, probs, preds, labels)
    """
    model.eval()
    loss_m = AverageMeter()
    all_probs, all_preds, all_labels = [], [], []
    for imgs, labels in tqdm(loader, desc=desc, leave=False):
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = model(imgs)
            loss   = criterion(logits, labels)
        loss_m.update(loss.item(), imgs.size(0))
        all_probs.extend(F.softmax(logits, 1)[:, 1].cpu().tolist())
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    f1  = f1_score(all_labels, all_preds)
    return loss_m.avg, acc, auc, f1, all_probs, all_preds, all_labels
# --------------------------
# VISUALISATION
# --------------------------
def plot_training_curves(history, save_path="training_curves.png"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    epochs = range(1, len(history['tr_loss']) + 1)
    # Loss
    axes[0].plot(epochs, history['tr_loss'], label='Train', marker='o', markersize=4)
    axes[0].plot(epochs, history['vl_loss'], label='Val',   marker='o', markersize=4)
    axes[0].set_title('Loss', fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    # Accuracy
    axes[1].plot(epochs, history['tr_acc'], label='Train Acc', marker='o', markersize=4)
    axes[1].plot(epochs, history['vl_acc'], label='Val Acc',   marker='o', markersize=4)
    axes[1].set_title('Accuracy', fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    # AUC
    axes[2].plot(epochs, history['vl_auc'], label='Val AUC', color='darkorange',
                 marker='o', markersize=4, linestyle='--')
    axes[2].set_title('Val ROC-AUC', fontweight='bold')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('AUC')
    axes[2].legend(); axes[2].grid(alpha=0.3)
    plt.suptitle('ViT + ResNet50 — Training History', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[PLOT] Saved to {save_path}")
def plot_confusion_and_roc(gts, preds, probs, ts_auc, save_prefix="test"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Confusion matrix
    cm = confusion_matrix(gts, preds)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0], cmap='Blues',
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    axes[0].set_title('Confusion Matrix', fontweight='bold')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
    # ROC curve
    fpr, tpr, _ = roc_curve(gts, probs)
    axes[1].plot(fpr, tpr, lw=2, label=f'AUC = {ts_auc:.4f}')
    axes[1].fill_between(fpr, tpr, alpha=0.08)
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve', fontweight='bold')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.suptitle('ViT + ResNet50 — Test Results', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_results.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[PLOT] Saved to {save_prefix}_results.png")
# --------------------------
# MAIN
# --------------------------
def main():
    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_data_loaders()
    # ── Model ─────────────────────────────────────────────────────────────────
    model = ViTResNet50().to(DEVICE)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Total params:     {total / 1e6:.2f}M")
    print(f"[MODEL] Trainable params: {trainable / 1e6:.2f}M")
    # ── Optimiser & Scheduler ─────────────────────────────────────────────────
    criterion   = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer   = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4,
    )
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler   = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=total_steps,
        pct_start=0.1, anneal_strategy='cos',
    )
    scaler      = torch.cuda.amp.GradScaler()
    early_stop  = EarlyStopping(patience=3, mode='max')
    # ── Training Loop ─────────────────────────────────────────────────────────
    history = defaultdict(list)
    best_auc       = 0.0
    best_acc       = 0.0                # ← NEW: track best accuracy separately
    best_state_auc = None
    best_state_acc = None               # ← NEW: separate weights for best acc
    t0 = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch:02d}/{NUM_EPOCHS}  |  LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"{'='*60}")
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, scheduler
        )
        vl_loss, vl_acc, vl_auc, vl_f1, *_ = evaluate(
            model, val_loader, criterion, desc="  Val  "
        )
        # Record history
        history['tr_loss'].append(tr_loss)
        history['tr_acc'].append(tr_acc)
        history['vl_loss'].append(vl_loss)
        history['vl_acc'].append(vl_acc)
        history['vl_auc'].append(vl_auc)
        # ── Save best by AUC ──────────────────────────────────────────────────
        improved = early_stop.step(vl_auc)
        if improved:
            best_auc       = vl_auc
            best_state_auc = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save({
                'model_name' : 'ViT+ResNet50',
                'state_dict' : best_state_auc,
                'val_auc'    : vl_auc,
                'val_acc'    : vl_acc,
                'val_f1'     : vl_f1,
                'epoch'      : epoch,
                'img_size'   : IMG_SIZE,
                'history'    : dict(history),
            }, "best_by_auc.pt")
            print(f"  [CKPT] New best AUC: {best_auc:.4f} → best_by_auc.pt")
        # ── Save best by Accuracy ─────────────────────────────────────────────
        if vl_acc > best_acc:
            best_acc       = vl_acc
            best_state_acc = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save({
                'model_name' : 'ViT+ResNet50',
                'state_dict' : best_state_acc,
                'val_acc'    : vl_acc,
                'val_auc'    : vl_auc,
                'val_f1'     : vl_f1,
                'epoch'      : epoch,
                'img_size'   : IMG_SIZE,
                'history'    : dict(history),
            }, "best_by_acc.pt")
            print(f"  [CKPT] New best Acc: {best_acc:.4f} → best_by_acc.pt")
        print(f"  tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  |  "
              f"vl_loss={vl_loss:.4f}  vl_acc={vl_acc:.4f}  "
              f"vl_auc={vl_auc:.4f}  vl_f1={vl_f1:.4f}")
        if early_stop.stop:
            print(f"\n[EARLY STOP] No improvement for {early_stop.patience} epochs. Stopping.")
            break
    elapsed = (time.time() - t0) / 60
    print(f"\n[INFO] Training complete in {elapsed:.1f} min")
    print(f"[INFO] Best val AUC: {best_auc:.4f}  |  Best val Acc: {best_acc:.4f}")
    # ── Test Evaluation (best-AUC checkpoint) ─────────────────────────────────
    print("\n[TEST] Loading best-AUC checkpoint...")
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state_auc.items()})
    _, ts_acc, ts_auc, ts_f1, probs, preds, gts = evaluate(
        model, test_loader, criterion, desc="  Test "
    )
    print(f"\n{'='*60}")
    print(f"  TEST RESULTS  (best-AUC checkpoint)")
    print(f"{'='*60}")
    print(f"  Accuracy : {ts_acc:.4f}")
    print(f"  ROC-AUC  : {ts_auc:.4f}")
    print(f"  F1-Score : {ts_f1:.4f}")
    print(f"{'='*60}\n")
    print(classification_report(gts, preds, target_names=['Real', 'Fake']))
    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_training_curves(history, save_path="training_curves.png")
    plot_confusion_and_roc(gts, preds, probs, ts_auc, save_prefix="ViT_ResNet50")
    # ── Final save: update best_by_auc.pt with test metrics ───────────────────
    torch.save({
        'model_name' : 'ViT+ResNet50',
        'state_dict' : model.state_dict(),
        'test_acc'   : ts_acc,
        'test_auc'   : ts_auc,
        'test_f1'    : ts_f1,
        'img_size'   : IMG_SIZE,
        'num_epochs' : NUM_EPOCHS,
        'history'    : dict(history),
    }, "best_by_auc.pt")
    print(f"[SAVE] best_by_auc.pt — saved (with test metrics)")
    print(f"[SAVE] best_by_acc.pt — already saved during training")
# --------------------------
# INFERENCE HELPER
# --------------------------
@torch.no_grad()
def predict(model, image_path, threshold=0.5):
    """Run inference on a single image and display the result."""
    model.eval()
    img = Image.open(image_path).convert('RGB')
    t   = eval_transform(img).unsqueeze(0).to(DEVICE)
    prob_fake = F.softmax(model(t), 1)[0, 1].item()
    label     = 'FAKE' if prob_fake >= threshold else 'REAL'
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(f'{label}  (p_fake={prob_fake:.3f})',
              color='red' if label == 'FAKE' else 'green',
              fontsize=12, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return {'label': label, 'prob_fake': prob_fake}
# --------------------------
# ENTRY POINT
# --------------------------
if __name__ == '__main__':
    main()
