#!/usr/bin/env python
"""Test script to verify model loading works correctly"""

import torch
import torch.nn as nn
from torchvision import transforms, models
import timm
import traceback

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# =========== Model Classes ===========
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

# =========== Load and Test ===========
try:
    print("\n[1] Creating model...")
    model = ViTResNet50()
    model.to(DEVICE)
    print("[✓] Model created successfully")
    
    print("\n[2] Loading checkpoint...")
    checkpoint_path = "best_by_acc.pt"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    print(f"[✓] Checkpoint loaded. Keys: {checkpoint.keys()}")
    
    print("\n[3] Loading state dict...")
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print("[✓] Loaded from 'state_dict' key")
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("[✓] Loaded from 'model_state_dict' key")
    else:
        model.load_state_dict(checkpoint)
        print("[✓] Loaded directly")
    
    model.eval()
    print("[✓] Model set to eval mode")
    
    print("\n[4] Testing inference...")
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"[✓] Inference successful. Output shape: {output.shape}")
    
    print("\n✅ All tests passed! Model is ready for deployment.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    traceback.print_exc()
