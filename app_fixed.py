import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import timm

# --------------------------
# CONFIG
# --------------------------
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# MODEL (Matches ViTxResNet504ds.py training model)
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
# LOAD MODEL
# --------------------------
@st.cache_resource
def load_model():
    model = ViTResNet50()

    checkpoint_path = "best_by_acc.pt"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    # Handle checkpoint format (saved with 'state_dict' key)
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# --------------------------
# TRANSFORM
# --------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# --------------------------
# UI
# --------------------------
st.set_page_config(page_title="Deepfake Detector", layout="centered")

st.title("🧠 Deepfake Detection App")
st.write("Powered by ViT + ResNet50 Hybrid Model")
st.write("---")

uploaded_file = st.file_uploader("📷 Upload an Image", type=["jpg", "jpeg", "png"])

# --------------------------
# PREDICTION
# --------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze Image", use_container_width=True):
        with st.spinner("⏳ Analyzing..."):
            try:
                img = transform(image).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    output = model(img)
                    probs = F.softmax(output, dim=1)[0]

                real_conf = probs[0].item()
                fake_conf = probs[1].item()

                # --------------------------
                # RESULT
                # --------------------------
                st.subheader("🎯 Analysis Result")

                if fake_conf > real_conf:
                    st.error(f"🚨 **FAKE IMAGE DETECTED**")
                    result_text = f"This image is likely **FAKE** (Confidence: {fake_conf*100:.2f}%)"
                else:
                    st.success(f"✅ **REAL IMAGE**")
                    result_text = f"This image is likely **REAL** (Confidence: {real_conf*100:.2f}%)"
                
                st.info(result_text)

                # --------------------------
                # CONFIDENCE SCORES
                # --------------------------
                st.subheader("📊 Confidence Scores")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Fake Confidence", f"{fake_conf*100:.2f}%")
                with col2:
                    st.metric("Real Confidence", f"{real_conf*100:.2f}%")

                # Visualization
                import pandas as pd
                chart_data = pd.DataFrame({
                    'Score': [fake_conf, real_conf]
                }, index=['Fake', 'Real'])
                
                st.bar_chart(chart_data)
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
