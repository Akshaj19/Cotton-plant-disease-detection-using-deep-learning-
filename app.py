import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cotton Disease Detection",
    page_icon="🌿",
    layout="centered"
)
# ── Class names ────────────────────────────────────────────────────
class_names = ["bacterial_blight", "curl_virus", "fussarium_wilt", "healthy"]
# ── Image transform (no augmentation for inference) ────────────────
img_size = 224
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Input validation: check if image looks like a leaf ─────────────
def is_likely_leaf_image(image: Image.Image) -> bool:
    img_array = np.array(image.convert("RGB"))
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    green_dominant = (g.astype(int) > r.astype(int) + 10) & \
                     (g.astype(int) > b.astype(int) + 10)
    green_ratio = green_dominant.sum() / green_dominant.size
    return green_ratio > 0.10   # at least 10% pixels should be greenish

# ── Model definition ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    resnet  = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

    for p in resnet.parameters():
        p.requires_grad = False
    for p in densenet.parameters():
        p.requires_grad = False

    resnet_feature_size   = resnet.fc.in_features
    densenet_feature_size = densenet.classifier.in_features

    resnet.fc           = nn.Identity()
    densenet.classifier = nn.Identity()

    class FusionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.m1 = resnet
            self.m2 = densenet
            self.fc = nn.Sequential(
                nn.Linear(resnet_feature_size + densenet_feature_size, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, len(class_names))
            )

        def forward(self, x):
            f1    = self.m1(x)
            f2    = self.m2(x)
            fused = torch.cat([f1, f2], dim=1)
            return self.fc(fused)

    net = FusionNet()
    net.load_state_dict(torch.load("cotton_model.pth", map_location="cpu"))
    net.eval()
    return net

model = load_model()

# ── Disease info for display ───────────────────────────────────────
disease_info = {
    "bacterial_blight": {
        "emoji": "🦠",
        "color": "red",
        "desc": "Bacterial infection causing water-soaked lesions and blight on leaves.",
        "action": "Apply copper-based bactericides and remove infected plants."
    },
    "curl_virus": {
        "emoji": "🌀",
        "color": "orange",
        "desc": "Viral disease causing upward leaf curling and stunted growth.",
        "action": "Control whitefly vectors and use virus-resistant varieties."
    },
    "fussarium_wilt": {
        "emoji": "🍂",
        "color": "orange",
        "desc": "Fungal disease causing yellowing, wilting, and vascular discoloration.",
        "action": "Use resistant varieties and apply fungicide soil treatments."
    },
    "healthy": {
        "emoji": "✅",
        "color": "green",
        "desc": "No disease detected. The plant appears healthy.",
        "action": "Continue regular monitoring and good agronomic practices."
    }
}

# ── UI ─────────────────────────────────────────────────────────────
st.title("🌿 Cotton Disease Detection")
st.markdown("Upload a cotton leaf image to detect diseases using a hybrid ResNet50 + DenseNet121 model.")

uploaded_file = st.file_uploader(
    "Upload Cotton Leaf Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ── Input validation ───────────────────────────────────────────
    if not is_likely_leaf_image(image):
        st.error(
            "⚠️ This doesn't appear to be a cotton leaf image. "
            "Please upload a clear photo of a cotton leaf (not a screenshot or document)."
        )
        st.stop()

    # ── Run prediction ─────────────────────────────────────────────
    with st.spinner("Analysing leaf..."):
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs    = model(img_tensor)
            probs      = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)

    pred_label      = class_names[pred.item()]
    confidence_pct  = float(confidence) * 100
    info            = disease_info[pred_label]

    # ── Low confidence warning ─────────────────────────────────────
    if confidence_pct < 75:
        st.warning(
            f"⚠️ Low confidence ({confidence_pct:.1f}%). "
            "The model is unsure — try a clearer, well-lit leaf photo."
        )

    # ── Result card ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"{info['emoji']} Prediction: `{pred_label.replace('_', ' ').title()}`")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Confidence", f"{confidence_pct:.2f}%")
    with col2:
        st.metric("Status", "Healthy" if pred_label == "healthy" else "Disease Detected")

    st.markdown(f"**About:** {info['desc']}")
    st.markdown(f"**Recommended Action:** {info['action']}")

    # ── All class probabilities ────────────────────────────────────
    with st.expander("View all class probabilities"):
        prob_list = probs[0].tolist()
        for name, prob in zip(class_names, prob_list):
            st.progress(prob, text=f"{name.replace('_', ' ').title()}: {prob*100:.2f}%")
