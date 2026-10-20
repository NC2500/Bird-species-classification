import os
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

RELEASE_URL = "https://github.com/NC2500/Bird-species-classification/releases/download/1"

MODELS = {
    "Model B: ResNet50, layer4 fine-tuned + augmentation (best)": {
        "file": "model_b.pth",
        "arch": "resnet",
        "scores": "Test Top-1 72.2% | Test average per class 71.4%",
    },
    "Model C: EfficientNet-B0, last blocks fine-tuned + augmentation": {
        "file": "model_c.pth",
        "arch": "effnet",
        "scores": "Test Top-1 65.3% | Test average per class 64.5%",
    },
    "Model A: ResNet50, new head only, no augmentation (baseline)": {
        "file": "model_a.pth",
        "arch": "resnet",
        "scores": "Validation accuracy 44.0%",
    },
}

st.set_page_config(page_title="Bird Species Classifier", page_icon="🐦", layout="wide")


@st.cache_data
def load_class_names():
    with open("class_names.txt") as f:
        return [line.strip() for line in f if line.strip()]


def pretty(name):
    # "001.Black_footed_Albatross" -> "Black footed Albatross"
    return name.split(".", 1)[-1].replace("_", " ")


def is_real_model(file_name):
    with open(file_name, "rb") as f:
        return f.read(2) == b"PK"


def download(file_name):
    if os.path.exists(file_name):
        if is_real_model(file_name):
            return
        os.remove(file_name)
    url = RELEASE_URL.rstrip("/") + "/" + file_name
    urllib.request.urlretrieve(url, file_name)
    if not is_real_model(file_name):
        os.remove(file_name)
        st.error(f"Could not download {file_name}. Check RELEASE_URL.")
        st.stop()


@st.cache_resource
def load_model(file_name, arch):
    download(file_name)
    n = len(load_class_names())
    if arch == "resnet":
        model = models.resnet50()
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, n))
    else:
        model = models.efficientnet_b0()
        model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1280, n))
    model.load_state_dict(torch.load(file_name, map_location="cpu"))
    model.eval()
    return model


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(model, image):
    # Returns the full probability vector.
    with torch.no_grad():
        return torch.softmax(model(preprocess(image).unsqueeze(0)), dim=1)[0]


def top_k(probs, k=5):
    names = load_class_names()
    top = torch.topk(probs, k)
    return [(pretty(names[int(i)]), float(p) * 100) for p, i in zip(top.values, top.indices)]


def entropy_score(probs):
    # 0 = fully certain, 1 = completely unsure (spread over all 200 classes).
    p = probs.clamp_min(1e-12)
    return float(-(p * p.log()).sum() / np.log(len(p)))


# ---------- Grad-CAM heat map ----------

def target_layer(model, arch):
    return model.layer4[-1] if arch == "resnet" else model.features[-1]


def grad_cam(model, arch, image, class_idx):
    # Shows which image regions pushed the model toward class_idx.
    acts, grads = {}, {}
    layer = target_layer(model, arch)
    h1 = layer.register_forward_hook(lambda m, i, o: acts.update(v=o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.update(v=go[0]))
    try:
        x = preprocess(image).unsqueeze(0).requires_grad_(True)
        out = model(x)
        model.zero_grad()
        out[0, class_idx].backward()
    finally:
        h1.remove()
        h2.remove()
    weights = grads["v"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * acts["v"]).sum(dim=1))[0].detach().numpy()
    cam = cam / (cam.max() + 1e-8)
    return cam


def jet(x):
    # Simple blue -> green -> red colour map, no matplotlib needed.
    r = np.clip(1.5 - abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def overlay(image, cam, alpha=0.45):
    img = image.resize((224, 224))
    cam_big = np.array(Image.fromarray(np.uint8(cam * 255)).resize((224, 224), Image.BILINEAR)) / 255.0
    heat = jet(cam_big)
    mixed = (1 - alpha) * (np.array(img) / 255.0) + alpha * heat
    return Image.fromarray(np.uint8(np.clip(mixed, 0, 1) * 255)), Image.fromarray(np.uint8(heat * 255))


# ---------- Page ----------

st.title("🐦 Bird Species Classifier")
st.write("Three models trained on 200 bird species (CUB-200). Upload a photo to see "
         "predictions, a heat map of where the model looked, and how sure it is.")

with st.sidebar:
    choice = st.selectbox("Choose a model", list(MODELS))
    info = MODELS[choice]
    st.caption(info["scores"])
    k = st.slider("Number of predictions to show", 3, 10, 5)
    alpha = st.slider("Heat map strength", 0.1, 0.9, 0.45)
    st.markdown("**How to read the heat map**  \nRed areas drove the prediction most. "
                "Blue areas mattered least. A good model focuses on the bird, not the background.")

uploaded = st.file_uploader("Upload a bird photo", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload a JPG or PNG photo of a bird to begin.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
names = load_class_names()

with st.spinner("Loading model and predicting..."):
    model = load_model(info["file"], info["arch"])
    probs = predict(model, image)
    results = top_k(probs, k)

top_name, top_pct = results[0]
unsure = entropy_score(probs)
gap = results[0][1] - results[1][1]

tab1, tab2, tab3 = st.tabs(["Prediction", "Heat map", "Compare models"])

with tab1:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.image(image, caption="Your photo", use_container_width=True)
    with c2:
        m1, m2, m3 = st.columns(3)
        m1.metric("Top guess", top_name)
        m2.metric("Confidence", f"{top_pct:.1f}%")
        m3.metric("Lead over 2nd", f"{gap:.1f} pts")

        if top_pct < 40 or unsure > 0.5:
            st.warning("Low confidence. The photo may be blurry, the bird may be small, "
                       "or the species may not be one of the 200 classes.")
        elif gap < 15:
            st.info(f"Close call between **{results[0][0]}** and **{results[1][0]}**.")
        else:
            st.success("Confident prediction.")

        st.caption(f"Uncertainty score: {unsure:.2f} (0 = certain, 1 = random guess)")

        df = pd.DataFrame(results, columns=["Species", "Confidence (%)"]).set_index("Species")
        st.bar_chart(df, horizontal=True)

with tab2:
    target = st.selectbox("Explain which prediction?", [r[0] for r in results])
    idx = [pretty(n) for n in names].index(target)
    with st.spinner("Building heat map..."):
        cam = grad_cam(model, info["arch"], image, idx)
        mixed, heat = overlay(image, cam, alpha)
    c1, c2, c3 = st.columns(3)
    c1.image(image.resize((224, 224)), caption="Original", use_container_width=True)
    c2.image(heat, caption="Grad-CAM heat map", use_container_width=True)
    c3.image(mixed, caption=f"Overlay: why '{target}'", use_container_width=True)
    focus = float((cam > 0.5).mean()) * 100
    st.caption(f"The model's strong attention covers about {focus:.0f}% of the image.")

with tab3:
    if st.button("Run all three models"):
        rows, cams = [], []
        with st.spinner("Running all three models..."):
            for label, m in MODELS.items():
                mdl = load_model(m["file"], m["arch"])
                p = predict(mdl, image)
                (n1, p1), (n2, p2) = top_k(p, 2)
                rows.append({"Model": label.split(":")[0], "Top prediction": n1,
                             "Confidence": f"{p1:.1f}%", "2nd guess": n2,
                             "Uncertainty": f"{entropy_score(p):.2f}"})
                cam = grad_cam(mdl, m["arch"], image, int(p.argmax()))
                cams.append((label.split(":")[0], overlay(image, cam, alpha)[0]))
        st.table(rows)

        tops = {r["Top prediction"] for r in rows}
        if len(tops) == 1:
            st.success("All three models agree.")
        else:
            st.warning(f"The models disagree: {', '.join(tops)}.")

        cols = st.columns(3)
        for col, (lbl, img) in zip(cols, cams):
            col.image(img, caption=f"{lbl} focus", use_container_width=True)
