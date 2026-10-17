import os
import urllib.request

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# The folder link of your GitHub Release. All three model files sit inside it.
# Take your old model_b link and delete "model_b.pth" from the end. Example:
RELEASE_URL = "https://github.com/NC2500/Bird-species-classification/releases/download/v1"

# The three trained models. The first one is the default choice.
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

st.set_page_config(page_title="Bird Species Classifier", page_icon="🐦")


@st.cache_data
def load_class_names():
    # Class names, one per line, in label order (0 to 199).
    with open("class_names.txt") as f:
        return [line.strip() for line in f if line.strip()]


def is_real_model(file_name):
    # Real .pth files start with the letters "PK". A web page does not.
    with open(file_name, "rb") as f:
        return f.read(2) == b"PK"


def download(file_name):
    # Download a model file once. On your laptop the files already exist.
    if os.path.exists(file_name):
        if is_real_model(file_name):
            return
        os.remove(file_name)  # a broken download from before
    url = RELEASE_URL.rstrip("/") + "/" + file_name
    urllib.request.urlretrieve(url, file_name)
    if not is_real_model(file_name):
        os.remove(file_name)
        st.error(f"Could not download {file_name}. Check RELEASE_URL and that "
                 f"{file_name} is in the same GitHub Release.")
        st.stop()


@st.cache_resource  # each model loads once, then stays in memory
def load_model(file_name, arch):
    download(file_name)
    n = len(load_class_names())
    # Rebuild the exact architecture used in training, then fill in the weights.
    if arch == "resnet":   # Models A and B
        model = models.resnet50()
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, n))
    else:                  # Model C
        model = models.efficientnet_b0()
        model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1280, n))
    model.load_state_dict(torch.load(file_name, map_location="cpu"))
    model.eval()  # turns dropout off
    return model


# Same preprocessing as the plain transform used in training.
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def top5(model, image):
    # Return the 5 most likely species as (name, percent) pairs.
    names = load_class_names()
    with torch.no_grad():
        probs = torch.softmax(model(preprocess(image).unsqueeze(0)), dim=1)[0]
    top = torch.topk(probs, 5)
    return [(names[int(i)], float(p) * 100) for p, i in zip(top.values, top.indices)]


st.title("🐦 Bird Species Classifier")
st.write("Three models trained on 200 bird species (CUB-200). "
         "Pick a model, upload a bird photo, and see the top 5 predictions.")

choice = st.selectbox("Choose a model", list(MODELS))
info = MODELS[choice]
st.caption(info["scores"])

uploaded = st.file_uploader("Upload a bird photo", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Your photo", use_container_width=True)

    with st.spinner("Loading model and predicting..."):
        results = top5(load_model(info["file"], info["arch"]), image)

    st.subheader("Top 5 predictions")
    for rank, (name, pct) in enumerate(results, start=1):
        st.write(f"**{rank}. {name}** : {pct:.1f}%")
        st.progress(pct / 100)

    if st.checkbox("Compare all three models on this photo"):
        rows = []
        with st.spinner("Running all three models..."):
            for label, m in MODELS.items():
                name, pct = top5(load_model(m["file"], m["arch"]), image)[0]
                rows.append({"Model": label.split(":")[0],
                             "Top prediction": name,
                             "Confidence": f"{pct:.1f}%"})
        st.table(rows)
