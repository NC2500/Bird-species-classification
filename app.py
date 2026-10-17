import os
import urllib.request

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# Paste the download link of model_b.pth from your GitHub Release here (Step 3).
# It looks like: https://github.com/YOUR_NAME/bird-classifier/releases/download/v1/model_b.pth
MODEL_URL = "https://github.com/NC2500/Bird-species-classification/releases/tag/vs22"
MODEL_PATH = "model_b.pth"

st.set_page_config(page_title="Bird Species Classifier", page_icon="🐦")


@st.cache_resource  # load once, then reuse for every user
def load_model():
    # Class names, one per line, in label order (0 to 199).
    with open("class_names.txt") as f:
        class_names = [line.strip() for line in f if line.strip()]

    # Download the weights the first time the app starts online.
    # On your laptop the file already exists, so this is skipped.
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    # Rebuild Model B exactly as trained: ResNet50 with a new head.
    model = models.resnet50()
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, len(class_names)))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()  # turns dropout off
    return model, class_names


# Same preprocessing as the plain transform used in training.
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

st.title("🐦 Bird Species Classifier")
st.write(
    "ResNet50 fine-tuned on 200 bird species (CUB-200). "
    "Upload a bird photo to see the top 5 predictions."
)

with st.spinner("Loading model..."):
    model, class_names = load_model()

uploaded = st.file_uploader("Upload a bird photo", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Your photo", use_container_width=True)

    with torch.no_grad():
        probs = torch.softmax(model(preprocess(image).unsqueeze(0)), dim=1)[0]
    top = torch.topk(probs, 5)

    st.subheader("Top 5 predictions")
    for rank, (p, i) in enumerate(zip(top.values, top.indices), start=1):
        name, pct = class_names[int(i)], float(p) * 100
        st.write(f"**{rank}. {name}** : {pct:.1f}%")
        st.progress(float(p))
