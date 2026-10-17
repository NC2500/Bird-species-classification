import torch
import torch.nn as nn
from torchvision import models, transforms
import gradio as gr

# Class names, one per line, in label order (0 to 199).
# Made by make_class_names.py from train.txt.
with open('class_names.txt') as f:
    CLASS_NAMES = [line.strip() for line in f if line.strip()]

# Rebuild the exact architecture used in training (Model B: ResNet50 with a new head).
# Then fill it with the trained weights. No pretrained download is needed.
model = models.resnet50()
model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, len(CLASS_NAMES)))
model.load_state_dict(torch.load('model_b.pth', map_location='cpu'))
model.eval()   # turns dropout off for prediction

# Same preprocessing as the plain transform used in training.
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict(image):
    # Turn the photo into a batch of one, run the model, convert scores to probabilities.
    x = preprocess(image.convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    # Return the 5 most likely species with their probabilities.
    top = torch.topk(probs, 5)
    return {CLASS_NAMES[int(i)]: float(p) for p, i in zip(top.values, top.indices)}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil', label='Upload a bird photo'),
    outputs=gr.Label(num_top_classes=5, label='Top 5 species'),
    title='Bird Species Classifier (CUB-200)',
    description='ResNet50 fine-tuned on 200 bird species. Upload a photo to see the top 5 predictions.',
)

demo.launch()
