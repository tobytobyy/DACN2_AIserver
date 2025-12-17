import json
from pathlib import Path
from typing import List, Dict

import torch
from PIL import Image
from torchvision import transforms
import timm

# Đường dẫn: DACN2_AIserver/artifacts
ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"

# Load config + classes
with open(ARTIFACTS_DIR / "model_config.json", "r") as f:
    cfg = json.load(f)

with open(ARTIFACTS_DIR / "food101_classes.json", "r") as f:
    CLASSES = json.load(f)

# Build model đúng kiến trúc đã train
device = torch.device("cpu")

model = timm.create_model(cfg["arch"], pretrained=False, num_classes=cfg["num_classes"])
state = torch.load(ARTIFACTS_DIR / "best.pt", map_location=device)
model.load_state_dict(state)
model.eval()

# Preprocess giống test_tf
tf = transforms.Compose(
    [
        transforms.Resize(cfg["resize"]),
        transforms.CenterCrop(cfg["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def predict(image: Image.Image, top_k: int = 3) -> List[Dict]:
    x = tf(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)[0]
        scores, idxs = torch.topk(prob, k=top_k)

    return [
        {"label": CLASSES[i], "score": float(s)}
        for i, s in zip(idxs.tolist(), scores.tolist())
    ]
