import json
from pathlib import Path
from typing import List, Dict, Tuple, Any

import timm
import torch
from PIL import Image
from torchvision import transforms

from app.core.settings import settings

# Đường dẫn: DACN2_AIserver/artifacts
ARTIFACTS_DIR = Path(settings.artifacts_dir).resolve()


def load_artifacts() -> Tuple[dict, list[str]]:
    """
    Load config + classes from artifacts folder.
    """
    cfg_path = ARTIFACTS_DIR / "model_config.json"
    classes_path = ARTIFACTS_DIR / "food101_classes.json"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing: {cfg_path}")
    if not classes_path.exists():
        raise FileNotFoundError(f"Missing: {classes_path}")

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    with open(classes_path, "r") as f:
        classes = json.load(f)

    # Sanity check
    if "num_classes" in cfg and len(classes) != int(cfg["num_classes"]):
        raise ValueError(
            f"Classes mismatch: len(classes)={len(classes)} != cfg['num_classes']={cfg['num_classes']}"
        )

    return cfg, classes


def build_preprocess(cfg: dict) -> Any:
    """
    Keep EXACT same preprocess as current repo:
    Resize(cfg["resize"]) -> CenterCrop(cfg["img_size"]) -> ToTensor -> Normalize(ImageNet)
    """
    if "resize" not in cfg or "img_size" not in cfg:
        raise ValueError("model_config.json must include keys: 'resize' and 'img_size'")

    tf = transforms.Compose(
        [
            transforms.Resize(cfg["resize"]),
            transforms.CenterCrop(cfg["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return tf


def build_model(cfg: dict) -> torch.nn.Module:
    """
    Build timm model and load weights.
    """
    if "arch" not in cfg or "num_classes" not in cfg:
        raise ValueError(
            "model_config.json must include keys: 'arch' and 'num_classes'"
        )

    device = torch.device("cpu")
    model = timm.create_model(
        cfg["arch"], pretrained=False, num_classes=cfg["num_classes"]
    )

    weights_path = ARTIFACTS_DIR / "best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing: {weights_path}")

    state = torch.load(weights_path, map_location=device)

    # Nếu checkpoint của bạn có dạng {"state_dict": ...} thì mở comment dưới:
    # if isinstance(state, dict) and "state_dict" in state:
    #     state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()
    return model


def predict_with(
    model: torch.nn.Module,
    preprocess: Any,
    classes: list[str],
    image: Image.Image,
    top_k: int = 3,
) -> List[Dict]:
    """
    Predict top_k from a PIL image using provided model+preprocess+classes.
    """
    if model is None or preprocess is None or classes is None:
        raise RuntimeError("Model/preprocess/classes not initialized")

    top_k = max(1, min(int(top_k), len(classes)))

    x = preprocess(image.convert("RGB")).unsqueeze(0)

    # move input to model device
    device = next(model.parameters()).device
    x = x.to(device)

    with torch.inference_mode():
        out = model(x)
        prob = torch.softmax(out, dim=1)[0]

    # move prob to cpu and detach
    prob = prob.detach().cpu()
    scores, idxs = torch.topk(prob, k=top_k)

    return [
        {"label": classes[i], "score": float(s)}
        for i, s in zip(idxs.tolist(), scores.tolist())
    ]
