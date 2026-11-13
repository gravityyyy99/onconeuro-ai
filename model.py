"""OncoNeuro AI - shared model code: ResNet-18 classifier + Grad-CAM heatmaps.

Learning project only. Not a medical device and cannot diagnose anyone.

Speed note: PyTorch and Grad-CAM take 10+ seconds to import, so nothing heavy is imported at the top of
this file. `warmup()` loads them in a background thread when the site starts, and the first scan then
only waits for whatever is left.
"""
from __future__ import annotations

import threading
from pathlib import Path

from labels import DEFAULT_CLASSES, LABELS, MEAN, STD  # noqa: F401  (re-exported for older imports)

WEIGHTS_PATH = Path(__file__).parent / "onconeuro_resnet18.pt"

_lock = threading.Lock()
_state: dict = {}          # filled once: {"model", "classes", "transform"}
_warm_started = False


def weights_available() -> bool:
    return WEIGHTS_PATH.exists()


def transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


TRANSFORM = None  # built lazily by scripts that need it (see train_local.py / evaluate.py)


def build_model(num_classes: int = 4):
    import torch.nn as nn
    from torchvision import models
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(path: Path = WEIGHTS_PATH):
    """Returns (model, class_names). Expects the file written by the notebook / train_local.py."""
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    classes = ckpt.get("classes", DEFAULT_CLASSES)
    model = build_model(len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, classes


def get_model():
    """Loads everything once (thread-safe) and returns (model, classes, transform)."""
    with _lock:
        if not _state:
            import pytorch_grad_cam  # noqa: F401  (slow import, done here so the first scan is quick)
            model, classes = load_model()
            _state.update(model=model, classes=classes, transform=transform())
    return _state["model"], _state["classes"], _state["transform"]


_proc = None
_proc_lock = threading.Lock()   # one scan at a time (Grad-CAM edits the shared model)


def _spawn():
    import subprocess
    import sys
    return subprocess.Popen([sys.executable, str(Path(__file__).parent / "worker.py")],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)


def warmup():
    """Start the background worker (a separate process that loads PyTorch + the model)."""
    global _proc, _warm_started
    with _proc_lock:
        if _warm_started or not weights_available():
            return
        _warm_started = True
        try:
            _proc = _spawn()
        except OSError:
            _proc = None          # fall back to loading inside this process on first use


def _call(proc, payload):
    import pickle
    import struct
    payload = pickle.dumps(payload, protocol=4)
    proc.stdin.write(struct.pack("<Q", len(payload)) + payload)
    proc.stdin.flush()
    while True:                    # first reply may be "ready" (the worker finished loading)
        head = proc.stdout.read(8)
        if len(head) < 8:
            raise EOFError("worker stopped")
        kind, value = pickle.loads(proc.stdout.read(struct.unpack("<Q", head)[0]))
        if kind == "ready":
            continue
        if kind == "err":
            raise RuntimeError(value)
        return value


def analyze_bytes(data: bytes):
    """Used by the website. Runs in the background worker so the pages never freeze."""
    global _proc
    with _proc_lock:
        for attempt in (1, 2):
            if _proc is None or _proc.poll() is not None:
                try:
                    _proc = _spawn()
                except OSError:
                    break
            try:
                return _call(_proc, data)
            except (EOFError, BrokenPipeError, OSError):
                _proc = None
        # last resort: do it inside this process (slow the first time)
        import io

        from PIL import Image
        return analyze_image(Image.open(io.BytesIO(data)))


def analyze_image(image):
    """In-process version (used by the worker itself and as a fallback)."""
    model, classes, tf = get_model()
    return analyze(model, classes, image, tf)


def analyze(model, classes, image, tf=None):
    """Predict + Grad-CAM. Returns (probs dict, overlay uint8 RGB array, resized RGB array)."""
    import numpy as np
    import torch
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    tf = tf or transform()
    x = tf(image.convert("RGB"))
    with torch.no_grad():
        probs = torch.softmax(model(x.unsqueeze(0)), dim=1)[0].numpy()

    for p in model.parameters():
        p.requires_grad = True  # Grad-CAM needs gradients
    with GradCAM(model=model, target_layers=[model.layer4[-1]]) as cam:
        heat = cam(input_tensor=x.unsqueeze(0), targets=None)[0]

    img = x.permute(1, 2, 0).numpy() * np.array(STD) + np.array(MEAN)
    img = img.clip(0, 1).astype("float32")
    overlay = show_cam_on_image(img, heat, use_rgb=True)
    return dict(zip(classes, map(float, probs))), overlay, (img * 255).astype("uint8")
