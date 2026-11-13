"""Light constants shared by every page. Importing this does NOT import PyTorch, so pages open fast."""
DEFAULT_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumor",
    "pituitary": "Pituitary tumor",
}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
