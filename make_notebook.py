"""Generates onconeuro_train.ipynb (run: python make_notebook.py)."""
import json

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(True)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": s.strip().splitlines(True)}

cells = [
md("""# OncoNeuro AI - brain tumor MRI classifier
Learning project, **not a medical tool**. Set **Runtime > Change runtime type > T4 GPU**, then run each cell in order.
Dataset: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (glioma, meningioma, pituitary, no tumor)."""),
md("## Step 1: Get the dataset"),
code("""!pip install -q kagglehub grad-cam
import kagglehub

path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
print(path)  # contains the Training and Testing folders"""),
md("## Step 2: Transfer learning with ResNet-18\nFreeze everything ImageNet taught it; train only the last layer."),
code("""import torch, torch.nn as nn
from torchvision import datasets, models, transforms

tf = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
train = datasets.ImageFolder(f"{path}/Training", tf)
test = datasets.ImageFolder(f"{path}/Testing", tf)
train_dl = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
test_dl = torch.utils.data.DataLoader(test, batch_size=32)

model = models.resnet18(weights="IMAGENET1K_V1")  # already trained on ImageNet
for p in model.parameters():
    p.requires_grad = False  # keep what it already knows
model.fc = nn.Linear(model.fc.in_features, len(train.classes))  # the new last step

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
opt = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    print(f"epoch {epoch + 1}: loss {loss.item():.3f}")"""),
md("## Step 3: Heatmap - where is the AI looking?"),
code("""import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

for p in model.parameters():
    p.requires_grad = True  # Grad-CAM needs gradients
model.eval()

x, y = test[0]  # try other numbers to see other scans
with GradCAM(model=model, target_layers=[model.layer4[-1]]) as cam:
    heat = cam(input_tensor=x.unsqueeze(0).to(device), targets=None)[0]

img = x.permute(1, 2, 0).numpy() * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
plt.imshow(show_cam_on_image(img.clip(0, 1).astype("float32"), heat, use_rgb=True))
plt.title(f"real: {test.classes[y]}")
plt.axis("off")
plt.show()"""),
md("## Step 4: Test on scans it has never seen"),
code("""correct = total = 0
model.eval()
with torch.no_grad():
    for x, y in test_dl:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += len(y)
print(f"accuracy on scans it has never seen: {correct / total:.1%}")"""),
md("## Step 5 (optional): Fine-tune the whole model\nUnfreezes everything and trains a few more epochs at a small learning rate. Usually more accurate."),
code("""opt = torch.optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(3):
    model.train()
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    print(f"fine-tune epoch {epoch + 1}: loss {loss.item():.3f}")

model.eval()
correct = total = 0
with torch.no_grad():
    for x, y in test_dl:
        correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
        total += len(y)
print(f"accuracy after fine-tuning: {correct / total:.1%}")"""),
md("## Step 6: Save the model for the OncoNeuro AI app\nDownload `onconeuro_resnet18.pt` and put it next to `app.py`."),
code("""torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "classes": train.classes}, "onconeuro_resnet18.pt")
from google.colab import files
files.download("onconeuro_resnet18.pt")"""),
]

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
with open("onconeuro_train.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote onconeuro_train.ipynb")
