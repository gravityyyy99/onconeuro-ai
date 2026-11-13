"""CPU-friendly local version of the notebook: frozen ResNet-18, so features are computed once
and only the last layer is trained. Usage: python train_local.py <dataset_dir>"""
import sys
import torch, torch.nn as nn
from torchvision import datasets, models
import model as M

root = sys.argv[1]
torch.set_num_threads(16)
train = datasets.ImageFolder(f"{root}/Training", M.transform())
test = datasets.ImageFolder(f"{root}/Testing", M.transform())
print("classes:", train.classes, len(train), len(test), flush=True)

net = models.resnet18(weights="IMAGENET1K_V1").eval()
fc = nn.Linear(net.fc.in_features, len(train.classes))
net.fc = nn.Identity()


def feats(ds):
    dl = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=0)
    X, Y = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(dl):
            X.append(net(x)); Y.append(y)
            if i % 10 == 0: print(f"  batch {i}/{len(dl)}", flush=True)
    return torch.cat(X), torch.cat(Y)

Xtr, Ytr = feats(train); Xte, Yte = feats(test)
opt = torch.optim.Adam(fc.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(30):
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(perm), 32):
        idx = perm[i:i + 32]
        opt.zero_grad(); loss = loss_fn(fc(Xtr[idx]), Ytr[idx]); loss.backward(); opt.step()
    if epoch % 5 == 4: print(f"epoch {epoch + 1}: loss {loss.item():.3f}", flush=True)

acc = (fc(Xte).argmax(1) == Yte).float().mean().item()
print(f"accuracy on scans it has never seen: {acc:.1%}", flush=True)

full = M.build_model(len(train.classes))
sd = models.resnet18(weights="IMAGENET1K_V1").state_dict()
sd["fc.weight"], sd["fc.bias"] = fc.weight.detach(), fc.bias.detach()
full.load_state_dict(sd)
torch.save({"state_dict": full.state_dict(), "classes": train.classes}, M.WEIGHTS_PATH)
print("saved", M.WEIGHTS_PATH)
