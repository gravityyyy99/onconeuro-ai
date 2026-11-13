"""Scores the saved model on the held-out Testing folder, writes evaluation.json and example images.
Usage: python evaluate.py <dataset_dir>"""
import glob, json, shutil, sys
import numpy as np, torch
from PIL import Image
from torchvision import datasets
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import model as M

root = sys.argv[1]
torch.set_num_threads(16)
net, classes = M.load_model()
test = datasets.ImageFolder(f"{root}/Testing", M.transform())
train_counts = {c: len(glob.glob(f"{root}/Training/{c}/*")) for c in classes}
dl = torch.utils.data.DataLoader(test, batch_size=64)
preds, confs = [], []
with torch.no_grad():
    for x, _ in dl:
        p = torch.softmax(net(x), 1); preds += p.argmax(1).tolist(); confs += p.max(1).values.tolist()
y = np.array([t for _, t in test.samples]); pr = np.array(preds); conf = np.array(confs)
n = len(classes)
cm = np.zeros((n, n), int)
for a, b in zip(y, pr): cm[a, b] += 1
per = {}
for i, c in enumerate(classes):
    tp = cm[i, i]; prec = tp / max(cm[:, i].sum(), 1); rec = tp / cm[i].sum()
    per[c] = dict(precision=float(prec), recall=float(rec), f1=float(2 * prec * rec / max(prec + rec, 1e-9)),
                  support=int(cm[i].sum()))
ev = dict(classes=classes, accuracy=float((y == pr).mean()), test_size=int(len(y)), train_counts=train_counts,
          confusion=cm.tolist(), per_class=per, mean_conf_correct=float(conf[y == pr].mean()),
          mean_conf_wrong=float(conf[y != pr].mean()))
json.dump(ev, open("evaluation.json", "w"), indent=1)
print("accuracy", ev["accuracy"], per)

# one sample scan per class for the "try a sample" buttons (a correctly classified one)
for i, c in enumerate(classes):
    idx = [k for k in range(len(y)) if y[k] == i and pr[k] == i and conf[k] > .9]
    shutil.copy(test.samples[idx[3] if len(idx) > 3 else idx[0]][0], f"assets/samples/{c}.jpg")

def gallery(idxs, name, title):
    fig, ax = plt.subplots(2, len(idxs), figsize=(3.2 * len(idxs), 7.6), gridspec_kw=dict(hspace=0.22, wspace=0.06))
    for j, k in enumerate(idxs):
        probs, ov, img = M.analyze(net, classes, Image.open(test.samples[k][0]))
        ax[0, j].imshow(img); ax[0, j].set_title(f"real: {M.LABELS[classes[y[k]]]}", fontsize=11, pad=8)
        ax[1, j].imshow(ov); ax[1, j].set_title(f"model: {M.LABELS[classes[pr[k]]]} {conf[k]:.0%}", fontsize=11, pad=8)
        ax[0, j].axis("off"); ax[1, j].axis("off")
    fig.suptitle(title, fontsize=13, y=0.99); fig.subplots_adjust(top=0.9, bottom=0.02, left=0.02, right=0.98); fig.savefig(f"assets/examples/{name}.jpg", dpi=100, facecolor="white", pil_kwargs=dict(quality=82, optimize=True))
    plt.close(fig)

rng = np.random.default_rng(1)
good = [rng.choice([k for k in range(len(y)) if y[k] == i and pr[k] == i]) for i in range(n)]
bad = list(rng.choice(np.where(y != pr)[0], 4, replace=False))
gallery(good, "gradcam_right", "Right answers: the heat usually sits on or near the tumor or brain structure")
gallery(bad, "gradcam_wrong", "Wrong answers: the heat is often somewhere else, which is the warning sign")
