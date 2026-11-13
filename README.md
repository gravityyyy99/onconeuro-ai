# OncoNeuro AI

Brain-tumor MRI classifier built with **ResNet-18 transfer learning** and **Grad-CAM** heatmaps, with a
multi-page Streamlit site in a Claude-style look (light and dark, works on phones, tablets and desktops).

> **Learning project, not a medical tool. It cannot diagnose anyone.** Every visitor sees a welcome message saying so.

Result on 1,600 never-seen test scans: **87.3%** accuracy (glioma recall only 68%; see the Results chapter).

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```
`onconeuro_resnet18.pt` (45 MB) is included. To retrain: use `onconeuro_train.ipynb` in Colab (T4 GPU), or
`python train_local.py <dataset_dir>` then `python evaluate.py <dataset_dir>` to refresh `evaluation.json` and the example images.

## Speed
PyTorch and Grad-CAM take ~20 s to import, so `worker.py` runs them in a separate background process (started by `model.warmup()`). The website itself stays instant and sends scans to the worker.

## Site map
Analyze a scan · Our journey · About me · Documentation (12 chapters in `docs_content/`)

## Files
| File | Purpose |
|---|---|
| `app.py` | Navigation, welcome pop-up, theme switch |
| `ui.py` | Claude-style CSS, responsive rules, sun/moon switch |
| `analyze_page.py`, `journey_page.py`, `about_page.py`, `docs_pages.py` | Pages |
| `model.py` | Model loading, preprocessing, prediction + Grad-CAM (PyTorch + pytorch-grad-cam) |
| `worker.py` | Background process that runs `model.py` so the pages never freeze |
| `labels.py` | Light constants (class names) shared by every page |
| `train_local.py`, `evaluate.py`, `onconeuro_train.ipynb` | Training and scoring |
| `assets/samples/` | Four sample test scans (Brain Tumor MRI Dataset) |

## Credits
[Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) ·
[pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) ·
[Grad-CAM paper](https://arxiv.org/abs/1610.02391) ·
[PyTorch transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
