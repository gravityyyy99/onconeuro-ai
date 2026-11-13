# The app

*How the website you are using works, and the choices behind it.*

## What it does

1. You upload a picture or choose a sample scan.
2. The picture is prepared exactly like the training pictures.
3. The model gives four probabilities.
4. Grad-CAM paints the heatmap.
5. The page shows the picture, the heatmap, the answer, a plain-words description of the group, and a warning when the model is unsure.

## What it does not do

- It does not save your picture.
- It does not send your picture to any other service.
- It does not give medical advice.

## Built with

| Part | Tool |
|---|---|
| Interface | Streamlit (Python) |
| Model | PyTorch and torchvision (ResNet-18) |
| Heatmaps | pytorch-grad-cam |
| Charts | Altair |
| Training | Google Colab notebook, or `train_local.py` on a laptop |

## Design choices

- **The welcome message.** Every new visitor sees a pop-up explaining that this is a learning project and not a medical tool, and why. The same message is repeated on the main page and in the menu.
- **Light and dark.** The sun and moon switch in the top right changes the whole site, and the site remembers your choice.
- **Every screen size.** The layout is fluid. Wide screens get wider pages, tablets and phones stack things in a single column, buttons are large enough to tap, tables scroll sideways instead of breaking the page, and people who prefer less motion get calmer animations.
- **Plain words first.** Technical terms are explained when they appear.
- **Honest by default.** Warnings appear next to every confident-looking result.

## Using it well

Try the four samples first. Then upload something unusual, such as a photo of a cat or a drawing, to see how the model behaves when it has never seen anything like it. It will still choose one of the four groups. That is a valuable lesson about what these systems cannot do.
