# Seeing inside: Grad-CAM

*A score tells you how often a model is right. A heatmap helps you see whether it is right for the right reasons.*

## The problem with black boxes

Models can reach good scores by cheating. A known kind of failure: a model learns that pictures with a certain marker or edge pattern tend to have a certain label, and uses that instead of the actual condition. Its score looks great, and it is useless in the real world.

## What Grad-CAM does

**Grad-CAM** (Gradient-weighted Class Activation Mapping, Selvaraju et al., 2017) asks: *which parts of the picture most increased the score of the chosen answer?*

In steps:

1. Run the picture through the network and keep the feature maps of the last convolutional block (`layer4` in ResNet-18).
2. Work out how much the chosen answer's score would change if each feature map changed (the gradient).
3. Weight each feature map by that importance and add them up.
4. Keep only the positive influence, stretch it to the picture's size, and paint it on top.

## How to read it

| What you see | What it may mean |
|---|---|
| Red or yellow on a patch inside the brain | The model may be reacting to something real |
| Red on the skull edge, background or a corner | The model is likely using something unrelated. Do not trust that answer |
| Heat spread over the whole head | The model is not focusing on anything specific |

## Important cautions

- The heatmap is **coarse**. It is a 7 x 7 grid stretched over the picture, so it can never outline a tumor precisely.
- It explains **the model**, not the patient. A nice-looking heatmap does not prove a correct diagnosis.
- On the wrong answers shown in *Our journey*, the heat is often nowhere near an obvious tumor. That is exactly what this tool is for: it shows when the model should not be trusted.

## Credit

We use the open-source `pytorch-grad-cam` library by Jacob Gildenblat and the method from the Grad-CAM paper. Links are in the Credits chapter.
