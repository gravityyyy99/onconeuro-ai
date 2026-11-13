# OncoNeuro AI

### The documentation

*A learning project by Dibyajyoti Pradhan.*

## Abstract

OncoNeuro AI teaches a computer to sort a brain MRI picture into one of four groups: **glioma**, **meningioma**, **pituitary tumor** or **no tumor**. It uses a pretrained image network (ResNet-18) with a new final layer trained on {{n_train}} public scans, and it adds **Grad-CAM** heatmaps that show which parts of the picture drove each answer.

On {{n_test}} scans it never saw while learning, it was right **{{acc}}** of the time. That is far better than the 25% a random guess gives, and still not good enough for any real medical use.

> **This is a learning project, not a medical tool.** It cannot diagnose anyone. Nothing here is medical advice.

This book explains the idea, the data, the model, the heatmaps, the training, the results and the limits. It is written so a person with no background can follow it, and each chapter can be read on its own.

## Contents
