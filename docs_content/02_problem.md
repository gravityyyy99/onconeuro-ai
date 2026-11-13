# The problem

*Before building anything, we asked what we were really trying to do, and what we were not.*

## What are these groups?

In very simple terms:

| Group | Where it starts | Plain description |
|---|---|---|
| **Glioma** | Glial cells, the brain's support cells | Grows inside the brain tissue itself; can look very different from case to case |
| **Meningioma** | The meninges, the membranes around the brain | Often sits at the edge of the brain, next to the skull |
| **Pituitary tumor** | The pituitary gland at the base of the brain | A small gland behind the nose; tumors here are in a very specific place |
| **No tumor** | Not applicable | A scan with none of the above |

This table is a simplified description for learning. It is not medical information, and real cases are far more varied.

## Why use a computer at all?

Reading scans takes years of training and a lot of time. Researchers study whether AI can **assist** doctors, for example by flagging scans for a second look. The word is *assist*. In real hospitals such tools are tested for years and approved by regulators, and a doctor stays responsible.

## What this project tries to do

- Learn how an image-classification model is built, trained and tested.
- See **why** the model answers as it does, using heatmaps.
- Measure honestly how often it is wrong, and where.

## What this project does not try to do

- It does not diagnose anyone.
- It does not find the exact outline of a tumor.
- It does not recommend any treatment or tell anyone whether to see a doctor.
- It is not tested or approved for any medical use.

## The thought process

We wanted a first project that is small enough to finish and honest enough to teach something real. Four groups keep the question simple. A frozen pretrained network keeps training fast. A heatmap keeps the model accountable. A hidden test set keeps the score honest.
