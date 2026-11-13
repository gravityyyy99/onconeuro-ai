# The model

*We did not build a vision system from scratch. We borrowed one, froze it, and taught it one new job.*

## Step 1: ResNet-18

**ResNet-18** is a well-known image network with 18 layers, introduced by researchers at Microsoft in 2015. Its key idea is the *residual connection*: each block can pass its input straight through as well as transform it, which makes deep networks much easier to train.

Our copy was already trained on **ImageNet**, a set of over a million everyday photos in 1,000 categories (dogs, cars, mugs and so on). In doing so it learned general-purpose seeing: edges, corners, textures, shapes. Those skills carry over to MRI pictures surprisingly well.

## Step 2: Freeze it

ResNet-18 holds roughly 11 million adjustable numbers (weights). We **freeze** them, which means we lock them so they stay exactly as ImageNet taught them. This keeps training quick and protects the model from forgetting how to see.

## Step 3: A new last layer

ResNet-18 originally ends in a layer that picks one of 1,000 ImageNet categories. We remove it and attach a new one with just **4 outputs**: glioma, meningioma, no tumor and pituitary. That one layer has 512 x 4 + 4 = **2,052** adjustable numbers. These are the only numbers that learn.

## Step 4: Learn from labels

During training, the model looks at a scan, makes a guess, and we measure how wrong it was with **cross-entropy loss** (a number that is small when the model is confident and right, and large when it is confident and wrong). The **Adam optimizer** nudges the 2,052 numbers to make the loss smaller, using a learning rate of 0.001. This repeats over the whole training set many times; each full pass is an **epoch**.

## A speed trick

Because the frozen part never changes, a picture's features come out the same every time. In our local run we computed them once and trained the last layer on the saved features. That gives the same kind of result as the notebook's loop in a fraction of the time on an ordinary computer. The Colab notebook trains the standard way, on a free GPU.

## What the model outputs

For each scan the model produces four scores, converted by **softmax** into probabilities that add up to 100%. The largest is the answer. Remember that these percentages show how strongly the model leans, not how likely it is to be right.

## The thought process

- **Why ResNet-18?** Small, fast, famous and well understood.
- **Why freeze?** Fewer numbers to train means less risk of memorizing the training pictures.
- **Why not fine-tune everything?** It usually gives a better score, and the notebook includes it as an optional step. We kept the simple version first so every part is easy to understand.
