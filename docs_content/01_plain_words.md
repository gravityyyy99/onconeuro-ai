# In plain words

*Imagine showing a student thousands of brain scans, each with the right answer written on the back, until they start to recognize patterns. That is all this project does, with a computer as the student.*

*Read this chapter first if you have never met machine learning or medical imaging. No background is needed.*

## What is this project?

**OncoNeuro AI** is a program that looks at one brain MRI picture and says which of four groups it most resembles:

1. **Glioma**
2. **Meningioma**
3. **Pituitary tumor**
4. **No tumor**

It also draws a colored **heatmap** over the picture, so you can see where it was looking when it decided.

## Is it a doctor?

No. It is a **learning project**. A real diagnosis needs a trained doctor, the full scan (many slices, not one picture), your history, your symptoms and often other tests. This program has none of that. It only sees one flat picture.

## What is an MRI?

An MRI (magnetic resonance imaging) machine uses a strong magnet and radio waves to take detailed pictures of the inside of the body, with no radiation. A brain MRI is a stack of slices, like slices of bread. The pictures in this project are single slices.

## What is "AI" here?

A **model** is a program that learned from examples instead of being given rules. We showed it thousands of labelled scans. It adjusted tiny numbers inside itself until its answers matched the labels more often. Nobody typed "tumors look like this".

## What is "transfer learning"?

Learning to see from scratch takes millions of pictures and a lot of computing power. So we start from **ResNet-18**, a network that already learned to see edges, textures and shapes by studying over a million everyday photos. We keep all of that and teach it only one new thing: how to turn what it sees into our four groups.

Think of hiring someone who is already a good photographer and teaching them only to recognize four specific kinds of picture, rather than teaching a baby to see.

## What is the heatmap?

A model is a black box: pictures go in, an answer comes out. **Grad-CAM** opens the box a little. It colors the places that pushed the model toward its answer: red and yellow for the strongest push, blue for almost none. If the red sits on the tumor, that is a good sign. If it sits on the edge of the picture, the model is probably relying on something unrelated.

## What should I try?

Open **Analyze a scan** in the menu. Click a sample scan or upload a picture. Look at three things:

- the **heatmap**: where did it look?
- the **answer**: which group, and how sure?
- the **bars**: how did the other groups score?

Then read *Results and limits* to see how often it is wrong. That chapter is the most important one.
