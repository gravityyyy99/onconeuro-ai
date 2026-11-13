# The data

*A model is only as good as the examples it learned from. This chapter says exactly what those examples were, and where they fall short.*

## The dataset

We used the public **Brain Tumor MRI Dataset** on Kaggle by Masoud Nickparvar. It collects brain MRI pictures in four groups: glioma, meningioma, pituitary tumor and no tumor.

| Part | Scans | Purpose |
|---|---|---|
| **Training** | {{n_train}} | The model learns from these |
| **Testing** | {{n_test}} | Kept hidden. Used only at the end to get an honest score |

The two folders come already separated. The model never sees the Testing folder while it learns, so the final score tells us how it handles scans it has never met.

## What goes in

Every picture is prepared the same way before the model sees it:

1. **Grayscale, 3 channels.** MRI pictures are shades of gray, but ResNet-18 expects color, so the gray values are copied into three channels.
2. **Resize to 224 x 224 pixels.** The network expects this size.
3. **Scale the numbers** with the same averages that ResNet-18 was originally trained with, so the new pictures look familiar to it.

## What can go wrong with the data

Honest projects list the weaknesses of their data:

- **One source.** The pictures come from one public collection, not from many hospitals and machines. A different scanner may produce pictures that look different, and the model may fail on them.
- **Single slices.** Each file is one slice. Real scans have many slices.
- **Possible duplicates.** If slices of the same patient appear in both training and testing, the test score can look better than it really is. We cannot rule this out from the dataset alone.
- **Labels may contain mistakes.** The labels were made by people, and people make mistakes.
- **Missing information.** We do not know the ages, sexes or backgrounds in the data, so we cannot say the model works fairly for everyone.
- **Only four answers.** The model cannot say "this is something else". Show it any other picture and it will still pick one of the four groups.

## Privacy

We use public pictures only. This app does not store the pictures you upload. They are processed in memory to make the prediction.
