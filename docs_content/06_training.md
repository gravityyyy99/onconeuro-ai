# Training and testing

*Training builds the model. Testing tells the truth about it. They must never be mixed up.*

## The recipe

1. Download the dataset.
2. Prepare every picture (grayscale, resize, scale).
3. Load ResNet-18 with ImageNet weights, freeze it, attach the new 4-output layer.
4. Train the last layer with Adam, learning rate 0.001, on the Training folder.
5. Test once on the hidden Testing folder.
6. Draw Grad-CAM heatmaps to inspect behavior.
7. Save the model and use it in this app.

You can run these steps yourself in the included Google Colab notebook (free T4 GPU) or locally with `train_local.py`.

## Words you will meet in the results

- **Accuracy:** out of all test scans, the share it got right.
- **Recall** (also called sensitivity): out of the scans that really belong to a group, the share it found. *"Of all real gliomas, how many did it catch?"*
- **Precision:** out of the scans it put in a group, the share that really belong there. *"When it says glioma, how often is it right?"*
- **Confusion matrix:** a grid showing, for every real group, what the model said. Correct answers sit on the diagonal.

## Why a hidden test set

If we judged the model on the pictures it learned from, it could simply memorize them and look perfect. The Testing folder was never used to adjust anything, so its score is the honest one. We looked at it once and did not tune the model to improve it.

## Over-learning

A model can learn its training pictures too well and then stumble on new ones (*overfitting*). Freezing most of the network is one protection against it. Comparing behavior on training pictures with behavior on test pictures is how we notice it.

## Optional: fine-tuning

After the first round you can unfreeze everything and train a few more epochs with a much smaller learning rate (0.0001). That usually raises accuracy. It is included in the notebook as an optional step.
