# Results and limits

*The honest numbers, including the ones that are not flattering.*

## The headline

On the **{{n_test}}** hidden scans, the model was right **{{acc}}** of the time. A random guess among four groups would score 25%.

## Group by group

| Group | Found (recall) | Correct when it says it (precision) |
|---|---|---|
| Glioma | {{rec_glioma}} | {{prec_glioma}} |
| Meningioma | {{rec_meningioma}} | {{prec_meningioma}} |
| No tumor | {{rec_notumor}} | {{prec_notumor}} |
| Pituitary tumor | {{rec_pituitary}} | {{prec_pituitary}} |

## What the numbers tell us

- **It is good at spotting "no tumor"** and at pituitary tumors. These have clearer, more consistent looks.
- **It misses many gliomas.** Gliomas are the most varied group. Many were called meningioma. When the model says "glioma" it is usually right (high precision), but it only says so for part of the real gliomas (low recall). In a medical setting, missing a tumor is the costly error.
- **It is sometimes confidently wrong.** Its average confidence on wrong answers was {{conf_wrong}}, against {{conf_right}} on right answers. Confidence is not a reliable warning light here.

## Why this is not good enough

About 1 scan in 8 is wrong. A tool used on thousands of people would give thousands of wrong answers. Real medical AI is held to far higher standards, tested on data from many hospitals, checked for fairness across groups of people, and approved by regulators.

## Known limits

- Trained on one public collection; may not work on other scanners or hospitals.
- One slice only; real diagnosis uses the whole scan.
- Four answers only; it cannot say "something else" or "I don't know".
- Heatmaps are coarse and are not tumor outlines.
- The score is measured once on one test set; a different test set would give a different number.
- Simple version: only the last layer was trained.

## What would improve it

Fine-tuning the whole network, adding data variety, using several slices per patient, calibrating confidence, and testing on data from other sources. See *Future plans*.
