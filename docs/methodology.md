# How the Material Inspection Model Makes Decisions

This document explains how the anomaly detection system evaluates material images
and how results should be interpreted when communicating them to a QA operative.
This applies to all inspected material categories (leather, wood).

## 1. What the model does

For each image, the model produces a **raw anomaly score**. The score indicates how
unusual a region of the material looks compared with the normal patterns the model
has learned for that specific material.

The score is a **relative anomaly signal**:

- Higher score = more unusual compared with normal material.
- Lower score = more similar to normal material.
- The score is **not a percentage**.
- The score is **not a probability**.
- The score is **not a confidence value**.

For example, a score of `0.80` does **not** mean "80% confidence that there is a
defect." Do not convert the anomaly score into a probability or confidence percentage.

**Each material category has its own separately trained model and its own threshold**
(see Section 3). Scores are not comparable across categories.

## 2. How the model decides "defect" vs. "normal"

A raw anomaly score needs a threshold to determine the final classification:

- **Score < threshold:** classified as **normal**
- **Score >= threshold:** classified as **defect/anomaly**

The threshold for each category was selected using a separate labeled validation set,
kept apart from the independent test set used to evaluate final performance. The
threshold was chosen to balance two goals:

- **Recall:** avoid missing real defects.
- **Precision:** avoid incorrectly flagging good material as defective.

### Interpreting the distance from the threshold

The distance from the threshold can provide useful context, but it must not be
described as a probability or confidence. A score well below the threshold represents
a weak anomaly signal; a score well above it represents a strong one. These describe
signal strength relative to the threshold, not a calibrated probability that a
physical defect exists.

## 3. Per-category thresholds and performance

### Leather

- **Threshold: `0.5046`**
- **Precision: 100%** (no false alarms on the 100-image held-out test set)
- **Recall: 92%** (approximately 9 out of 10 real defects detected)
- **Known limitation:** subtle discoloration (the "color" defect type) tended to
  score closest to the normal range, making it the hardest case for the model.
  A normal result does not reliably rule out subtle color defects.

### Wood

- **Threshold: `0.5002`**
- **Precision: 94%** (a small number of false alarms on the 64-image held-out test set)
- **Recall: 98%** (nearly all real defects detected)
- **Known limitation:** liquid-type defects consistently scored closest to the
  normal range, making them the hardest case for the model to catch confidently.
  Scratch-type defects showed high score variance — some scored clearly above
  the threshold, others fell much closer to it — meaning detection confidence
  for scratches is inconsistent rather than uniformly weak.

These are test-set performance measurements, not guarantees about every future
production image. A "normal" result does not guarantee defect-free material, since
the model can miss some defects.

**Category comparison:** the same threshold-selection methodology produced
different operating points for each material. Leather's threshold favors
precision (no false alarms, but ~8% of defects missed), while wood's threshold
favors recall (nearly all defects caught, but ~6% false alarm rate). This
reflects differences in how each material's normal-vs-defect score distributions
separate, not a difference in methodology.

## 4. What the model does and does not tell you

The current model provides a **binary classification** based on the anomaly score
and threshold: **defect/anomaly** or **normal**.

The model does **not** currently provide:

- A calibrated probability that an image contains a defect.
- A confidence percentage such as "80% confident."
- A diagnosis of the exact defect type.
- The physical cause of a defect.
- A measurement of defect severity.
- A guarantee that a normal image contains no defect.

A future version may provide additional calibrated information.

## 5. Rules for explaining a result to a QA operative

1. **State the classification** — defect/anomaly or normal.
2. **State the material category and the anomaly score.**
3. **Compare the score with that category's threshold** — mention the threshold
   value, and whether the score is close to or substantially past it.
4. **Do not invent a confidence or probability** — never say "the model is 80%
   confident" or "there is an 80% chance of a defect" based on an anomaly score.
5. **Do not overstate what the model detected** — it identifies unusual material
   appearance, not exact defect type, cause, or severity.
6. **Mention category-specific known limitations when relevant** (Section 3).
7. **Keep aggregate performance separate from the individual prediction** — test-set
   precision/recall describe overall performance, not the confidence of any single
   current image.

## 6. Example explanations

### Example: normal result close to the threshold (leather)

**Anomaly score: `0.48`**

> **Result: Normal.**
>
> The anomaly score is `0.48`, which is below the leather defect threshold of
> `0.5046`. The model therefore classifies this image as normal. The score is
> relatively close to the threshold, so the image is less clearly separated from
> the decision boundary than an image with a much lower score.
>
> This score is not a probability or confidence percentage. Subtle discoloration is
> a known difficult case for this model and may require additional visual attention.

### Example: defect result close to the threshold (leather)

**Anomaly score: `0.537`**

> **Result: Anomaly detected.**
>
> The anomaly score is `0.537`, above the leather defect threshold of `0.5046`. The
> anomaly signal is present but not as strong as it would be for a substantially
> higher score. The score is not a probability or confidence percentage.

### Example: stronger anomaly signal (leather)

**Anomaly score: `0.91`**

> **Result: Anomaly detected.**
>
> The anomaly score is `0.91`, substantially above the leather defect threshold of
> `0.5046`, indicating a strong anomaly signal. This does not mean a 91% probability
> of a defect — the anomaly score is not a calibrated probability or confidence value.

### Example: normal result close to the threshold (wood)

**Anomaly score: `0.499`**

> **Result: Normal.**
>
> The anomaly score is `0.499`, just below the wood defect threshold of `0.5002`.
> The model classifies this image as normal, but the score sits very close to the
> decision boundary. Given that liquid-type defects are known to score close to
> the normal range for this model, a result this close to the threshold may
> warrant a closer visual check.
>
> This score is not a probability or confidence percentage.

### Example: defect result with a strong signal (wood)

**Anomaly score: `0.71`**

> **Result: Anomaly detected.**
>
> The anomaly score is `0.71`, well above the wood defect threshold of `0.5002`,
> indicating a clear anomaly signal. This does not mean a 71% probability of a
> defect — the anomaly score is not a calibrated probability or confidence value.

## 7. Important principle

The model answers:

> **"Does this image contain an appearance that is unusual compared with normal
> [material]?"**

It does not, by itself, answer:

> **"What exact defect is present, why did it occur, how severe is it, or what is the
> probability that the material is defective?"**

Those distinctions should be preserved when communicating model results to QA.