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

Each material category has its own separately trained model with a unique threshold,
precision, recall, and known limitations.

**See [category_performance.md](category_performance.md) for detailed thresholds,
test-set metrics, and known limitations for each material.**

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

**See [explanation_rules.md](explanation_rules.md) for complete guidelines on how
to communicate model results, including what to include, what to avoid, and
special considerations when scores are close to the threshold.**

## 6. Example explanations

**See [examples.md](examples.md) for concrete examples showing how to explain
results for both leather and wood, including cases where scores are close to or
well above the threshold.**

## 7. Important principle

The model answers:

> **"Does this image contain an appearance that is unusual compared with normal
> [material]?"**

It does not, by itself, answer:

> **"What exact defect is present, why did it occur, how severe is it, or what is the
> probability that the material is defective?"**

Those distinctions should be preserved when communicating model results to QA.