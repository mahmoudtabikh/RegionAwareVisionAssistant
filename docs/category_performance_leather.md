# Leather Performance & Threshold

- **Threshold: `0.5046`**
- **Precision: 100%** (no false alarms on the 100-image held-out test set)
- **Recall: 92%** (approximately 9 out of 10 real defects detected)
- **Known limitation:** subtle discoloration (the "color" defect type) tended to
  score closest to the normal range, making it the hardest case for the model.
  A normal result does not reliably rule out subtle color defects.

## Interpretation

The threshold of `0.5046` was selected to maximize precision—avoiding false
alarms on normal material. This means approximately 8% of real defects may be
missed. Subtle discoloration is the most difficult case for this model to detect.
