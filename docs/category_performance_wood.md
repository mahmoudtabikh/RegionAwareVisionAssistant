# Wood Performance & Threshold

- **Threshold: `0.5002`**
- **Precision: 94%** (a small number of false alarms on the 64-image held-out test set)
- **Recall: 98%** (nearly all real defects detected)
- **Known limitation:** liquid-type defects consistently scored closest to the
  normal range, making them the hardest case for the model to catch confidently.
  Scratch-type defects showed high score variance — some scored clearly above
  the threshold, others fell much closer to it — meaning detection confidence
  for scratches is inconsistent rather than uniformly weak.

## Interpretation

The threshold of `0.5002` was selected to maximize recall—catching nearly all
real defects. This means approximately 6% false alarm rate on normal material.
Liquid-type defects are the most difficult case; scratches show inconsistent
detection confidence depending on how pronounced they are.
