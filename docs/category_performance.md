# Per-Category Performance & Thresholds

## Leather

**See [category_performance_leather.md](category_performance_leather.md) for
threshold, precision, recall, and known limitations specific to leather.**

## Wood

**See [category_performance_wood.md](category_performance_wood.md) for
threshold, precision, recall, and known limitations specific to wood.**

## General Notes

These are test-set performance measurements, not guarantees about every future
production image. A "normal" result does not guarantee defect-free material, since
the model can miss some defects.

**Category comparison:** the same threshold-selection methodology produced
different operating points for each material. Leather's threshold favors
precision (no false alarms, but ~8% of defects missed), while wood's threshold
favors recall (nearly all defects caught, but ~6% false alarm rate). This
reflects differences in how each material's normal-vs-defect score distributions
separate, not a difference in methodology.
