# Example Explanations - Wood

## Normal Result Close to Threshold

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

## Defect Result with a Strong Signal

**Anomaly score: `0.71`**

> **Result: Anomaly detected.**
>
> The anomaly score is `0.71`, well above the wood defect threshold of `0.5002`,
> indicating a clear anomaly signal. This does not mean a 71% probability of a
> defect — the anomaly score is not a calibrated probability or confidence value.
