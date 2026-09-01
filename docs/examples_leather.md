# Example Explanations - Leather

## Normal Result Close to Threshold

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

## Defect Result Close to Threshold

**Anomaly score: `0.537`**

> **Result: Anomaly detected.**
>
> The anomaly score is `0.537`, above the leather defect threshold of `0.5046`. The
> anomaly signal is present but not as strong as it would be for a substantially
> higher score. The score is not a probability or confidence percentage.

## Stronger Anomaly Signal

**Anomaly score: `0.91`**

> **Result: Anomaly detected.**
>
> The anomaly score is `0.91`, substantially above the leather defect threshold of
> `0.5046`, indicating a strong anomaly signal. This does not mean a 91% probability
> of a defect — the anomaly score is not a calibrated probability or confidence value.
