# Example Explanations

## Normal Result Close to Threshold (Leather)

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

## Defect Result Close to Threshold (Leather)

**Anomaly score: `0.537`**

> **Result: Anomaly detected.**
>
> The anomaly score is `0.537`, above the leather defect threshold of `0.5046`. The
> anomaly signal is present but not as strong as it would be for a substantially
> higher score. The score is not a probability or confidence percentage.

## Stronger Anomaly Signal (Leather)

**Anomaly score: `0.91`**

> **Result: Anomaly detected.**
>
> The anomaly score is `0.91`, substantially above the leather defect threshold of
> `0.5046`, indicating a strong anomaly signal. This does not mean a 91% probability
> of a defect — the anomaly score is not a calibrated probability or confidence value.

## Normal Result Close to Threshold (Wood)

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

## Defect Result with a Strong Signal (Wood)

**Anomaly score: `0.71`**

> **Result: Anomaly detected.**
>
> The anomaly score is `0.71`, well above the wood defect threshold of `0.5002`,
> indicating a clear anomaly signal. This does not mean a 71% probability of a
> defect — the anomaly score is not a calibrated probability or confidence value.
