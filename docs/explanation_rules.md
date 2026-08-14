# Rules for Explaining Results to QA

When communicating a model result to a QA operative, follow these guidelines:

1. **State the classification** — defect/anomaly or normal.

2. **State the material category and the anomaly score.**

3. **Compare the score with that category's threshold** — mention the threshold
   value, and whether the score is close to or substantially past it.

4. **Do not invent a confidence or probability** — never say "the model is 80%
   confident" or "there is an 80% chance of a defect" based on an anomaly score.

5. **Do not overstate what the model detected** — it identifies unusual material
   appearance, not exact defect type, cause, or severity.

6. **Mention category-specific known limitations when relevant** (see category_performance.md).

7. **Keep aggregate performance separate from the individual prediction** — test-set
   precision/recall describe overall performance, not the confidence of any single
   current image.

## What to Include When Score is Close to Threshold

- Mention that the score sits close to the decision boundary
- Reference known limitations for that category (e.g., liquid-type defects for wood, discoloration for leather)
- Suggest additional visual attention if appropriate

## What NOT to Do

- Do not convert the anomaly score to a probability or percentage
- Do not say "X% confident"
- Do not assume the model identified the exact defect type
- Do not guarantee a normal result means zero defects
