# Multiple choices

Oultines allows you to make sure the generated text is chosen between different options:

```python
from outlines import models, generate

model = models.transformers("microsoft/Phi-3-mini-4k-instruct")
generator = generate.choice(model, ["skirt", "dress", "pen", "jacket"])
answer = generator("Pick the odd word out: skirt, dress, pen, jacket")

```

!!! Note "Performance"

    `generation.choice` computes an index that helps Outlines guide generation. This can take some time, but only needs to be done once. If you want to generate from the same list of choices several times make sure that you only call `generate.choice` once.
