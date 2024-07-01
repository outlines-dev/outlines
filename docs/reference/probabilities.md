# Multiple choices with Probabilities

Outlines allows you to generate probabilities for different options, giving you insights into the model's confidence for each choice.

```python
from outlines import models, generate

model = models.transformers("mistralai/Mistral-7B-v0.1")
probabilities = generate.probabilities(model, ["skirt", "dress", "pen", "jacket"])
answer = probabilities("Pick the odd word out: skirt, dress, pen, jacket")
print(answer)
```

!!! Warning "Compatibility"

    `generate.probabilities` uses a beam search sampler. It is not compatible with other samplers. Ensure that no other samplers are used in conjunction with this method.

## How It Works


Beam search is a heuristic search algorithm used to explore the most promising sequences in a limited set. In text generation, it maintains the top `k` sequences (beams) at each step based on their cumulative probabilities. Each sequence has a weight, which is the product of the probabilities of its tokens, representing the likelihood of the sequence according to the model.

!!! Warning "Probabilities Summation"

    The probabilities returned by `generate.probabilities` might not sum to one because the `topk` limitation only keeps the best sequences. This means other sequences with potentially non-negligible probabilities are not taken into account, leading to an incomplete probability distribution.
