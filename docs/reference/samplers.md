# Samplers

Outlines offers different sequence sampling algorithms, and we will integrate more in the future. You can read [this blog post](https://huggingface.co/blog/how-to-generate) for an overview of the different sampling algorithm.

## Multinomial sampling

Outlines defaults to the multinomial sampler without top-p or top-k sampling, and temperature equal to 1. Not specifying a sampler is equivalent to:

```python
from outlines import models, generate, samplers


model = models.transformers("microsoft/Phi-3-mini-4k-instruct")
sampler = samplers.multinomial()

generator = generate.text(model, sampler)
answer = generator("What is 2+2?")

print(answer)
# 4
```

You can ask the generator to take multiple samples by passing the number of samples when initializing the sampler:

```python
from outlines import models, generate, samplers


model = models.transformers("microsoft/Phi-3-mini-4k-instruct")
sampler = samplers.multinomial(3)

generator = generate.text(model, sampler)
answer = generator("What is 2+2?")

print(answer)
# [4, 4, 4]
```

If you ask multiple samples for a batch of prompt the returned array will be of shape `(num_samples, num_batches)`:

```python
from outlines import models, generate, samplers


model = models.transformers("microsoft/Phi-3-mini-4k-instruct")
sampler = samplers.multinomial(3)

generator = generate.text(model, sampler)
answer = generator(["What is 2+2?", "What is 3+3?"])

print(answer)
# [[4, 4, 4], [6, 6, 6]]
```

### Top-k sampling

You can ask Outlines to only consider the top-k logits at each step by specifying the value of the `top-k` keyword argument when initializing the sampler.

```python
sampler = samplers.multinomial(3, top_k=10)
```

### Top-p sampling

You can ask Outlines to only consider the highest probability tokens such that their cumulative probability is greater than a threshold `p`. Specify the `top_p` keyword argument when initializing the sampler:


```python
sampler = samplers.multinomial(3, top_p=0.95)
```

## Greedy sampler

You can also use the greedy sampler. For this you need to initialize the generator with the sampler:


```python
from outlines import models, generate, samplers


model = models.transformers("microsoft/Phi-3-mini-4k-instruct")
sampler = samplers.greedy()

generator = generate.text(model, sampler)
answer = generator("What is 2+2?")

print(answer)
# 4
```

You cannot ask for multiple samples with the greedy sampler since it does not clear what the result should be.


## Beam Search

Outlines also comes with the Beam Search sampling algorithm:

```python
from outlines import models, generate, samplers


model = models.transformers("microsoft/Phi-3-mini-4k-instruct")
sampler = samplers.beam_search(beams=5)

generator = generate.text(model, sampler)
answer = generator("What is 2+2?")

print(answer)
# 4
```


!!! Warning "Compatibility"

    Only models from the `transformers`  and `exllamav2 ` libraries are compatible with Beam Search.
