# Text generation

Outlines provides a unified interface to generate text with many language models, API-based and local. The same pattern is used throughout the library:

1. Instantiate a generator by calling `outlines.generate.text` with the model to be used.
2. Call the generator with the prompt and (optionally) some generation parameters.


```python
from outlines import models, generate

model = models.openai("gpt-4")
generator = generate.text(model)
answer = generator("What is 2+2?")

model = models.transformers("microsoft/Phi-3-mini-4k-instruct")
generator = generate.text(model)
answer = generator("What is 2+2?")
```

By default Outlines uses the multinomial sampler with `temperature=1`. See [this section](samplers.md) to learn how to use different samplers.

## Streaming

Outlines allows you to stream the model's response by calling the `.stream` method of the generator with the prompt:


```python
from outlines import models, generate

model = models.transformers("microsoft/Phi-3-mini-4k-instruct")
generator = generate.text(model)

tokens = generator.stream("What is 2+2?")
for token in tokens:
    print(token)
```

## Parameters

### Limit the number of tokens generated

To limit the number of tokens generated you can pass the `max_tokens` positional argument to the generator:

```python
from outlines import models, generate

model = models.transformers("microsoft/Phi-3-mini-4k-instruct")
generator = generate.text(model)

answer = generator("What is 2+2?", 5)
answer = generator("What is 2+2?", max_tokens=5)
```

### Stop after a given string is generated

You can also ask the model to stop generating text after a given string has been generated, for instance a period or a line break. You can pass a string or a line of string for the `stop_at` argument:


```python
from outlines import models, generate

model = models.transformers("microsoft/Phi-3-mini-4k-instruct")
generator = generate.text(model)

answer = generator("What is 2+2?", stop_at=".")
answer = generator("What is 2+2?", stop_at=[".", "\n"])
```

*The stopping string will be included in the response.*


### Seed the generation

It can be useful to seed the generation in order to get reproducible results:

```python
import torch
from outlines import models, generate

model = models.transformers("microsoft/Phi-3-mini-4k-instruct")

seed = 789001

answer = generator("What is 2+2?", seed=seed)
```
