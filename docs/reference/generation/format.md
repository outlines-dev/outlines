# Type constraints

We can ask completions to be restricted to valid python types:

```python
from outlines import models, generate

model = models.transformers("microsoft/Phi-3-mini-4k-instruct")
generator = generate.format(model, int)
answer = generator("When I was 6 my sister was half my age. Now I’m 70 how old is my sister?")
print(answer)
# 67
```

The following types are currently available:

- int
- float
- bool
- datetime.date
- datetime.time
- datetime.datetime
- We also provide [custom types](types.md)
