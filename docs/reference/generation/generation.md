---
title: Generation
---

# Generation

Once an [Outlines model](../models) is constructed you can use `outlines.generate` to generate text. Standard LLM generation is possible via `outlines.generate.text`, along with a variety of structured generation methods described below. (For a detailed technical explanation of how structured generation works, you may review the [Structured Generation Explanation](./structured_generation_explanation.md) page)

Before generating text, you must construct an `outlines.model`. Example:

```python
import outlines

model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct", device="cuda")
```
### Text generator

```python
generator = outlines.generate.text(model)

result = generator("Question: What's 2+2? Answer:", max_tokens=100)
print(result)
# The answer is 4

# Outlines also supports streaming output
stream = generator.stream("What's 2+2?", max_tokens=4)
for i in range(5):
	token = next(stream)
	print(repr(token))
# '2'
# '+'
# '2'
# ' equals'
# '4'
```

### [Multi-label classification](./choices.md)

Outlines allows you to do multi-label classification by guiding the model so it can only output either of the specified choices:

```python
import outlines

model = outlines.models.transformers("microsoft/Phi-3-mini-128k-instruct")
generator = outlines.generate.choice(model, ["Blue", "Red", "Yellow"])

color = generator("What is the closest color to Indigo? ")
print(color)
# Blue
```

### [JSON-structured generation](./json.md)

Outlines can guide models so that they output valid JSON **100%** of the time. You can either specify the structure using [Pydantic][pydantic]{:target="_blank"} or a string that contains a [JSON Schema][jsonschema]{:target="_blank"}:

=== "Pydantic"

    ```python
    from enum import Enum
    from pydantic import BaseModel, constr, conint

    import outlines

    class Armor(str, Enum):
        leather = "leather"
        chainmail = "chainmail"
        plate = "plate"


    class Character(BaseModel):
        name: constr(max_length=10)
        age: conint(gt=18, lt=99)
        armor: Armor
        strength: conint(gt=1, lt=100)

    model = outlines.models.transformers("microsoft/Phi-3-mini-128k-instruct")
    generator = outlines.generate.json(model, Character)

    character = generator(
        "Generate a new character for my awesome game: "
        + "name, age (between 1 and 99), armor and strength. "
        )
    print(character)
    # name='Orla' age=21 armor=<Armor.plate: 'plate'> strength=8
    ```

=== "JSON Schema"

    ```python
    import outlines

    schema = """{
        "$defs": {
            "Armor": {
                "enum": ["leather", "chainmail", "plate"],
                "title": "Armor",
                "type": "string"
            }
        },
        "properties": {
            "name": {"maxLength": 10, "title": "Name", "type": "string"},
            "age": {"title": "Age", "type": "integer"},
            "armor": {"$ref": "#/$defs/Armor"},
            "strength": {"title": "Strength", "type": "integer"}\
        },
        "required": ["name", "age", "armor", "strength"],
        "title": "Character",
        "type": "object"
    }"""

    model = outlines.models.transformers("microsoft/Phi-3-mini-128k-instruct")
    generator = outlines.generate.json(model, schema)
    character = generator(
        "Generate a new character for my awesome game: "
        + "name, age (between 1 and 99), armor and strength. "
        )
    print(character)
    # {'name': 'Yuki', 'age': 24, 'armor': 'plate', 'strength': 3}
    ```

!!! Note

    We advise you to constrain the length of the strings fields when first testing your schema, especially with small models.

### [Grammar-structured generation](./cfg.md)

Outlines also allows to generate text that is valid to any [context-free grammar][cfg]{:target="_blank"} (CFG) in the [EBNF format][ebnf]{:target="_blank"}. Grammars can be intimidating, but they are a very powerful tool! Indeed, they determine the syntax of every programming language, valid chess moves, molecule structure, can help with procedural graphics generation, etc.

Here we show a simple example of a grammar that defines arithmetic operations:

```python
from outlines import models, generate

arithmetic_grammar = """
    ?start: sum

    ?sum: product
        | sum "+" product   -> add
        | sum "-" product   -> sub

    ?product: atom
        | product "*" atom  -> mul
        | product "/" atom  -> div

    ?atom: NUMBER           -> number
         | "-" atom         -> neg
         | "(" sum ")"

    %import common.NUMBER
    %import common.WS_INLINE

    %ignore WS_INLINE
"""

model = models.transformers("microsoft/Phi-3-mini-128k-instruct")
generator = generate.cfg(model, arithmetic_grammar, max_tokens=100)

result = generator("Question: How can you write 5*5 using addition?\nAnswer:")
print(result)
# 5+5+5+5+5
```


EBNF grammars can be cumbersome to write. This is why Outlines provides grammar definitions in the `outlines.grammars.` module

```python
from outlines import models, generate, grammars

model = models.transformers("microsoft/Phi-3-mini-128k-instruct")
generator = generate.cfg(model, grammars.arithmetic, max_tokens=100)

result = generator("Question: How can you write 5*5 using addition?\nAnswer:")
print(result)
# 5+5+5+5+5
```

The available grammars are listed [here](https://github.com/outlines-dev/outlines/tree/main/outlines/grammars).


### [Regex-structured generation](./regex.md)

Slightly simpler, but no less useful, Outlines can generate text that is in the language of a [regular expression](https://www.regular-expressions.info/tutorial.html). For instance to force the model to generate IP addresses:

```python
from outlines import models, generate

model = models.transformers("microsoft/Phi-3-mini-128k-instruct")

regex_str = r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)"
generator = generate.regex(model, regex_str)

result = generator("What is the IP address of localhost?\nIP: ")
print(result)
# 127.0.0.100
```

### [Generate a given Python type](./types.md)

We provide a shortcut to regex-structured generation for simple use cases. Pass a Python type to the `outlines.generate.format` function and the LLM will output text that matches this type:

```python
from outlines import models, generate

model = models.transformers("microsoft/Phi-3-mini-128k-instruct")
generator = generate.format(model, int)

result = generator("What is 2+2?")
print(result)
# 4
```
