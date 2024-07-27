# JSON structured generation

Outlines can make any open source model return a JSON object that follows a structure that is specified by the user. This is useful whenever we want the output of the model to be processed by code downstream: code does not understand natural language but rather the structured language it has been programmed to understand.

There are mostly two reasons why someone would want to get an output formatted as JSON from a LLM:

1. Parse the answer (e.g. with Pydantic), store it somewhere, return it to a user, etc.
2. Call a function with the result

Outlines has you covered in both cases! Indeed, to define the structure of the JSON you want the model to follow you can either provide a Pydantic model, or a function. No need to duplicate code!

## Using Pydantic

Outlines can infer the structure of the output from a Pydantic model. The result is an instance of the model that contains the values returned by the LLM:

```python
from pydantic import BaseModel

from outlines import models, generate


class User(BaseModel):
    name: str
    last_name: str
    id: int


model = models.transformers("microsoft/Phi-3-mini-4k-instruct")
generator = generate.json(model, User)
result = generator(
    "Create a user profile with the fields name, last_name and id"
)
print(result)
# User(name="John", last_name="Doe", id=11)
```

!!! Note "JSON and whitespaces"

    By default Outlines prevents the model from generating json with syntactic newlines, tabs, or multiple spaces. The default `whitespace_pattern` is `r"[ ]?"`. Small models tend to enter an infinite repetition loop if the `whitespace_pattern` allows infinite spacing. If you would like to allow the model to generate multiple tabs, newlines, and spaces, you can set the whitespace pattern as follows:

    ```python
    generator = generate.json(model, User, whitespace_pattern=r"[\n\t ]*")
    ```

!!! Note "Performance"

    `generation.json` computes an index that helps Outlines guide generation. This can take some time, but only needs to be done once. If you want to generate several times with the same schema make sure that you only call `generate.json` once.

!!! Tip "Custom types"

    Outlines provides [custom Pydantic types](types.md) so you do not have to write regular expressions for common types, such as phone numbers or zip codes.

## Using a JSON Schema

Instead of a Pydantic model you can pass a string that represents a [JSON Schema](https://json-schema.org/) specification to `generate.json`:

```python
from pydantic import BaseModel

from outlines import models
from outlines import generate

model = models.transformers("microsoft/Phi-3-mini-4k-instruct")

schema = """
{
  "title": "User",
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "last_name": {"type": "string"},
    "id": {"type": "integer"}
  },
  "required": ["name", "last_name", "id"]
}
"""

generator = generate.json(model, schema)
result = generator(
    "Create a user profile with the fields name, last_name and id"
)
print(result)
# User(name="John", last_name="Doe", id=11)
```

## From a function's signature

Outlines can infer the structure of the output from the signature of a function. The result is a dictionary, and can be passed directly to the function using the usual dictionary expansion syntax `**`:

```python
from outlines import models
from outlines import generate

def add(a: int, b: int):
    return a + b

model = models.transformers("microsoft/Phi-3-mini-4k-instruct")
generator = generate.json(model, add)
result = generator("Return two integers named a and b respectively. a is odd and b even.")

print(add(**result))
# 3
```

A great advantage of passing functions directly to specify the structure is that the structure of the LLM will change with the function's definition. No need to change the code at several places!
