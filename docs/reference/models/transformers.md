# transformers


!!! Installation

    You need to install the `transformer`, `datasets` and `torch` libraries to be able to use these models in Outlines:

    ```bash
    pip install torch transformers datasets
    ```


Outlines provides an integration with the `torch` implementation of causal models in the [transformers][transformers] library. You can initialize the model by passing its name:

```python
from outlines import models

model = models.transformers("mistralai/Mistral-7B-v0.3", device="cuda")
```

If you need more fine-grained control you can also initialize the model and tokenizer separately:


```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from outlines import models

llm = AutoModelForCausalLM.from_pretrained("gpt2", output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = models.Transformers(llm, tokenizer)
```

# Using Logits Processors

There are two ways to use Outlines Structured Generation with HuggingFace Transformers:
- 1) Use Outlines generation wrapper, `outlines.models.transformers`
- 2) Use `OutlinesLogitsProcessor` with `transformers.AutoModelForCausalLM`

Outlines supports a myriad of logits processors for structured generation. In these example, we will use the `RegexLogitsProcessor` which guarantees generated text matches the specified pattern.

## Example: `outlines.models.transformers`

```
import outlines

time_regex_pattern = r"(0?[1-9]|1[0-2]):[0-5]\d\s?(am|pm)?"

model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct", device="cuda")
generator = outlines.generate.regex(model, time_regex_pattern)

output = generator("The the best time to visit a dentist is at ")
print(output)
# 2:30 pm
```

## Example: Direct `transformers` library use

```
import outlines
import transformers


model_uri = "microsoft/Phi-3-mini-4k-instruct"

outlines_tokenizer = outlines.models.TransformerTokenizer(
    transformers.AutoTokenizer.from_pretrained(model_uri)
)
phone_number_logits_processor = outlines.processors.RegexLogitsProcessor(
    "\\+?[1-9][0-9]{7,14}",  # phone number pattern
    outlines_tokenizer,
)

generator = transformers.pipeline('text-generation', model=model_uri)

output = generator(
    "Jenny gave me her number it's ",
	logits_processor=transformers.LogitsProcessorList([phone_number_logits_processor])
)
print(output)
# [{'generated_text': "Jenny gave me her number it's 2125550182"}]
# not quite 8675309 what we expected, but it is a valid phone number
```

[transformers]: https://github.com/huggingface/transformers
