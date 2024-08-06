---
title: Cumulative Reasoning
---

# Cumulative Reasoning


Cumulative Reasoning is an advanced prompting technique that builds on the Chain-of-Thought (CoT) approach. It involves breaking down a complex problem into smaller, manageable steps and progressively accumulating knowledge and insights. In this technique, the language model is guided to solve a problem by incrementally building up its understanding, with each step building on the previous ones. This allows for a more structured and thorough problem-solving process, especially for tasks that require multiple levels of reasoning or complex logical deductions.
    
Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).

## A worked example

None    
## Code Example





```python
import outlines
from pydantic import BaseModel, conint

# Set up the model
model = outlines.models.transformers("WizardLM/WizardMath-7B-V1.1")
generator = outlines.generate.format(model, int)

# Define the problem
total_apples = 150

# Step 1: Calculate apples sold on Monday
prompt_monday = f"30% of {total_apples} = "
apples_sold_monday = generator(prompt_monday)

# Step 2: Calculate apples remaining after Monday
prompt_remaining_monday = f"{total_apples} - {apples_sold_monday} = "
apples_remaining_monday = generator(prompt_remaining_monday)

# Step 3: Calculate apples sold on Tuesday
prompt_tuesday = f"25% of {apples_remaining_monday} = "
apples_sold_tuesday = generator(prompt_tuesday)

# Step 4: Calculate final number of apples remaining
prompt_final = f"{apples_remaining_monday} - {apples_sold_tuesday} = "
apples_remaining = generator(prompt_final)

# Define a Pydantic model for the final result
class AppleInventory(BaseModel):
    initial_apples: conint(ge=0)
    sold_monday: conint(ge=0)
    sold_tuesday: conint(ge=0)
    remaining_apples: conint(ge=0)

# Create the final result
result = AppleInventory(
    initial_apples=total_apples,
    sold_monday=apples_sold_monday,
    sold_tuesday=apples_sold_tuesday,
    remaining_apples=apples_remaining
)

print(result)
```


    Downloading shards:   0%|          | 0/2 [00:00<?, ?it/s]



    pytorch_model-00001-of-00002.bin:   9%|9         | 933M/9.94G [00:00<?, ?B/s]



    pytorch_model-00002-of-00002.bin:   0%|          | 0.00/4.54G [00:00<?, ?B/s]



    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    /usr/local/lib/python3.10/dist-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      return self.fget.__get__(instance, owner)()



    generation_config.json:   0%|          | 0.00/111 [00:00<?, ?B/s]



    tokenizer_config.json:   0%|          | 0.00/948 [00:00<?, ?B/s]



    tokenizer.model:   0%|          | 0.00/493k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/1.80M [00:00<?, ?B/s]



    added_tokens.json:   0%|          | 0.00/42.0 [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/167 [00:00<?, ?B/s]


    Compiling FSM index for all state transitions: 100%|█| 4/4 [00:00<00:00, 44
    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)


    initial_apples=150 sold_monday=4 sold_tuesday=1 remaining_apples=1

