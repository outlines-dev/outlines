---
title: Analogical Prompting
---

# Analogical Prompting


Analogical Prompting automatically generates exemplars that include chains-of-thought (CoT) reasoning for a given task. It aims to improve performance on tasks like mathematical reasoning and code generation by providing the language model with relevant analogous examples to reference. The technique works by:

1. Analyzing the target task/problem
2. Generating analogous example problems 
3. Producing CoT solutions for those examples
4. Including the analogous examples + solutions as part of the prompt

This allows the model to see relevant reasoning patterns before approaching the actual task.

Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).
    

## A worked example


Let's say we want to use Analogical Prompting to help solve a complex math word problem. Here are the steps:

1. Analyze target problem: 
"A store is having a 30% off sale. If an item originally costs $80, how much will it cost after the discount?"

2. Generate analogous example:
"A restaurant offers a 25% discount for early bird diners. If a meal normally costs $60, how much would it cost with the discount?"

3. Produce CoT solution for example:
"Let's approach this step-by-step:
1) The discount is 25% of $60
2) 25% = 0.25
3) 0.25 * $60 = $15 (this is the discount amount)
4) The discounted price is the original price minus the discount
5) $60 - $15 = $45
Therefore, the meal would cost $45 with the early bird discount."

4. Construct final prompt:
"Here's an example of calculating a discounted price:

Q: A restaurant offers a 25% discount for early bird diners. If a meal normally costs $60, how much would it cost with the discount?

A: Let's approach this step-by-step:
1) The discount is 25% of $60
2) 25% = 0.25
3) 0.25 * $60 = $15 (this is the discount amount)
4) The discounted price is the original price minus the discount
5) $60 - $15 = $45
Therefore, the meal would cost $45 with the early bird discount.

Now, please solve this problem:

Q: A store is having a 30% off sale. If an item originally costs $80, how much will it cost after the discount?

A: Let's solve this step-by-step:"

By providing this analogous example with detailed reasoning, we give the model a pattern to follow when solving the target problem.
    
## Code Example





```python
from pydantic import BaseModel, Field
from typing import List
import outlines

class Step(BaseModel):
    description: str

class Problem(BaseModel):
    question: str
    steps: List[Step]
    solution: float

class AnalogicalPromptResponse(BaseModel):
    example_problem: Problem
    target_problem: Problem

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.1", device="cuda")

prompt = """
Use Analogical Prompting to solve a math problem. Here's an example:

Q: A restaurant offers a 25% discount for early bird diners. If a meal normally costs $60, how much would it cost with the discount?

A: Let's approach this step-by-step:
1) The discount is 25% of $60
2) 25% = 0.25
3) 0.25 * $60 = $15 (this is the discount amount)
4) The discounted price is the original price minus the discount
5) $60 - $15 = $45
Therefore, the meal would cost $45 with the early bird discount.

Now, solve this problem using the same step-by-step approach:

Q: A store is having a 30% off sale. If an item originally costs $80, how much will it cost after the discount?

Provide the solution in a structured format.
"""

generator = outlines.generate.json(model, AnalogicalPromptResponse)
response = generator(prompt)
print(response)
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)


    example_problem=Problem(question='A restaurant offers a 25% discount for early bird diners. If a meal normally costs $60, how much would it cost with the discount?', steps=[Step(description='Step 1: The discount is 25% of $60'), Step(description='Step 2: 25% = 0.25'), Step(description='Step 3: 0.25 * $60 = $15 (this is the discount amount)'), Step(description='Step 4: The discounted price is the original price minus the discount'), Step(description='Step 5: $60 - $15 = $45')], solution=45.0) target_problem=Problem(question='A store is having a 30% off sale. If an item originally costs $80, how much will it cost after the discount?', steps=[Step(description='Step 1: The discount is 30% of $80'), Step(description='Step 2: 30% = 0.3'), Step(description='Step 3: 0.3 * $80 = $24 (this is the discount amount)'), Step(description='Step 4: The discounted price is the original price minus the discount'), Step(description='Step 5: $80 - $24 = $56')], solution=56.0)

