---
title: Chain-of-Thought (CoT) Prompting
---

# Chain-of-Thought (CoT) Prompting


Chain-of-Thought (CoT) Prompting is a technique that encourages large language models to express their reasoning process step-by-step before providing a final answer. It typically uses few-shot prompting, where the prompt includes examples of questions along with their step-by-step reasoning and final answers. This approach helps the model break down complex problems into smaller, more manageable steps, often improving performance on tasks that require multi-step reasoning, such as math problems or logical deductions.
    
Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).

## A worked example


To implement Chain-of-Thought (CoT) Prompting:

1. Prepare your prompt with one or more examples that demonstrate the desired chain of thought. For instance:

Q: A store has 25 apples. If they sell 12 apples and then receive a shipment of 18 more, how many apples does the store have now?
A: Let's think through this step-by-step:
1. The store starts with 25 apples.
2. They sell 12 apples, so: 25 - 12 = 13 apples left.
3. They receive 18 more apples, so: 13 + 18 = 31 apples.
Therefore, the store now has 31 apples.

2. Add your actual question after the example(s):

Q: A bakery sold 45 cakes on Monday, 38 cakes on Tuesday, and 52 cakes on Wednesday. If each cake costs $12, how much money did the bakery make in total over these three days?
A: Let's approach this step-by-step:

3. Submit this prompt to the language model.

4. The model should now provide a step-by-step reasoning process for your question, followed by the final answer. For example:

1. First, let's calculate the total number of cakes sold:
   Monday: 45 cakes
   Tuesday: 38 cakes
   Wednesday: 52 cakes
   Total cakes = 45 + 38 + 52 = 135 cakes

2. Now, we know each cake costs $12.

3. To find the total money made, we multiply the number of cakes by the price per cake:
   Total money = 135 cakes × $12 per cake = $1,620

Therefore, the bakery made $1,620 in total over these three days.

5. Review the model's reasoning and final answer for accuracy and coherence.
    
## Code Example





```python
from pydantic import BaseModel
from typing import List
import outlines

class ChainOfThoughtResponse(BaseModel):
    steps: List[str]
    final_answer: str

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.1", device="cuda")

prompt = """You are a math assistant that provides step-by-step solutions.

Example:
Q: A store has 25 apples. If they sell 12 apples and then receive a shipment of 18 more, how many apples does the store have now?
A:
Steps:
1. The store starts with 25 apples.
2. They sell 12 apples, so: 25 - 12 = 13 apples left.
3. They receive 18 more apples, so: 13 + 18 = 31 apples.
Final answer: The store now has 31 apples.

Now, please solve this problem using the same step-by-step approach:
Q: A bakery sold 45 cakes on Monday, 38 cakes on Tuesday, and 52 cakes on Wednesday. If each cake costs $12, how much money did the bakery make in total over these three days?
A:
"""

generator = outlines.generate.json(model, ChainOfThoughtResponse)
response = generator(prompt)

print("Steps:")
for step in response.steps:
    print(f"- {step}")
print(f"\nFinal Answer: {response.final_answer}")
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)


    Steps:
    - On Monday, the bakery sold 45 cakes. Each day, they sell cakes at a price of $12. So, on Monday, the bakery made 45 x $12 = $540. 
    
    Final Answer: In total, the bakery made $540 + $38 + $52 = $540. 

