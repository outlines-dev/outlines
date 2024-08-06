---
title: Contrastive CoT Prompting
---

# Contrastive CoT Prompting


Contrastive CoT Prompting adds both exemplars with incorrect and correct explanations to the Chain-of-Thought prompt in order to show the language model how not to reason, as well as how to reason correctly. This technique aims to improve the model's ability to distinguish between valid and invalid reasoning paths, leading to more accurate responses. By presenting contrasting examples, the model learns to avoid common pitfalls and strengthens its understanding of proper problem-solving approaches.
    
Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).

## A worked example


To implement Contrastive CoT Prompting:

1. Identify the task or problem type you want the model to solve.

2. Create several example problems with both correct and incorrect reasoning paths:
   a. Write out the problem statement.
   b. Provide an incorrect chain of reasoning, clearly labeled as incorrect.
   c. Explain why the reasoning is flawed.
   d. Provide the correct chain of reasoning, clearly labeled as correct.
   e. Explain why this reasoning is valid.

3. Construct your prompt by combining:
   a. A clear instruction of the task.
   b. The contrasting examples you created (2-3 is usually sufficient).
   c. The new problem you want the model to solve.

4. Submit the prompt to the language model.

Here's a simple example for basic math word problems:

Instruction: Solve the following math word problem. Avoid common mistakes in reasoning and calculation.

Example 1:
Problem: Tom has 5 apples and gives 2 to his friend. How many apples does Tom have left?

Incorrect reasoning (DO NOT FOLLOW):
Tom starts with 5 apples. He gives away 2, so we add 2 to 5. Tom now has 7 apples.
This reasoning is wrong because we're adding when we should be subtracting.

Correct reasoning (FOLLOW THIS):
Tom starts with 5 apples. He gives away 2, so we subtract 2 from 5. 5 - 2 = 3. Tom now has 3 apples.
This reasoning is correct because we're subtracting the apples Tom gave away.

New Problem:
Sarah has 10 cookies. She eats 3 and then bakes 5 more. How many cookies does Sarah have now?

Please solve this problem using correct reasoning.

5. Review the model's response and assess if it followed the correct reasoning path while avoiding the pitfalls demonstrated in the incorrect examples.
    
## Code Example





```python
from pydantic import BaseModel, conint
import outlines

class MathSolution(BaseModel):
    reasoning_steps: list[str]
    final_answer: conint(gt=0)

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.1", device="cuda")
generator = outlines.generate.json(model, MathSolution)

prompt = """Solve the following math word problem. Avoid common mistakes in reasoning and calculation.

Example 1:
Problem: Tom has 5 apples and gives 2 to his friend. How many apples does Tom have left?

Incorrect reasoning (DO NOT FOLLOW):
Tom starts with 5 apples. He gives away 2, so we add 2 to 5. Tom now has 7 apples.
This reasoning is wrong because we're adding when we should be subtracting.

Correct reasoning (FOLLOW THIS):
Tom starts with 5 apples. He gives away 2, so we subtract 2 from 5. 5 - 2 = 3. Tom now has 3 apples.
This reasoning is correct because we're subtracting the apples Tom gave away.

New Problem:
Sarah has 10 cookies. She eats 3 and then bakes 5 more. How many cookies does Sarah have now?

Please solve this problem using correct reasoning. Provide your reasoning steps and the final answer."""

solution = generator(prompt)
print(solution)
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)


    reasoning_steps=['Sarah starts with 10 cookies.', 'She eats 3 cookies, reducing her total to 10 - 3 = 7.', 'Then she bakes 5 more cookies, increasing her total to 7 + 5 = 12.'] final_answer=12

