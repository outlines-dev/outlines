---
title: Self-Ask
---

# Self-Ask


Self-Ask is a prompting technique that encourages the language model to break down complex questions into simpler sub-questions, answer those sub-questions, and then use that information to answer the original question. This technique involves prompting the model to first determine if it needs to ask follow-up questions, generate those questions if needed, answer them, and finally answer the original question based on the accumulated information.
    
Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).

## A worked example


1. The Self-Ask technique involves breaking down complex questions into simpler sub-questions. This aligns well with structured generation.

2. We can use the Outlines library to implement this technique by:
   - Using a model that supports instruction-following (e.g., Mistral-7B-Instruct-v0.2)
   - Defining a structured output for the self-ask process
   - Using outlines.generate.json to generate the structured output

3. We'll create a Pydantic model to represent the Self-Ask process:
   - Include fields for the original question, follow-up questions, their answers, and the final answer
   - Use typing.List for multiple follow-up questions and answers

4. The prompt will include instructions for the Self-Ask process and the original question

5. We'll use outlines.generate.json to generate the structured output

Based on this analysis, here's the code snippet:

## Code Example

```python
from typing import List, Optional
from pydantic import BaseModel

import outlines

class SelfAskResponse(BaseModel):
    original_question: str
    needs_followup: bool
    followup_questions: Optional[List[str]]
    followup_answers: Optional[List[str]]
    final_answer: str

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.1", device="cuda")

prompt = """
Use the Self-Ask technique to answer the following question. First, determine if you need any additional information. If so, ask yourself follow-up questions and answer them. Then, use all the information to answer the original question.

Original question: What was the population of New York City in the year the Empire State Building was completed?
"""

generator = outlines.generate.json(model, SelfAskResponse)
response = generator(prompt)

print(f"Original question: {response.original_question}")
print(f"Needs follow-up: {response.needs_followup}")
if response.needs_followup:
    for q, a in zip(response.followup_questions, response.followup_answers):
        print(f"Follow-up Q: {q}")
        print(f"Follow-up A: {a}")
print(f"Final answer: {response.final_answer}")
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    Compiling FSM index for all state transitions: 100%|█| 172/172 [00:01<00:00
    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)


    Original question: What was the population of New York City in the year the Empire State Building was completed?
    Needs follow-up: True
    Follow-up Q: What year was the Empire State Building completed?
    Follow-up A: The Empire State Building was completed in 1931.
    Follow-up Q: What was the population of New York City in that year?
    Follow-up A: According to the United States Census Bureau, the estimated population of New York City in 1931 was 5,633,551.
    Final answer: According to the United States Census Bureau, the estimated population of New York City in 1931, the year the Empire State Building was completed, was 5,633,551.

