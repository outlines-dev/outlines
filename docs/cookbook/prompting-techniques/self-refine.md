---
title: Self-Refine
---

# Self-Refine


Self-Refine is an iterative prompting framework that uses an LLM to generate an initial answer, then repeatedly prompts the same LLM to provide feedback on its own answer and improve it based on that feedback. This process continues for multiple iterations until a stopping condition is met (e.g. a maximum number of iterations or satisfactory quality is achieved). The technique aims to leverage the LLM's ability to critically analyze its own outputs and make incremental improvements.
    
Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).

## A worked example


To implement the Self-Refine technique:

1. Initial answer generation:
   Prompt: "What are the key factors that led to the Industrial Revolution?"
   LLM response: "The Industrial Revolution was caused by technological advancements and economic changes."

2. Request for feedback:
   Prompt: "Please provide feedback on the following answer to 'What are the key factors that led to the Industrial Revolution?': [Insert LLM's previous response]"
   LLM feedback: "The answer is too vague and lacks specific details. It doesn't mention important factors like agricultural improvements, population growth, or the role of coal and steam power."

3. Request for improvement:
   Prompt: "Based on this feedback, please provide an improved answer to the original question."
   LLM improved response: "The key factors that led to the Industrial Revolution include technological advancements like the steam engine, economic changes such as the rise of capitalism, agricultural improvements that increased food production, population growth providing labor, and the availability of key resources like coal."

4. Repeat steps 2-3 for multiple iterations, each time using the latest improved response.

5. Stop when a predetermined number of iterations is reached or when the improvements become minimal.

6. Use the final refined answer as the output.
    
## Code Example





```python
import re
from pydantic import BaseModel, Field
import outlines

class RefinementStep(BaseModel):
    answer: str
    feedback: str
    quality_score: int = Field(ge=1, le=10)

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.1", device="cuda")
generator = outlines.generate.json(model, RefinementStep)

question = "What are the key factors that led to the Industrial Revolution?"
max_iterations = 3
quality_threshold = 8

current_step = generator(
    f"Answer this question: {question}\n"
    "Provide feedback on your answer, including a quality score from 1-10.\n"
    "Output as JSON with fields: answer, feedback, quality_score"
)

for i in range(max_iterations - 1):
    if current_step.quality_score >= quality_threshold:
        break

    current_step = generator(
        f"Previous answer: {current_step.answer}\n"
        f"Previous feedback: {current_step.feedback}\n"
        f"Improve the answer to the question: {question}\n"
        "Provide feedback on your new answer, including a quality score from 1-10.\n"
        "Output as JSON with fields: answer, feedback, quality_score"
    )

final_answer = current_step.answer
print(f"Final answer after {i+1} iterations:")
print(final_answer)
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    Compiling FSM index for all state transitions: 100%|█| 70/70 [00:00<00:00, 
    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)


    Final answer after 1 iterations:
    Answer goes here.

