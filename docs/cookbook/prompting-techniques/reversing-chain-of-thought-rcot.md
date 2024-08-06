---
title: Reversing Chain-of-Thought (RCoT)
---

# Reversing Chain-of-Thought (RCoT)


Reversing Chain-of-Thought (RCoT) is a self-criticism technique that aims to improve the accuracy of language model outputs. It works by first generating an answer, then having the model reconstruct the original problem based on that answer. The reconstructed problem is compared to the original to identify inconsistencies, which are then used as feedback for the model to revise its answer. This process helps catch logical errors or misunderstandings in the initial response.
    
Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).

## A worked example


1. Present the original problem to the language model:
   "What is the capital of France?"

2. Get the model's initial answer:
   "The capital of France is Paris."

3. Ask the model to reconstruct the original question based on its answer:
   "Based on the answer 'The capital of France is Paris', what was the original question?"

4. Compare the reconstructed question to the original:
   Original: "What is the capital of France?"
   Reconstructed: "What is the capital of France?"

5. In this case, there are no inconsistencies, so we would keep the original answer. 

6. If there were inconsistencies, we would prompt the model to revise its answer:
   "There seems to be an inconsistency between your answer and the original question. Please revise your answer to the question: What is the capital of France?"

7. Get the revised answer from the model.

8. Repeat steps 3-7 if necessary until the reconstructed question matches the original or a maximum number of iterations is reached.
    
## Code Example





```python
from typing import Optional
from pydantic import BaseModel
import outlines

class RCoTStep(BaseModel):
    original_question: str
    initial_answer: str
    reconstructed_question: str
    is_consistent: bool
    revised_answer: Optional[str] = None

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.1", device="cuda")
text_generator = outlines.generate.text(model)

def rcot_process(question: str, max_iterations: int = 3) -> RCoTStep:
    for _ in range(max_iterations):
        # Get initial answer
        initial_answer = text_generator(f"Answer this question: {question}")

        # Reconstruct the question
        reconstruct_prompt = f"Based on the answer '{initial_answer}', what was the original question?"
        reconstructed_question = text_generator(reconstruct_prompt)

        # Check consistency
        is_consistent = question.lower().strip() == reconstructed_question.lower().strip()

        step = RCoTStep(
            original_question=question,
            initial_answer=initial_answer,
            reconstructed_question=reconstructed_question,
            is_consistent=is_consistent
        )

        if is_consistent:
            return step

        # If inconsistent, revise the answer
        revision_prompt = f"There seems to be an inconsistency between your answer and the original question. Please revise your answer to the question: {question}"
        revised_answer = text_generator(revision_prompt)
        step.revised_answer = revised_answer

        # Update for next iteration
        question = step.original_question

    return step

# Example usage
result = rcot_process("What is the capital of France?")
print(result)
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)


    original_question='What is the capital of France?' initial_answer='\nAnswer: Paris' reconstructed_question='' is_consistent=False revised_answer='\n\nAnswer: The capital of France is Paris.'

