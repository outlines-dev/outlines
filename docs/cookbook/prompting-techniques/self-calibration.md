---
title: Self-Calibration
---

# Self-Calibration


Self-Calibration is a prompting technique that aims to improve the reliability of language model outputs by having the model assess its own confidence in its answers. It works in two main steps:

1. The model is first prompted to answer a given question.
2. A new prompt is then constructed that includes the original question, the model's answer, and an additional instruction asking the model to evaluate whether its answer is correct.

This technique allows for gauging the model's confidence level in its responses, which can be useful when deciding whether to accept the original answer or seek further refinement.
    
Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).

## A worked example


Here's a step-by-step implementation of the Self-Calibration technique:

1. Initial Prompt:
   "What is the capital of France?"

2. Model Response:
   "The capital of France is Paris."

3. Self-Calibration Prompt:
   "Question: What is the capital of France?
    Your answer: The capital of France is Paris.
    
    Is the above answer correct? Please respond with 'Yes' if you are certain the answer is correct, 'No' if you are certain it is incorrect, or 'Uncertain' if you are not sure. Explain your reasoning."

4. Model Self-Calibration Response:
   "Yes, I am certain the answer is correct. Paris is indeed the capital city of France. It has been the capital for centuries and is widely recognized as the political, cultural, and economic center of the country. This is a well-established fact in geography and global knowledge."

5. Interpretation:
   Based on the model's confident affirmation, we can consider the original answer to be highly reliable. If the model had expressed uncertainty or stated the answer was incorrect, it would signal a need for further verification or refinement of the response.

This Self-Calibration step provides valuable insight into the model's confidence level, helping to determine when to trust its outputs and when additional verification might be necessary.
    
## Code Example





```python
from enum import Enum
from pydantic import BaseModel
import outlines

class ConfidenceLevel(str, Enum):
    YES = "Yes"
    NO = "No"
    UNCERTAIN = "Uncertain"

class SelfCalibrationResponse(BaseModel):
    initial_answer: str
    confidence: ConfidenceLevel
    explanation: str

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.1", device="cuda")
generator = outlines.generate.json(model, SelfCalibrationResponse)

question = "What is the capital of France?"
initial_prompt = f"Answer the following question: {question}"
initial_answer = outlines.generate.text(model)(initial_prompt)

self_calibration_prompt = f"""
Question: {question}
Your answer: {initial_answer}

Is the above answer correct? Please respond with 'Yes' if you are certain the answer is correct, 'No' if you are certain it is incorrect, or 'Uncertain' if you are not sure. Explain your reasoning.
"""

response = generator(self_calibration_prompt)
print(response)
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    Compiling FSM index for all state transitions: 100%|█| 89/89 [00:00<00:00, 
    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)


    initial_answer='Yes' confidence=<ConfidenceLevel.UNCERTAIN: 'Uncertain'> explanation="The capital of France is commonly known as Paris, but there is a possibility that it could be any other major city in the country. The best response would be 'Uncertain', as there is a chance that the capital of France could be a city such as Lyon, Marseille, or Toulouse. However, based on common knowledge and the most recent information available, the capital of France most likely is Paris."

