---
title: AutoPrompt
---

# AutoPrompt


AutoPrompt is a prompt engineering technique that uses a frozen language model and a prompt template with "trigger tokens" whose values are updated via backpropagation during training. It is a form of soft prompting that aims to automatically optimize the prompt to improve task performance. The trigger tokens are initialized randomly and then iteratively updated to maximize the model's performance on a given task.

Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).
   
## A worked example


To implement AutoPrompt:

1. Define a task and dataset (e.g. sentiment classification).

2. Create a prompt template with placeholders for trigger tokens:
   "[T1] [T2] [T3] Classify the sentiment: {input_text}"

3. Initialize trigger tokens ([T1], [T2], [T3]) with random words.

4. For each training iteration:
   a. Sample a batch of examples from the dataset.
   b. Pass the examples through the prompt template and model.
   c. Calculate the loss based on model outputs and true labels.
   d. Backpropagate the loss to update only the trigger token embeddings.
   e. Project updated embeddings to nearest word embeddings in vocabulary.

5. After training, use the optimized trigger tokens in the final prompt:
   "great positive amazing Classify the sentiment: {input_text}"

6. Evaluate the AutoPrompt on a test set to measure performance improvement.
    
## Code Example





```python
from enum import Enum
import outlines

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.1", device="cuda")

# Simulated "optimized" prompt with trigger tokens
prompt_template = """great positive amazing
Classify the sentiment of the following text:
Text: {input_text}
Sentiment:"""

generator = outlines.generate.choice(model, list(Sentiment))

input_text = "This movie was absolutely fantastic!"
prompt = prompt_template.format(input_text=input_text)

sentiment = generator(prompt)
print(f"Input: {input_text}")
print(f"Detected sentiment: {sentiment}")
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)


    Input: This movie was absolutely fantastic!
    Detected sentiment: positive

