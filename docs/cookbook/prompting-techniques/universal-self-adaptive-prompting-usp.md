---
title: Universal Self-Adaptive Prompting (USP)
---

# Universal Self-Adaptive Prompting (USP)


Universal Self-Adaptive Prompting (USP) is an advanced prompting technique that aims to generate high-quality few-shot prompts for any given task without requiring labeled data. It builds upon the Consistency-based Self-adaptive Prompting (COSP) method but is designed to be more generalizable across different tasks. 

USP works by:
1. Using unlabeled data to generate potential exemplars
2. Employing a sophisticated scoring function to select the best exemplars
3. Constructing a few-shot prompt using the selected exemplars
4. Applying the generated prompt to the target task

Unlike COSP, USP does not rely on Self-Consistency for final output generation, making it more efficient for larger language models.
    

Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).

## A worked example


Here's a step-by-step guide to implement the USP technique:

1. Gather unlabeled data:
   Collect a set of unlabeled examples related to your target task. For instance, if your task is sentiment analysis, gather a variety of product reviews without labels.

2. Generate potential exemplars:
   Use a large language model to generate answers for each unlabeled example. For sentiment analysis, you might prompt the model to classify each review as positive, negative, or neutral.

3. Score and select exemplars:
   Implement a scoring function that evaluates the quality and diversity of the generated exemplars. This function should consider factors such as:
   - Consistency of the model's outputs across multiple runs
   - Diversity of the examples (to cover different aspects of the task)
   - Confidence of the model in its predictions
   Select the top-scoring exemplars to use in your few-shot prompt.

4. Construct the few-shot prompt:
   Create a prompt that includes:
   - A clear instruction for the task
   - The selected high-quality exemplars
   - A placeholder for the new input

5. Apply the prompt:
   Use the constructed few-shot prompt with your target language model to perform the desired task on new inputs.

Example for sentiment analysis:

Prompt:
"Classify the sentiment of the following product reviews as positive, negative, or neutral. Here are some examples:

Review: 'This phone exceeded my expectations. Great camera and long battery life!'
Sentiment: Positive

Review: 'The product arrived damaged and customer service was unhelpful.'
Sentiment: Negative

Review: 'It's an okay laptop. Nothing special but gets the job done.'
Sentiment: Neutral

Now, classify this review:
[Insert new review here]"

By using USP, you create a task-specific prompt that can adapt to various inputs without requiring manually labeled training data.
    
## Code Example





```python
from enum import Enum
from pydantic import BaseModel, Field
import outlines
from typing import List
import random

class Sentiment(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"

class Review(BaseModel):
    text: str = Field(..., max_length=200)
    sentiment: Sentiment

# Initialize the model
model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.1", device="cuda")

# Generate potential exemplars
generator = outlines.generate.json(model, Review)
exemplars = [
    generator("Generate a product review with sentiment:")
    for _ in range(10)
]

# Score exemplars (simulated for simplicity)
def score_exemplar(exemplar):
    return random.random()  # In a real scenario, implement a proper scoring function

scored_exemplars = [(ex, score_exemplar(ex)) for ex in exemplars]
best_exemplars = sorted(scored_exemplars, key=lambda x: x[1], reverse=True)[:3]

# Create few-shot prompt
few_shot_prompt = "Classify the sentiment of the following product reviews as positive, negative, or neutral. Here are some examples:\n\n"
for exemplar, _ in best_exemplars:
    few_shot_prompt += f"Review: '{exemplar.text}'\nSentiment: {exemplar.sentiment.value}\n\n"
few_shot_prompt += "Now, classify this review:\n{review}"

# Use the prompt for sentiment analysis
sentiment_generator = outlines.generate.choice(model, list(Sentiment))

new_review = "This product is amazing! I couldn't be happier with my purchase."
result = sentiment_generator(few_shot_prompt.format(review=new_review))

print(f"Review: {new_review}")
print(f"Sentiment: {result}")
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    Compiling FSM index for all state transitions: 100%|█| 1250/1250 [00:12<00:
    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)
    Compiling FSM index for all state transitions: 100%|█| 16/16 [00:00<00:00, 


    Review: This product is amazing! I couldn't be happier with my purchase.
    Sentiment: neutral

