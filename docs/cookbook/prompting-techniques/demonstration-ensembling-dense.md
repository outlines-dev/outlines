---
title: Demonstration Ensembling (DENSE)
---

# Demonstration Ensembling (DENSE)


Demonstration Ensembling (DENSE) is a prompting technique that improves the performance of few-shot learning by creating multiple prompts, each containing a different subset of examples from the training data. The technique then aggregates the outputs from these multiple prompts to generate a final response. This approach helps reduce variance in the model's outputs and often improves overall accuracy, though at the cost of increased computation due to multiple model calls.
    
Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).

## A worked example


1. Prepare your dataset:
   Assume you have a sentiment analysis task with a training set of 100 labeled examples.

2. Create multiple prompts:
   Generate 5 different prompts, each containing a random subset of 3 examples from your training set.

   Prompt 1:
   "Classify the sentiment of the following movie review as positive or negative:
   
   Examples:
   - 'This film was a waste of time.' Sentiment: Negative
   - 'I couldn't stop laughing throughout the movie!' Sentiment: Positive
   - 'The acting was superb and the plot kept me engaged.' Sentiment: Positive
   
   Review to classify: [INSERT TEST REVIEW HERE]"

   Prompt 2-5: Similar structure, but with different randomly selected examples.

3. Submit prompts to the language model:
   Send each of the 5 prompts to the language model, inserting the same test review into each prompt.

4. Collect outputs:
   Record the sentiment classification (positive or negative) from each of the 5 prompts.

5. Aggregate results:
   Use a majority voting system to determine the final classification. For example:
   - If 3 or more prompts classify the review as positive, the final output is positive.
   - If 3 or more prompts classify the review as negative, the final output is negative.

6. Return final classification:
   Provide the aggregated sentiment classification as the final output of the DENSE technique.
    
## Code Example





```python
from pydantic import BaseModel
from typing import List
import random
from collections import Counter
import outlines

class SentimentExample(BaseModel):
    review: str
    sentiment: str

class SentimentAnalysis(BaseModel):
    examples: List[SentimentExample]
    review_to_classify: str

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.1", device="cuda")

def generate_prompt(examples: List[SentimentExample], review_to_classify: str) -> str:
    examples_text = "\n".join([f"Review: {e.review}\nSentiment: {e.sentiment}" for e in examples])
    return f"""Classify the sentiment of the following movie review as positive or negative:

Examples:
{examples_text}

Review to classify: {review_to_classify}
Sentiment:"""

def dense_sentiment_analysis(data: SentimentAnalysis, num_prompts: int = 5, examples_per_prompt: int = 3) -> str:
    generator = outlines.generate.choice(model, ["Positive", "Negative"])
    results = []

    for _ in range(num_prompts):
        prompt_examples = random.sample(data.examples, examples_per_prompt)
        prompt = generate_prompt(prompt_examples, data.review_to_classify)
        result = generator(prompt)
        results.append(result)

    final_sentiment = Counter(results).most_common(1)[0][0]
    return final_sentiment

# Example usage
sentiment_data = SentimentAnalysis(
    examples=[
        SentimentExample(review="This film was a waste of time.", sentiment="Negative"),
        SentimentExample(review="I couldn't stop laughing throughout the movie!", sentiment="Positive"),
        SentimentExample(review="The acting was superb and the plot kept me engaged.", sentiment="Positive"),
        SentimentExample(review="I fell asleep halfway through.", sentiment="Negative"),
        SentimentExample(review="A masterpiece of modern cinema.", sentiment="Positive"),
    ],
    review_to_classify="The special effects were amazing, but the story was confusing."
)

result = dense_sentiment_analysis(sentiment_data)
print(f"Final sentiment: {result}")
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    Compiling FSM index for all state transitions: 100%|█| 12/12 [00:00<00:00, 
    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)


    Final sentiment: Negative

