---
title: Zero-Shot Prompting
---

# Zero-Shot Prompting


Zero-Shot Prompting is a technique where you provide instructions to a language model without giving it any examples to learn from. The model is expected to complete the task based solely on the instructions and its pre-existing knowledge. This approach relies on the model's ability to understand and execute instructions without specific training examples.
    
Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).

## A worked example


To implement Zero-Shot Prompting:

1. Identify the task you want the model to perform.
2. Craft clear and concise instructions that explain the task.
3. Provide any necessary context or background information.
4. Ask the model to complete the task.

For example, if you want the model to classify a movie review sentiment:

1. Task: Sentiment analysis of a movie review
2. Instructions: "Determine whether the following movie review is positive or negative. Respond with either 'Positive' or 'Negative'."
3. Context: Provide the movie review text
4. Task completion request: "Based on this review, is the sentiment positive or negative?"

Full prompt:
"Determine whether the following movie review is positive or negative. Respond with either 'Positive' or 'Negative'.

Review: 'The special effects were impressive, but the plot was confusing and the characters were poorly developed. I wouldn't recommend this movie.'

Based on this review, is the sentiment positive or negative?"

The model would then analyze the review and respond with "Negative" based on the instructions and context provided, without having seen any labeled examples.
    
## Code Example





```python
import outlines

# Set up the model
model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.1", device="cuda")

# Create the prompt with instructions and context
prompt = """Determine whether the following movie review is positive or negative. Respond with either 'Positive' or 'Negative'.

Review: 'The special effects were impressive, but the plot was confusing and the characters were poorly developed. I wouldn't recommend this movie.'

Based on this review, is the sentiment positive or negative?"""

# Generate the sentiment using zero-shot prompting
generator = outlines.generate.choice(model, ["Positive", "Negative"])
sentiment = generator(prompt)

print(f"The sentiment of the movie review is: {sentiment}")
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)


    The sentiment of the movie review is: Negative

