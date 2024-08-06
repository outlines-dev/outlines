---
title: Max Mutual Information Method
---

# Max Mutual Information Method


The Max Mutual Information Method is an ensembling technique that aims to find the optimal prompt template from a set of diverse options. It works by:

1. Creating multiple prompt templates with varied styles and exemplars for a given task.
2. Generating outputs from the language model using each template.
3. Calculating the mutual information between each prompt template and the corresponding outputs.
4. Selecting the template that maximizes the mutual information as the optimal prompt.

This method helps identify the most effective prompt by measuring how much information is shared between the prompt and the model's responses, potentially leading to more relevant and consistent outputs.

Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).

## A worked example


Let's implement the Max Mutual Information Method for a sentiment analysis task:

1. Create diverse prompt templates:
   Template A: "Analyze the sentiment of this text: [INPUT]. Is it positive or negative?"
   Template B: "Rate the following on a scale from very negative to very positive: [INPUT]"
   Template C: "What emotions does this text convey? [INPUT]"

2. Generate outputs for a set of input texts using each template.

3. Calculate mutual information:
   - For each template, measure how well the outputs align with expected sentiments.
   - Quantify the consistency and relevance of responses.

4. Select the optimal template:
   Suppose Template B yields the highest mutual information score.

5. Use the selected template for future inputs:
   "Rate the following on a scale from very negative to very positive: [NEW INPUT]"

By using this method, you identify the prompt template that consistently produces the most informative and relevant responses for your specific task.
    
## Code Example





```python
import outlines
from typing import List, Tuple
import numpy as np

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.1", device="cuda")

# Define prompt templates
templates = [
    "Analyze the sentiment of this text: [INPUT]. Is it positive or negative?",
    "Rate the following on a scale from very negative to very positive: [INPUT]",
    "What emotions does this text convey? [INPUT]"
]

# Function to calculate basic mutual information score
def calculate_mi_score(outputs: List[str]) -> float:
    unique_outputs = set(outputs)
    scores = [outputs.count(output) / len(outputs) for output in unique_outputs]
    return -sum(score * np.log2(score) for score in scores)

# Sample input texts
input_texts = [
    "I love this product!",
    "This is terrible.",
    "It's okay, I guess.",
    "Absolutely amazing experience!"
]

# Generate outputs and calculate MI scores for each template
template_scores: List[Tuple[str, float]] = []
for template in templates:
    generator = outlines.generate.choice(model, ["Positive", "Negative", "Neutral"])
    outputs = [generator(template.replace("[INPUT]", text)) for text in input_texts]
    mi_score = calculate_mi_score(outputs)
    template_scores.append((template, mi_score))

# Select the best template
best_template = max(template_scores, key=lambda x: x[1])[0]

# Use the best template for a new input
new_input = "This product exceeded all my expectations!"
generator = outlines.generate.choice(model, ["Positive", "Negative", "Neutral"])
result = generator(best_template.replace("[INPUT]", new_input))

print(f"Best template: {best_template}")
print(f"Sentiment for new input: {result}")
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)


    Best template: Analyze the sentiment of this text: [INPUT]. Is it positive or negative?
    Sentiment for new input: Positive

