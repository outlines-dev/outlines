---
title: Prompt Mining
---

# Prompt Mining


Prompt Mining is a technique for discovering optimal prompt templates by analyzing large text corpora. The goal is to find "middle words" or phrasings that occur frequently in the corpus and are likely to elicit better performance from language models. Rather than using standard prompting formats like "Q: A:", Prompt Mining seeks to identify more natural phrasings that the model may have encountered more often during pre-training. This technique leverages the insight that prompt formats which appear more frequently in training data tend to yield improved results when used with language models.
    
Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).

## A worked example


To implement Prompt Mining:

1. Select a large, diverse text corpus representative of the language model's training data (e.g. web pages, books, articles).

2. Define the task type you want to optimize prompts for (e.g. question-answering, text completion).

3. Use natural language processing techniques to extract sentence pairs or segments that resemble the desired task format.

4. Analyze the extracted pairs to identify common patterns or "middle words" that frequently appear between the input and output.

5. Rank the discovered prompt templates based on their frequency in the corpus.

6. Test the top-ranking templates with your language model on a validation set to determine which ones perform best.

7. Use the best-performing mined prompt template for your task instead of a standard format.

For example, instead of using "Q: What is the capital of France? A:", prompt mining might discover that "Question: What is the capital of France? The answer is:" appears more frequently in the corpus and leads to better model performance.
    
## Code Example





```python
from enum import Enum
from pydantic import BaseModel, constr

import outlines

class MinedTemplate(str, Enum):
    STANDARD = "Q: {question} A:"
    MINED_1 = "Question: {question} The answer is:"
    MINED_2 = "Here's a question for you: {question} What do you think?"

class Answer(BaseModel):
    content: constr(max_length=100)

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.1", device="cuda")

def generate_answer(template: MinedTemplate, question: str):
    prompt = template.value.format(question=question)
    generator = outlines.generate.json(model, Answer)
    return generator(prompt)

question = "What is the capital of France?"

for template in MinedTemplate:
    answer = generate_answer(template, question)
    print(f"Template: {template.name}")
    print(f"Answer: {answer.content}\n")
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    Compiling FSM index for all state transitions: 100%|█| 619/619 [00:05<00:00
    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)


    Template: STANDARD
    Answer: Paris
    


    Compiling FSM index for all state transitions: 100%|█| 619/619 [00:05<00:00


    Template: MINED_1
    Answer: paris
    
    Template: MINED_2
    Answer: Answer: The capital of France is Paris. Do you think that is correct?
    

