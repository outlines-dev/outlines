---
title: Decomposed Prompting (DECOMP)
---

# Decomposed Prompting (DECOMP)


Decomposed Prompting (DECOMP) is a technique that breaks down complex problems into simpler sub-problems using predefined functions. It works by first few-shot prompting a language model to demonstrate how to use specific functions (e.g., string splitting, internet searching). Then, given a complex problem, the model decomposes it into sub-problems and uses the appropriate functions to solve each part. This approach allows the model to tackle complex tasks by leveraging specialized functions and breaking down the problem into manageable pieces.

Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608). 

## A worked example


To implement Decomposed Prompting (DECOMP), follow these steps:

1. Define a set of functions that can help solve sub-problems. For example:
   - search_internet(query): Searches the internet for information
   - split_string(text, delimiter): Splits a string into parts
   - calculate_date_difference(date1, date2): Calculates the difference between two dates

2. Create few-shot examples showing the model how to use these functions. For example:

   Q: What is the capital of France?
   A: To answer this, I'll use the search_internet function.
   search_internet("capital of France")
   Result: Paris
   The capital of France is Paris.

   Q: Split the sentence "Hello, world!" into words.
   A: I'll use the split_string function for this.
   split_string("Hello, world!", " ")
   Result: ["Hello,", "world!"]
   The sentence is split into two words: "Hello," and "world!".

3. Present the complex problem to the model along with the few-shot examples and available functions.

4. Ask the model to decompose the problem and use the appropriate functions to solve it. For example:

   Q: How many days have passed between the Wright brothers' first flight and the Apollo 11 moon landing?

   A: To solve this problem, I'll break it down into steps and use our functions:

   1. Find the date of the Wright brothers' first flight:
      search_internet("date of Wright brothers first flight")
      Result: December 17, 1903

   2. Find the date of the Apollo 11 moon landing:
      search_internet("date of Apollo 11 moon landing")
      Result: July 20, 1969

   3. Calculate the difference between these dates:
      calculate_date_difference("December 17, 1903", "July 20, 1969")
      Result: 23,970 days

   Therefore, 23,970 days passed between the Wright brothers' first flight and the Apollo 11 moon landing.

5. The model will then execute this plan, using the defined functions to solve each sub-problem and combine the results to answer the original question.
    
## Code Example





```python
from typing import List
from pydantic import BaseModel
import outlines

# Placeholder functions for sub-problems
def search_internet(query: str) -> str:
    return f"Result for '{query}'"

def calculate_date_difference(date1: str, date2: str) -> int:
    return 23970  # Placeholder result

class Step(BaseModel):
    description: str
    function: str
    arguments: List[str]
    result: str

class Solution(BaseModel):
    steps: List[Step]
    final_answer: str

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.1", device="cuda")
generator = outlines.generate.json(model, Solution)

prompt = """
Decompose and solve the following problem:
How many days have passed between the Wright brothers' first flight and the Apollo 11 moon landing?

Available functions:
- search_internet(query): Searches the internet for information
- calculate_date_difference(date1, date2): Calculates the difference between two dates

Provide a step-by-step solution using these functions.
"""

solution = generator(prompt)
print(solution)
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    Compiling FSM index for all state transitions: 100%|█| 149/149 [00:01<00:00
    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)


    steps=[Step(description="Determine the dates of the Wright brothers' first flight and the Apollo 11 moon landing.", function='search_internet', arguments=['Wright brothers first flight date', 'Apollo 11 moon landing date'], result='Dec 17, 1903 & Jul 20, 1969'), Step(description='Parse the dates obtained and convert them into Python datetime objects.', function='parse_dates', arguments=['dates'], result='datetime(2003, 12, 17), datetime(1969, 7, 20)'), Step(description='Calculate the difference between the two dates.', function='calculate_date_difference', arguments=['date1', 'date2'], result='298 days')] final_answer="298 days have passed between the Wright brothers' first flight and the Apollo 11 moon landing."

