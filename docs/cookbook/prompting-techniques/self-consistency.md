---
title: Self-Consistency
---

# Self-Consistency


Self-Consistency is an ensemble prompting technique that aims to improve accuracy by generating multiple reasoning paths for a given problem and then selecting the most consistent answer. It works as follows:

1. Given a problem, prompt the language model multiple times to perform Chain-of-Thought (CoT) reasoning, using a non-zero temperature setting to encourage diverse reasoning paths.

2. Collect all the generated responses and final answers.

3. Use a majority vote over all the generated final answers to select the most consistent response as the output.

This technique leverages the idea that multiple valid reasoning paths can lead to the same correct answer, while incorrect reasoning is more likely to produce inconsistent results. By aggregating multiple attempts, Self-Consistency can often achieve higher accuracy than a single CoT prompt, especially on tasks involving arithmetic, commonsense, and symbolic reasoning.
    
Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).

## A worked example


Let's implement Self-Consistency for a simple math problem:

Problem: "If a train travels 120 miles in 2 hours, what is its average speed in miles per hour?"

Step 1: Generate multiple CoT responses (let's do 3 for this example)
- Prompt the model 3 times with:
"Solve this step-by-step: If a train travels 120 miles in 2 hours, what is its average speed in miles per hour?"

Response 1:
1. We know the train traveled 120 miles in total.
2. The time taken is 2 hours.
3. Average speed = Total distance / Total time
4. Average speed = 120 miles / 2 hours
5. Average speed = 60 miles per hour
Therefore, the train's average speed is 60 mph.

Response 2:
1. To find average speed, we need to divide distance by time.
2. Distance = 120 miles
3. Time = 2 hours
4. Speed = Distance ÷ Time
5. Speed = 120 ÷ 2
6. Speed = 60 miles per hour
The average speed of the train is 60 mph.

Response 3:
1. Let's use the formula: Speed = Distance / Time
2. We have: Distance = 120 miles, Time = 2 hours
3. Plugging in the values: Speed = 120 / 2
4. Calculating: Speed = 60
So, the average speed is 60 miles per hour.

Step 2: Collect final answers
All three responses conclude with 60 miles per hour.

Step 3: Perform majority vote
Since all responses agree on 60 miles per hour, this is selected as the final answer.

Output: The train's average speed is 60 miles per hour.

In this case, all reasoning paths led to the same correct answer, demonstrating high self-consistency. In more complex problems, different paths might lead to different answers, and the most frequent answer would be selected.
    
## Code Example





```python
import outlines
from collections import Counter

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.1", device="cuda")

def generate_cot_response(prompt: str) -> str:
    generator = outlines.generate.text(model)
    return generator(prompt, max_tokens=200)

def extract_final_answer(response: str) -> float:
    # Assuming the final answer is the last number in the response
    numbers = [float(s) for s in response.split() if s.replace('.', '').isdigit()]
    return numbers[-1] if numbers else None

def self_consistency(question: str, num_samples: int = 3) -> float:
    prompt = f"Solve this step-by-step: {question}"
    responses = [generate_cot_response(prompt) for _ in range(num_samples)]
    
    final_answers = [extract_final_answer(response) for response in responses]
    answer_counts = Counter(final_answers)
    
    most_common_answer, _ = answer_counts.most_common(1)[0]
    return most_common_answer

# Example usage
question = "If a train travels 120 miles in 2 hours, what is its average speed in miles per hour?"
result = self_consistency(question)
print(f"The most consistent answer is: {result} mph")
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)


    The most consistent answer is: 60.0 mph

