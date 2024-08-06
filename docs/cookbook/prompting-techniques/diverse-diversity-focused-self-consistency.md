---
title: DiVeRSe (Diversity-Focused Self-Consistency)
---

# DiVeRSe (Diversity-Focused Self-Consistency)


DiVeRSe is an ensemble prompting technique that combines multiple prompts with self-consistency to generate diverse reasoning paths and improve answer quality. It works by:
1. Creating multiple different prompts for a given problem
2. Applying self-consistency (generating multiple outputs) for each prompt
3. Aggregating the results across all prompts and reasoning paths to select the final answer
This approach aims to increase diversity in the reasoning process and reduce errors by considering multiple perspectives on the problem.
    
Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).

## A worked example


To implement DiVeRSe for a math word problem:

1. Create multiple prompts:
   Prompt 1: "Solve this step-by-step:"
   Prompt 2: "Explain your reasoning as you solve this problem:"
   Prompt 3: "What information do we need to solve this? Then solve it:"

2. Apply self-consistency to each prompt (generate 3 outputs per prompt):
   For Prompt 1:
     Output 1A: [reasoning path 1]
     Output 1B: [reasoning path 2]
     Output 1C: [reasoning path 3]
   
   For Prompt 2:
     Output 2A: [reasoning path 1]
     Output 2B: [reasoning path 2]
     Output 2C: [reasoning path 3]
   
   For Prompt 3:
     Output 3A: [reasoning path 1]
     Output 3B: [reasoning path 2]
     Output 3C: [reasoning path 3]

3. Aggregate results:
   - Collect all final answers from the 9 outputs
   - Select the most common answer as the final result
   - If there's a tie, use a predefined method to break it (e.g., confidence scoring)

4. Return the selected answer as the final output of the DiVeRSe technique
    
## Code Example





```python
from pydantic import BaseModel, Field
from typing import List
import outlines
from collections import Counter

class MathSolution(BaseModel):
    reasoning: str
    answer: float

class DiVeRSeSolution(BaseModel):
    prompt: str
    solutions: List[MathSolution] = Field(min_items=3, max_items=3)

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.1", device="cuda")
generator = outlines.generate.json(model, DiVeRSeSolution)

problem = "If a train travels 120 miles in 2 hours, what is its average speed in miles per hour?"

prompts = [
    f"Solve this step-by-step: {problem}",
    f"Explain your reasoning as you solve this problem: {problem}",
    f"What information do we need to solve this? Then solve it: {problem}"
]

diverse_solutions = []

for prompt in prompts:
    solutions = generator(prompt)
    diverse_solutions.append(solutions)

# Aggregate results
all_answers = [solution.answer for solutions in diverse_solutions for solution in solutions.solutions]
final_answer = Counter(all_answers).most_common(1)[0][0]

print(f"Final answer: {final_answer} mph")

# Print all reasoning paths
for i, solutions in enumerate(diverse_solutions):
    print(f"\nPrompt {i+1}: {solutions.prompt}")
    for j, solution in enumerate(solutions.solutions):
        print(f"  Solution {j+1}:")
        print(f"    Reasoning: {solution.reasoning}")
        print(f"    Answer: {solution.answer}")
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    Compiling FSM index for all state transitions: 100%|█| 189/189 [00:01<00:00
    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)


    Final answer: 60.0 mph
    
    Prompt 1: If a train travels 120 miles in 2 hours, what is its average speed in miles per hour?
      Solution 1:
        Reasoning: Step 1: Determine the speed of the train in miles per hour by dividing the distance traveled by the time taken. This can be done using the formula: Speed = Distance ÷ Time. Plugging in the values, we get: Speed = 120 miles ÷ 2 hours = 60 miles/hour.
        Answer: 60.0
      Solution 2:
        Reasoning: The question asks for the average speed. An average can be found by taking the total distance traveled and dividing it by the total time taken. Since the train traveled 120 miles and the time taken was 2 hours, the average speed is: Average Speed = Total Distance ÷ Total Time = 120 miles ÷ 2 hours = 60 miles/hour.
        Answer: 60.0
      Solution 3:
        Reasoning: Another way to solve this question is by finding the rate of speed. The formula for rate of speed is: Rate = Distance ÷ Time. Plugging in the values, we get: Rate = 120 miles ÷ 2 hours = 60 miles/hour. Since the train traveled 120 miles and the time taken was 2 hours, the rate of speed is also 60 miles/hour. However, since we are looking for the average speed, we should use the second method.
        Answer: 60.0
    
    Prompt 2: Calculate the average speed of the train
      Solution 1:
        Reasoning: To calculate the average speed of the train, we need to use the formula: speed = distance/time. In this case, the distance traveled is 120 miles, and the time taken is 2 hours. Plugging these values into the formula, we get: speed = 120 miles / 2 hours = 60 miles/hour.
        Answer: 60.0
      Solution 2:
        Reasoning: Another way to think about this problem is by visualizing the train's movement. Imagine the train is traveling on a straight, flat track with two bicycles next to it. The first bicycle represents speed (because it measures the rate at which the bicycle is moving), and the second bicycle represents distance (because it measures the distance between two points). The train will be moving at a constant speed and distance, meaning it will take the same amount of time to travel the same distance. So if we measure the distance traveled by the train we can determine its speed by dividing the distance by the time taken. In this case, the distance taken is 120 miles and the time taken is 2 hours. Therefore, the train's average speed is 60 miles/hour.
        Answer: 60.0
      Solution 3:
        Reasoning: If we were to divide this problem into smaller parts, we could determine the average speed of the squares within these parts, but then adding up all these distances would not give us the total distance traveled by the train.
        Answer: 60.0
    
    Prompt 3: Solve the given problem. If a train travels 120 miles in 2 hours, what is its average speed in miles per hour?
      Solution 1:
        Reasoning: To calculate the average speed of the train, we need to divide the total distance traveled by the time it took to travel that distance. ",  
        Answer: 60.0
      Solution 2:
        Reasoning: Average speed is defined as the total distance traveled divided by the time taken to travel that distance. Therefore, the average speed of the train is 120/2 = 60 miles per hour. ",  
        Answer: 60.0
      Solution 3:
        Reasoning: We have been given that the train covers a distance of 120 in 2 hours. To find the average speed, we just need to divide the distance by time. Thus the average speed of train is 120 miles divided by 2 hours which is equal to 60 miles per hour.
        Answer: 60.0

