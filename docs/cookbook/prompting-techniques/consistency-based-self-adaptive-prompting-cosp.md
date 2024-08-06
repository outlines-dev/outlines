---
title: Consistency-based Self-adaptive Prompting (COSP)
---

# Consistency-based Self-adaptive Prompting (COSP)


COSP is an advanced prompting technique that automatically constructs few-shot Chain-of-Thought (CoT) prompts by leveraging zero-shot CoT and self-consistency. It works by first applying zero-shot CoT with self-consistency on a set of example problems to generate multiple reasoning paths. It then selects a subset of these examples with high agreement among the generated paths to use as exemplars in the final few-shot CoT prompt. This final prompt is then used with self-consistency again to produce the ultimate output.
    
Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).
## A worked example


1. Prepare a set of example problems related to your target task.

2. For each example problem:
   a. Apply zero-shot CoT prompting to generate multiple reasoning paths (e.g. 5-10 paths).
   b. Use self-consistency to select the most common answer among these paths.

3. For each example, calculate the agreement level among its generated reasoning paths.

4. Select a subset of examples (e.g. 3-5) with the highest agreement levels.

5. Construct a few-shot CoT prompt using the selected examples and their most consistent reasoning paths as exemplars.

6. Use this constructed prompt with self-consistency on your target problem:
   a. Generate multiple reasoning paths for the target problem.
   b. Select the most common answer as the final output.

Example implementation:

Problem: "If 5 apples cost $2, how much do 8 apples cost?"

Step 1-4: [Assume we've gone through the process and selected high-agreement examples]

Step 5: Construct the prompt:
"Q: If 12 oranges cost $3, how much do 5 oranges cost?
A: Let's approach this step-by-step:
1. First, let's find the cost of one orange.
2. If 12 oranges cost $3, then we can divide $3 by 12.
3. $3 ÷ 12 = $0.25 per orange
4. Now, we need to find the cost of 5 oranges.
5. We multiply the cost per orange by 5.
6. $0.25 × 5 = $1.25
Therefore, 5 oranges would cost $1.25.

Q: If 5 apples cost $2, how much do 8 apples cost?
A: Let's solve this step-by-step:"

Step 6: Generate multiple reasoning paths using this prompt and select the most common answer:

Path 1:
1. First, let's find the cost of one apple.
2. If 5 apples cost $2, then we can divide $2 by 5.
3. $2 ÷ 5 = $0.40 per apple
4. Now, we need to find the cost of 8 apples.
5. We multiply the cost per apple by 8.
6. $0.40 × 8 = $3.20
Therefore, 8 apples would cost $3.20.

[Generate more paths and select the most common answer]

Final output: 8 apples would cost $3.20.
    
## Code Example






```python
from pydantic import BaseModel, Field
from typing import List
import outlines
from collections import Counter

class ReasoningPath(BaseModel):
    steps: List[str] = Field(..., description="Steps in the reasoning process")
    answer: float = Field(..., description="Final numerical answer")

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.1", device="cuda")

prompt = """Q: If 12 oranges cost $3, how much do 5 oranges cost?
A: Let's approach this step-by-step:
1. First, let's find the cost of one orange.
2. If 12 oranges cost $3, then we can divide $3 by 12.
3. $3 ÷ 12 = $0.25 per orange
4. Now, we need to find the cost of 5 oranges.
5. We multiply the cost per orange by 5.
6. $0.25 × 5 = $1.25
Therefore, 5 oranges would cost $1.25.

Q: If 5 apples cost $2, how much do 8 apples cost?
A: Let's solve this step-by-step:"""

generator = outlines.generate.json(model, ReasoningPath)

paths = []
for _ in range(5):  # Generate 5 reasoning paths
    path = generator(prompt)
    paths.append(path)

# Select the most common answer
answers = [path.answer for path in paths]
most_common_answer = Counter(answers).most_common(1)[0][0]

print(f"The most common answer is: ${most_common_answer:.2f}")
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)


    The most common answer is: $3.00

