---
title: Dialogue-comprised Policy-gradient-based Discrete Prompt Optimization (DP2O)
---

# Dialogue-comprised Policy-gradient-based Discrete Prompt Optimization (DP2O)


DP2O is an advanced prompt engineering technique that uses reinforcement learning, a custom prompt scoring function, and conversations with a language model to construct optimal prompts. It involves:

1. Initializing a prompt template
2. Using the template to generate outputs from a language model
3. Scoring the outputs with a custom scoring function
4. Engaging in a dialogue with the language model to suggest improvements to the prompt
5. Updating the prompt using policy gradient reinforcement learning
6. Iterating this process to optimize the prompt
    
Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).

## A worked example


To implement DP2O:

1. Start with an initial prompt template, e.g. "Classify the sentiment of this text: {input}"

2. Generate outputs for a set of inputs using this prompt

3. Score the outputs using a custom function (e.g. accuracy compared to ground truth labels)

4. Have a "dialogue" with the LM to improve the prompt:
   LM: "The current prompt could be improved by specifying sentiment categories."
   Human: "How would you rewrite the prompt to include that?"
   LM: "Classify the sentiment of this text as positive, negative, or neutral: {input}"

5. Update the prompt template using the LM's suggestion and policy gradient RL

6. Repeat steps 2-5 for several iterations, tracking the best performing prompt

7. Use the final optimized prompt for the task
    
## Code Example





```python
from pydantic import BaseModel, Field
import outlines

class PromptTemplate(BaseModel):
    template: str
    score: float = Field(default=0.0)

# Initialize model
model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.1", device="cuda")

# Initial prompt template
prompt = PromptTemplate(template="Classify the sentiment of this text: {input}")

# Simulated function to score outputs
def score_outputs(outputs):
    return 0.7  # Simulated score

# Function to generate sentiment classifications
def classify_sentiments(template, inputs):
    generator = outlines.generate.choice(model, ["Positive", "Negative", "Neutral"])
    return [generator(template.format(input=input)) for input in inputs]

# Function to improve prompt through dialogue
def improve_prompt(current_prompt):
    dialogue_prompt = f"The current prompt is: '{current_prompt}'. How can we improve it for sentiment analysis?"
    suggestion = outlines.generate.text(model)(dialogue_prompt, max_tokens=50)
    return suggestion

# DP2O process
for iteration in range(3):  # Perform 3 iterations
    # Generate outputs
    inputs = ["I love this product!", "This is terrible.", "It's okay."]
    outputs = classify_sentiments(prompt.template, inputs)
    
    # Score outputs
    score = score_outputs(outputs)
    prompt.score = score
    
    # Improve prompt
    suggestion = improve_prompt(prompt.template)
    
    # Update prompt (simulating policy gradient update)
    prompt.template = suggestion

    print(f"Iteration {iteration + 1}:")
    print(f"Prompt: {prompt.template}")
    print(f"Score: {prompt.score}\n")

# Final optimized prompt
print("Final optimized prompt:", prompt.template)
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    Compiling FSM index for all state transitions: 100%|█| 16/16 [00:00<00:00, 
    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)


    Iteration 1:
    Prompt: 
    
    The current prompt is quite straightforward and can be used for basic sentiment analysis. However, there are a few ways to improve it: 
    
    1. Provide more information: The prompt could be improved by providing more information about the
    Score: 0.7
    


    Compiling FSM index for all state transitions: 100%|█| 16/16 [00:00<00:00, 


    Iteration 2:
    Prompt: 
    Score: 0.7
    
    Iteration 3:
    Prompt: 
    Let me know.
    Score: 0.7
    
    Final optimized prompt: 
    Let me know.

