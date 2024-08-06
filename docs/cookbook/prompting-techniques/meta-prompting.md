---
title: Meta Prompting
---

# Meta Prompting


Meta prompting is a technique where you prompt a language model to generate or improve prompts for itself or other language models. This involves asking the model to create, refine, or analyze prompts based on given criteria or goals. The technique leverages the model's understanding of effective prompting to generate more sophisticated or targeted prompts.
    
Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).

## A worked example


To implement meta prompting:

1. Define your ultimate task or goal (e.g., generating a marketing slogan for a new product).

2. Craft an initial meta prompt asking the model to generate a prompt for that task. For example:
   "Create an effective prompt that will help a language model generate a catchy marketing slogan for a new eco-friendly water bottle."

3. Use the model's response as your new prompt.

4. If needed, refine further by asking the model to improve the generated prompt:
   "Please improve the following prompt to make it more specific and effective: [Insert previously generated prompt]"

5. Once satisfied with the meta-generated prompt, use it to accomplish your original task.

6. Evaluate the results and repeat the process if necessary, asking the model to further refine the prompt based on the output.

This iterative process allows you to leverage the model's capabilities to create increasingly effective prompts tailored to your specific needs.
    
## Code Example





```python
import outlines

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.1", device="cuda")

# Generate a meta prompt
meta_prompt = "Create an effective prompt that will help a language model generate a catchy marketing slogan for a new eco-friendly water bottle."
meta_generator = outlines.generate.text(model)
generated_prompt = meta_generator(meta_prompt, max_tokens=100)

# Use the generated prompt to create the final slogan
slogan_generator = outlines.generate.text(model)
slogan = slogan_generator(generated_prompt, max_tokens=50)

print("Generated Meta Prompt:", generated_prompt)
print("Final Slogan:", slogan)
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)


    Generated Meta Prompt: 
    
    "Think outside the plastic and...go green with our eco-friendly water bottle!"
    Final Slogan: 
    
    We strive to make it easy for our customers to keep hydrated while helping the planet! Our eco-friendly water bottles are made from sustainable materials, BPA and chemical free, and designed to be reused over and over

