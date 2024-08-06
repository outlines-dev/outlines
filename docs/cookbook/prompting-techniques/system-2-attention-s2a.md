---
title: System 2 Attention (S2A)
---

# System 2 Attention (S2A)


System 2 Attention (S2A) is a two-step prompting technique that aims to improve the accuracy and focus of language model responses. In the first step, the model is asked to rewrite the original prompt, removing any information unrelated to the core question or task. This filtered prompt is then used in the second step to generate the final response. The goal is to help the model focus on the most relevant information by explicitly separating the process of understanding the prompt from generating the answer.
    
Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).

## A worked example


Step 1: Initial Prompt Refinement
- Start with the original prompt: "I'm planning a trip to Paris next month with my family. We love art and history. What are some must-see attractions, and can you recommend a good restaurant near the Eiffel Tower? Also, what's the weather like in Paris in June?"

- Ask the model to rewrite the prompt, focusing only on the core questions:
"Please rewrite the following prompt, removing any unnecessary information and focusing only on the key questions or tasks:
[Insert original prompt here]"

- The model might respond with:
"Refined prompt: What are some must-see attractions in Paris for art and history lovers? Can you recommend a good restaurant near the Eiffel Tower? What's the weather like in Paris in June?"

Step 2: Generate Response Using Refined Prompt
- Use the refined prompt to ask for the final response:
"Based on this refined prompt, please provide a detailed answer:
[Insert refined prompt here]"

- The model will then generate a response focused specifically on the key questions, without being distracted by extraneous information from the original prompt.

This two-step process helps ensure that the model's attention is directed to the most relevant aspects of the query, potentially leading to more accurate and focused responses.
    
## Code Example





```python
import outlines
from pydantic import BaseModel

class S2AOutput(BaseModel):
    refined_prompt: str
    response: str

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.1", device="cuda")

def system_2_attention(original_prompt: str) -> S2AOutput:
    refine_prompt = f"""Please rewrite the following prompt, removing any unnecessary information and focusing only on the key questions or tasks:
    {original_prompt}"""
    
    refined_prompt_generator = outlines.generate.text(model)
    refined_prompt = refined_prompt_generator(refine_prompt, max_tokens=100)
    
    response_prompt = f"""Based on this refined prompt, please provide a detailed answer:
    {refined_prompt}"""
    
    response_generator = outlines.generate.text(model)
    response = response_generator(response_prompt, max_tokens=200)
    
    return S2AOutput(refined_prompt=refined_prompt, response=response)

# Example usage
original_prompt = """I'm planning a trip to Paris next month with my family. We love art and history. 
What are some must-see attractions, and can you recommend a good restaurant near the Eiffel Tower? 
Also, what's the weather like in Paris in June?"""

result = system_2_attention(original_prompt)
print(f"Refined Prompt: {result.refined_prompt}")
print(f"\nResponse: {result.response}")
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)


    Refined Prompt: 
    
    Response: 
    Explain your approach to learning new languages.
           
    My approach to learning new languages involves a combination of immersion and structured study. Firstly, I believe in actively immersing oneself in the language, culture and environment of the target language. This includes spending time in the country where the language is spoken, listening to native speakers, watching movies and TV shows in the target language, and engaging in conversations with locals. These experiences help develop an ear for the language’s sound patterns, intonation, and rhythm, which are essential for effective communication.
    
    In addition to immersion, I also find structured study to be an effective way to learn a new language. This involves using textbooks, language learning apps, and online courses to learn the grammar, vocabulary, and syntax of the language. I also believe that practice is key, and regularly practicing speaking, writing, reading, and listening to the language helps to reinforce my understanding and improve my communication skills

