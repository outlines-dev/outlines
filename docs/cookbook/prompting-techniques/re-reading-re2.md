---
title: Re-reading (RE2)
---

# Re-reading (RE2)


The Re-reading (RE2) technique is a simple yet effective prompting method that aims to improve an AI model's reasoning capabilities, especially for complex questions. It involves adding the phrase "Read the question again:" to the prompt, followed by repeating the original question. This approach encourages the model to process the question more thoroughly, potentially leading to more accurate and thoughtful responses.
    
Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).

## A worked example


To implement the Re-reading (RE2) technique:

1. Start with your original question or prompt.
2. Add the phrase "Read the question again:" after the original question.
3. Repeat the original question.
4. Submit the modified prompt to the AI model.

Here's a step-by-step example:

Original question: 
"What are the potential environmental impacts of electric vehicles?"

RE2 modified prompt:
"What are the potential environmental impacts of electric vehicles?

Read the question again:

What are the potential environmental impacts of electric vehicles?"

By using this technique, you encourage the AI model to reconsider the question, potentially leading to a more comprehensive and accurate response that addresses various aspects of the environmental impacts of electric vehicles.
    
## Code Example




```python
import outlines

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.1", device="cuda")

question = "What are the potential environmental impacts of electric vehicles?"
prompt = f"""You are an environmental science expert. Please answer the following question:

{question}

Read the question again:

{question}
"""

generator = outlines.generate.text(model)
answer: str = generator(prompt, max_tokens=300)

print(f"Question: {question}\n")
print(f"Answer: {answer}")
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)


    Question: What are the potential environmental impacts of electric vehicles?
    
    Answer: 
    Electric vehicles (EVs) offer numerous environmental benefits, including reduced air pollution and greenhouse gas emissions. However, they also come with certain environmental impacts, such as:
    
    1. Battery production: The manufacturing of batteries required for electric vehicles can have negative environmental impacts, such as deforestation, loss of biodiversity, and soil erosion. The production of batteries also comes with the release of hazardous materials and contaminants into the environment.
    
    2. Recycling: When electric vehicle batteries reach the end of their useful life, they need to be recycled to recover valuable materials such as lithium, cobalt, and nickel. However, the process of recycling EV batteries can be difficult and can have negative environmental impacts.
    
    3. Energy consumption: EVs require an increasing amount of electricity to charge, which may result in higher energy consumption in certain parts of the world.
    
    4. Land-use: The widespread adoption of EVs may require more land for charging stations and potentially battery production facilities.
    
    Overall, while EVs offer many environmental benefits, there are potential environmental impacts that need to be carefully considered and addressed.

