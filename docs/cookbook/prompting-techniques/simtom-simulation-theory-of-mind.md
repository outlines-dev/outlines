---
title: SimToM (Simulation Theory of Mind)
---

# SimToM (Simulation Theory of Mind)


SimToM is a prompting technique that leverages a language model's ability to simulate different perspectives or mental states. It involves instructing the model to imagine itself as a specific entity (e.g. a person, character, or even object) and respond from that perspective. This technique aims to generate more nuanced, context-aware responses by encouraging the model to consider the knowledge, biases, and thought processes of the simulated entity.
    
Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).

## A worked example


To implement the SimToM technique:

1. Define the entity or perspective you want the model to simulate. For example: "Imagine you are a 19th century British explorer."

2. Provide context about the entity's knowledge, experiences, or biases. For example: "You have just returned from a voyage to Africa and are writing in your journal."

3. Present the task or question from this perspective. For example: "Describe your thoughts on encountering a giraffe for the first time."

4. Optionally, include instructions on how to frame the response. For example: "Write your journal entry using language and terminology common in the 19th century."

5. Allow the model to generate a response from this simulated perspective.

Example prompt:
"Imagine you are a 19th century British explorer. You have just returned from a voyage to Africa and are writing in your journal. Describe your thoughts on encountering a giraffe for the first time. Write your journal entry using language and terminology common in the 19th century."

This technique encourages the model to consider the limited knowledge, cultural context, and writing style of a 19th century explorer, potentially resulting in a more authentic and era-appropriate response.
    
## Code Example





```python
from pydantic import BaseModel, constr
from datetime import date
import outlines

class JournalEntry(BaseModel):
    date: date
    location: constr(max_length=50)
    observation: constr(max_length=500)

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.1", device="cuda")
generator = outlines.generate.json(model, JournalEntry)

prompt = """Imagine you are a 19th century British explorer. 
You have just returned from a voyage to Africa and are writing in your journal. 
Describe your thoughts on encountering a giraffe for the first time. 
Write your journal entry using language and terminology common in the 19th century."""

entry = generator(prompt)
print(f"Date: {entry.date}")
print(f"Location: {entry.location}")
print(f"Observation: {entry.observation}")
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    Compiling FSM index for all state transitions: 100%|█| 3368/3368 [00:35<00:
    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)


    Date: 1793-09-05
    Location: Tanzania
    Observation: On my recent expedition through Africa, I was greeted with the most peculiar and remarkable creature I have ever encountered. It was a giraffe, a tall, lanky animal which stands over 6 feet tall when fully grown. Its neck is so long that I could not help but marvel at its peculiar shape and wonder if it could possibly turn its head to look around. Although its coat is not particularly soft to the touch, its curved horns and bright yellow eyes are quite striking. At first, I was quite intimidated

