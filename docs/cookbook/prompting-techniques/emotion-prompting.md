---
title: Emotion Prompting
---

# Emotion Prompting


Emotion Prompting incorporates phrases or statements with psychological and emotional relevance into the prompt to potentially improve the language model's performance. This technique taps into the model's understanding of human emotions and motivations, which may lead to more thoughtful or nuanced responses. By framing the task or question in a way that emphasizes its personal or emotional importance, the model may engage more deeply with the prompt and produce higher quality outputs.
    
Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).

## A worked example


To implement Emotion Prompting:

1. Identify the core task or question you want the model to address.

2. Consider the emotional or psychological context that would make this task important to a human.

3. Craft a phrase that expresses this emotional context.

4. Incorporate this emotional phrase into your prompt, typically at the beginning.

5. Follow with your specific task or question.

For example, if you want the model to proofread an important email, you might use this prompt:

"This email is crucial for my career advancement. Please help me proofread and improve the following draft email to my boss:

[Your draft email text here]

Please provide suggestions for improvements and explain why each change would be beneficial."

In this example, "This email is crucial for my career advancement" serves as the emotional prompt, potentially encouraging the model to approach the task with greater care and attention to detail.
    
## Code Example





```python
from enum import Enum
from pydantic import BaseModel, constr, conint

import outlines

class Armor(str, Enum):
    leather = "leather"
    chainmail = "chainmail"
    plate = "plate"

class Character(BaseModel):
    name: constr(max_length=10)
    age: conint(gt=18, lt=99)
    armor: Armor
    strength: conint(gt=1, lt=100)

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.1", device="cuda")
generator = outlines.generate.json(model, Character)

emotion_prompt = (
    "As a passionate game master, I'm deeply invested in creating memorable characters "
    "that will resonate with my players and bring our story to life. "
    "This next character is crucial for an upcoming plot twist. "
    "Please create a character that will leave a lasting impression: "
)

character = generator(
    emotion_prompt
    + "Generate a new character with a name (max 10 characters), "
    + "age (between 19 and 98), armor type, and strength (between 2 and 99)."
)
print(character)
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    Compiling FSM index for all state transitions: 100%|█| 143/143 [00:01<00:00
    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)


    name='Ren' age=50 armor=<Armor.plate: 'plate'> strength=80

