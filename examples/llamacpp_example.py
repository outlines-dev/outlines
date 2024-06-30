from enum import Enum

from pydantic import BaseModel, constr

import outlines


class Weapon(str, Enum):
    sword = "sword"
    axe = "axe"
    mace = "mace"
    spear = "spear"
    bow = "bow"
    crossbow = "crossbow"


class Armor(str, Enum):
    leather = "leather"
    chainmail = "chainmail"
    plate = "plate"


class Character(BaseModel):
    name: constr(max_length=10)
    age: int
    armor: Armor
    weapon: Weapon
    strength: int


if __name__ == "__main__":
    # curl -L -o mistral-7b-instruct-v0.2.Q5_K_M.gguf https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q5_K_M.gguf
    model = outlines.models.llamacpp("./mistral-7b-instruct-v0.2.Q5_K_M.gguf")

    # Construct structured sequence generator
    generator = outlines.generate.json(model, Character)

    # Draw a sample
    seed = 789005

    prompt = "Instruct: You are a leading role play gamer. You have seen thousands of different characters and their attributes.\nPlease return a JSON object with common attributes of an RPG character. Give me a character description\nOutput:"

    sequence = generator(prompt, seed=seed, max_tokens=512)
    print(sequence)
