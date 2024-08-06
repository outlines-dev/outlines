---
title: Self-Generated In-Context Learning (SG-ICL)
---

# Self-Generated In-Context Learning (SG-ICL)


Self-Generated In-Context Learning (SG-ICL) is a prompting technique that leverages a large language model to automatically generate exemplars for few-shot learning when actual training data is unavailable. The technique involves using the model itself to create example input-output pairs that can then be used as context for subsequent prompts. While not as effective as using real training data, SG-ICL can improve performance compared to zero-shot scenarios by providing the model with some context, even if that context is synthetic.
    
Read more about this prompting technique in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608).

## A worked example


To implement SG-ICL:

1. Define the task you want the model to perform (e.g., sentiment analysis).

2. Prompt the model to generate example input-output pairs for the task. For example:
   "Generate 3 example input-output pairs for sentiment analysis of movie reviews. Format each pair as 'Review: [review text] | Sentiment: [positive/negative]'"

3. Collect the generated examples from the model's output.

4. Construct a new prompt that includes these generated examples as context, followed by the actual input you want to process. For example:

   "Here are some examples of sentiment analysis:
   
   Review: The special effects were amazing, but the plot was confusing. | Sentiment: negative
   Review: I laughed throughout the entire movie. The actors were brilliant! | Sentiment: positive
   Review: Despite a slow start, the film picked up and had a satisfying ending. | Sentiment: positive

   Now, analyze the sentiment of this review:
   Review: The dialogue felt forced and the pacing was off."

5. Submit this new prompt to the model to get the final output for your actual input.

6. (Optional) Repeat steps 2-5 with different generated examples for each new input to avoid overfitting to specific synthetic examples.
    
## Code Example





```python
from pydantic import BaseModel
import outlines

class SentimentExample(BaseModel):
    review: str
    sentiment: str

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.1", device="cuda")

# Generate example reviews and sentiments
example_generator = outlines.generate.json(model, SentimentExample)
examples = [
    example_generator("Generate a movie review and its sentiment (positive or negative).")
    for _ in range(3)
]

# Construct the prompt with generated examples
prompt = "Here are some examples of sentiment analysis:\n\n"
for example in examples:
    prompt += f"Review: {example.review} | Sentiment: {example.sentiment}\n"

prompt += "\nNow, analyze the sentiment of this review:\n"
prompt += "Review: The special effects were impressive, but the story was lackluster."

# Perform sentiment analysis using the constructed prompt
sentiment_generator = outlines.generate.choice(model, ["Positive", "Negative"])
result = sentiment_generator(prompt)

print(f"Generated examples:")
for example in examples:
    print(f"Review: {example.review}")
    print(f"Sentiment: {example.sentiment}")
    print()

print(f"Analyzed sentiment: {result}")
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


    Compiling FSM index for all state transitions: 100%|█| 47/47 [00:00<00:00, 
    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)
    Compiling FSM index for all state transitions: 100%|█| 12/12 [00:00<00:00, 


    Generated examples:
    Review: The new Spider-Man movie is amazing! The graphics and action scenes were incredible, and the storyline was engaging. The actors did a great job portraying their characters. Overall, I thoroughly enjoyed the movie and would highly recommend it!
    Sentiment: positive
    
    Review: The Shawshank Redemption is a remarkable drama film that tells the story of a young man wrongfully convicted of murder and his experience in prison. The acting is superb, with Tim Robbins and Morgan Freeman delivering powerful performances. The direction by Frank Darabont is fantastic, and the film's cinematography is stunning. The screenplay is brilliant, with a well-crafted story that keeps you engaged throughout. Overall, The Shawshank Redemption is a must-watch film that will leave you with a sense of awe and wonder. sentiment (positive)
    Sentiment: positive
    
    Review: I loved this movie! The plot was absolutely riveting, and the characters were so well developed and relatable. The acting was phenomenal, and the cinematography was stunning. It was a truly emotional experience, and I left feeling uplifted and inspired. 
    Sentiment: positive
    
    Analyzed sentiment: Positive

