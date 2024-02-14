---
title: Debugging
---

# Debugging

Language models are a new and complex area of research. Adding constrained generation to them can bring up unexpected issues. We have some debugging methods documented here to help determine the source(s) of problems.

## Understanding Generation Quality Issues

Language models determine the probability of every possible next-token based on the full sequence of preceding tokens. In Outlines, we can either choose the highest probability next token ("Greedy") or select randomly weighted by token probability ("Multinomial").

If the output quality is lacking, the model (or prompt) might not be well-suited for your particular use case. Logging next-token-probabilities, based on the model's "logits" can help with troubleshooting. This can be accomplished via `outlines.logging.enable_logits_logging()`.

_(Note: `enable_logits_logging()` will slow down generation and shouldn't be used in production.)_

### Example:

In this debug example we attempt to extract sentiment from a restaurant review, but at first the model is struggling.

```python
import outlines.logging
outlines.logging.enable_logits_logging()

import outlines

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")

prompt = """You are a sentiment-labelling assistant.
Is the following review positive or negative?

Review: This restaurant is just awesome!
"""

generator = outlines.generate.choice(model, ["Positive", "Negative"])
answer = generator(prompt)
```

#### Output

```
Selected: 'N' for batch_item=0
	Top Raw Tokens: 'The': 0.560, '\n': 0.212, 'I': 0.106, 'Great': 0.017, 'From': 0.017, 'We': 0.013, 'F': 0.011, 'They': 0.006, EOS: 0.006
	Top Guided Tokens: 'P': 0.456, 'N': 0.330, 'Pos': 0.173, 'Ne': 0.018, 'Neg': 0.016, 'Po': 0.007, 'P': 0.000, 'N': 0.000, EOS: -0.000
Selected: 'ega' for batch_item=0
	Top Raw Tokens: 'ice': 0.964, 'ut': 0.006, 'at': 0.006, 'ood': 0.004, 'ic': 0.003, 'ick': 0.002, 'ear': 0.002, 'umer': 0.002, EOS: 0.002
	Top Guided Tokens: 'eg': 0.514, 'ega': 0.419, 'e': 0.065, 'e': 0.001, '\x00': 0.000, '\x04': 0.000, '': 0.000, '\x01': 0.000, EOS: -0.000
Selected: 't' for batch_item=0
	Top Raw Tokens: 'ive': 0.442, 'ative': 0.381, 'тив': 0.054, ':': 0.030, 'iv': 0.006, 'itive': 0.006, 'Review': 0.005, 'ativ': 0.003, EOS: 0.003
	Top Guided Tokens: 't': 0.903, 'ti': 0.097, 't': 0.000, '\x04': 0.000, '\x00': 0.000, '': 0.000, '\x01': 0.000, '\x02': 0.000, EOS: -0.000
Selected: 'ive' for batch_item=0
	Top Raw Tokens: 'ive': 0.993, 'ion': 0.005, 'iv': 0.002, 've': 0.000, 'ivity': 0.000, 'ively': 0.000, 'ives': 0.000, 'if': 0.000, EOS: 0.000
	Top Guided Tokens: 'ive': 0.998, 'iv': 0.002, 'i': 0.000, 'i': 0.000, '\x00': 0.000, '\x04': 0.000, '': 0.000, '\x01': 0.000, EOS: -0.000
Selected: '' for batch_item=0
	Top Raw Tokens: ':': 0.625, '\n': 0.227, 'or': 0.073, ',': 0.022, '/': 0.016, '.': 0.008, '?': 0.007, '-': 0.003, EOS: 0.003
	Top Guided Tokens: EOS: 1.000, '': 0.000, '\x04': 0.000, '\x01': 0.000, '\x00': 0.000, '': 0.000, '\x02': 0.000, '\x03': 0.000
```

#### Analysis

The model incorrectly classified the review as "Negative".

We can observe in the "Raw Tokens" section that prior to constraining generation to "Positive" / "Negative" the most likely next token was `The`, and tokens allowing for legal generations had very low probabilities:

```
	Top Raw Tokens: 'The': 0.560, '\n': 0.212, 'I': 0.106, 'Great': 0.017, 'From': 0.017, 'We': 0.013, 'F': 0.011, 'They': 0.006, EOS: 0.006
	Top Guided Tokens: 'P': 0.456, 'N': 0.330, 'Pos': 0.173, 'Ne': 0.018, 'Neg': 0.016, 'Po': 0.007, 'P': 0.000, 'N': 0.000, EOS: -0.000
```

#### Fix

Ideally the Raw Tokens are closely aligned to Guided Tokens. To accomplish this, we update the prompt by appending a statement which we would expect to immediately precede "Positive" or "Negative" - "Review label:"

```python
prompt = """You are a sentiment-labelling assistant.
Is the following review positive or negative?

Review: This restaurant is just awesome!

Review label:
"""
```

This change results in substantially better "Raw Tokens" and an accurate label being applied, the model assigns a 100% probability to "Positive", whereas previously the chance of "Positive" was ~64%:

```
Selected: 'Pos' for batch_item=0
	Top Raw Tokens: 'Pos': 0.858, 'POS': 0.060, '\n': 0.031, 'pos': 0.022, '+': 0.007, '**': 0.007, 'The': 0.004, 'Pos': 0.002, EOS: 0.002
	Top Guided Tokens: 'Pos': 1.000, 'P': 0.000, 'Po': 0.000, 'Neg': 0.000, 'Ne': 0.000, 'N': 0.000, 'P': 0.000, 'N': 0.000, EOS: -0.000
```
