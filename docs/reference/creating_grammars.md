# Overview

Outlines allows the use of [Lark](https://github.com/lark-parser/lark) grammars to guide generation. These grammars are used to construct parsers that filter out incompatible tokens during the generation process The result is a generation that adheres to the grammar's production rules.

# Primer on Creating Grammars

To create grammars for Outlines, a solid understanding of Lark grammars is necessary. Here's how you can get started:

- Read Lark's grammars documentations [here](https://lark-parser.readthedocs.io/en/latest/grammar.html).
- Review Outlines' existing grammars [here](/outlines/grammars).


# Compatibility With Outlines

It's important to note that not all Lark grammars work with Outlines. Changes may be necessary to ensure compatability.

### LALR(1) Parser

Outlines utilizes Larks LALR(1) parser, meaning the grammar must be unambiguous at least up to the next token (one token lookahead). Read Lark's official LALR(1) parser documentation [here](https://lark-parser.readthedocs.io/en/stable/parsers.html#lalr-1).

If your grammar is ambiguous, you will recieve the following error at runtime:

```
GrammarError: Reduce/Reduce collision in Terminal('B') between the following rules:
```

### Regex Terminal Restrictions

Outlines converts terminals to finite state machines using the [Interegular](https://github.com/MegaIng/interegular/) library. Not all regular expressions work with Interegular, mitigation is described in the subsections which follow.


#### Avoid Lookarounds

Examples of removing lookaround while maintaining the same functionality

##### Example: Escaped String

From Outlines' modified `ESCAPED_STRING` in [common.lark](/outlines/grammars/common.lark).

Before:
```
_STRING_INNER: /.*?/
_STRING_ESC_INNER: _STRING_INNER /(?<!\\)(\\\\)*?/

ESCAPED_STRING : "\"" _STRING_ESC_INNER "\""
```

After:
```
_NON_CONTROL_CHAR: /([^"\\\x00-\x1F\x7F-\x9F])/
_ESCAPED_CHAR: /\\/ (_NON_CONTROL_CHAR | /\\/ | /"/)
ESCAPED_STRING_INNER: _NON_CONTROL_CHAR | _ESCAPED_CHAR
ESCAPED_STRING: /"/ ESCAPED_STRING_INNER* /"/
```

#### Avoid Backreferences

Backreferences, for example `([ab]^*)\1`, cannot be simulated by a finite state machine, and will result in an error if used.

# Creating a Valid Grammar

You can use Outlines' test suite to verify your grammar.

### 1) Create Your Grammar

Create your grammar file named `your_new_grammar.lark`, adhering to the guidelines provided above. Add it to `outlines/grammars/` (ensure attribution is included and license is compatible).

Update `outlines/grammars.py` with a line including your grammar.

### 2) Test Your Grammar

Test grammar for false negatives, ensure sample grammars can be generated:
- Add valid example outputs which are compliant with the grammar to `tests/benchmark/cfg_samples/your_new_grammar/`
- Run the tests for your grammar via `pytest -s tests/fsm/test_cfg_guide.py::test_cfg_grammar_sample -k "your_new_grammar"`

Test grammar for false positives, ensure invalid outputs aren't generated.

Currently there isn't a builtin false positive testing utility. It is recommended you smoke test via
```
from outlines import models, generate, grammars
model = models.transformers("mistralai/Mistral-7B-v0.1")
generator = generate.cfg(model, grammars.your_new_grammar)
result = generator(<your prompt to generate output for your grammar>)
print(result)
```

# Converting
There are a few tools available for converting from other grammars to lark. These tools serve as a starting point. However, you will typically need to make additional adjustments to ensure full compatibility and proper functioning within Outlines.

Tools:
- Larks built in "Nearley-to-Lark" converter https://lark-parser.readthedocs.io/en/latest/tools.html
- Convert ANTLR4 to Lark (Note, most antlr4 grammars are not LALR(1) compatible, so will require additional tweaking) https://github.com/kaby76/Domemtech.Trash/blob/main/src/trconvert/readme.md
- Extract EBNF from Yacc files https://www.bottlecaps.de/rr/ui

Reference Grammars:
- Github Lark Grammars https://github.com/search?q=path%3A*.lark&type=code
- Github Nearley Grammars https://github.com/search?q=path%3A*.ne+%22-%3E%22&type=code
- Antlr4 grammars https://github.com/antlr/grammars-v4/
- Grammar zoo https://slebok.github.io/zoo/index.html#html
