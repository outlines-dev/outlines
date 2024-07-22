# Grammar-structured generation

You can pass any context-free grammar in the EBNF format and Outlines will generate an output that is valid to this grammar:

```python
from outlines import models, generate

arithmetic_grammar = """
    ?start: expression

    ?expression: term (("+" | "-") term)*

    ?term: factor (("*" | "/") factor)*

    ?factor: NUMBER
           | "-" factor
           | "(" expression ")"

    %import common.NUMBER
"""

model = models.transformers("WizardLM/WizardMath-7B-V1.1")
generator = generate.cfg(model, arithmetic_grammar)
sequence = generator(
  "Alice had 4 apples and Bob ate 2. "
  + "Write an expression for Alice's apples:"
)

print(sequence)
# (8-2)
```

!!! Note "Performance"

    The implementation of grammar-structured generation in Outlines is very naive. This does not reflect the performance of [.txt](https://dottxt.co)'s product, where we made grammar-structured generation as fast as regex-structured generation.


## Ready-to-use grammars

Outlines contains a (small) library of grammars that can be imported and use directly. We can rewrite the previous example as:

```python
from outlines import models, generate

arithmetic_grammar = outlines.grammars.arithmetic

model = models.transformers("WizardLM/WizardMath-7B-V1.1")
generator = generate.cfg(model, arithmetic_grammar)
sequence = generator(
  "Alice had 4 apples and Bob ate 2. "
  + "Write an expression for Alice's apples:"
)

print(sequence)
# (8-2)
```

The following grammars are currently available:

- Arithmetic grammar via `outlines.grammars.arithmetic`
- JSON grammar via `outlines.grammars.json`

If you would like more grammars to be added to the repository, please open an [issue](https://github.com/outlines-dev/outlines/issues) or a [pull request](https://github.com/outlines-dev/outlines/pulls).


## Grammar guide

A grammar is a list of rules and terminals that define a *language*:

- Terminals define the vocabulary of the language; they may be a string, regular expression or combination of these and other terminals.
- Rules define the structure of that language; they are a list of terminals and rules.

Outlines uses the [Lark library](https://github.com/lark-parser/lark) to make Large Language Models generate text in a language of a grammar, it thus uses grammars defined in a format that Lark understands, based on the [EBNF syntax](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form). Read the [Lark documentation](https://lark-parser.readthedocs.io/en/stable/grammar.html) for more details on grammar, the following is a small primer that should help get your started.

In the following we will define a [LOGO-like toy language](https://github.com/lark-parser/lark/blob/master/examples/turtle_dsl.py) for python's [turtle](https://docs.python.org/3/library/turtle.html) library.

### Terminals

A turtle can take 4 different `MOVEMENT` move instructions: forward (`f`), backward (`b`), turn right (`r`) and turn left (`l`). It can take `NUMBER` number of steps in each direction, and draw lines in a specified `COLOR`. These define the vocabulary of our language:

```ebnf
MOVEMENT: "f"|"b"|"r"|"l"
COLOR: LETTER+

%import common.LETTER
%import common.INT -> NUMBER
%import common.WS
%ignore WS
```

The lines that start with `%` are called "directive". They allow to import pre-defined terminals and rules, such as `LETTER` and `NUMBER`. `LETTER+` is a regular expressions, and indicates that a `COLOR` is made of at least one `LETTER`. The last two lines specify that we will ignore white spaces (`WS`) in the grammar.

### Rules

We now need to define our rules, by decomposing instructions we can send to the turtle via our python program. At each line of the program, we can either choose a direction and execute a given number of steps, change the color used to draw the pattern. We can also choose to start filling, make a series of moves, and stop filling. We can also choose to repeat a series of move.

We can easily write the first two rules:

```ebnf
instruction: MOVEMENT NUMBER   -> movement
           | "c" COLOR [COLOR] -> change_color
```

where `movement` and `change_color` represent aliases for the rules. A whitespace implied concatenating the elements, and `|` choosing either of the elements. The `fill` and `repeat` rules are slightly more complex, since they apply to a code block, which is made of instructions. We thus define a new `code_block`  rule that refers to `instruction` and finish implementing our rules:

```ebnf
instruction: MOVEMENT NUMBER            -> movement
           | "c" COLOR [COLOR]          -> change_color
           | "fill" code_block          -> fill
           | "repeat" NUMBER code_block -> repeat

code_block: "{" instruction "}"
```

We can now write the full grammar:

```ebnf
start: instruction+

instruction: MOVEMENT NUMBER            -> movement
            | "c" COLOR [COLOR]          -> change_color
            | "fill" code_block          -> fill
            | "repeat" NUMBER code_block -> repeat

code_block: "{" instruction+ "}"

MOVEMENT: "f"|"b"|"l"|"r"
COLOR: LETTER+

%import common.LETTER
%import common.INT -> NUMBER
%import common.WS
%ignore WS
```

Notice the `start` rule, which defines the starting point of the grammar, i.e. the rule with which a program must start. This full grammars allows us to parse programs such as:

```python
c red yellow
    fill { repeat 36 {
        f200 l170
    }}
```

The result of the parse, the parse tree, can then easily be translated into a Python program that uses the `turtle` library to draw a pattern.

### Next steps

This section provides a very brief overview of grammars and their possibilities. Check out the [Lark documentation](https://lark-parser.readthedocs.io/en/stable/index.html) for more thorough explanations and more examples.
