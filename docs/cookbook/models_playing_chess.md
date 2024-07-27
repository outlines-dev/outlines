# Large language models playing chess

In this example we will make a Phi-2 model play chess against itself. On its own the model easily generates invalid moves, so we will give it a little help. At each step we will generate a regex that only matches valid move, and use it to help the model only generating valid moves.

## The chessboard

The game will be played on a standard checkboard. We will use the `chess` [library](https://github.com/niklasf/python-chess) to track the opponents' moves, and check that the moves are valid.

```python
%pip install outlines -q
%pip install chess -q
%pip install transformers accelerate einops -q

import chess

board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
```

## The opponents

Phi-2 will be playing against itself:

```python
from outlines import models

model = models.transformers("microsoft/Phi-3-mini-4k-instruct")

```

## A little help for the language model

To make sure Phi-2 generates valid chess moves we will use Outline's regex-structured generation. We define a function that takes the current state of the board and returns a regex that matches all possible legal moves:

```python
import re

def legal_moves_regex(board):
    """Build a regex that only matches valid moves."""
    legal_moves = list(board.legal_moves)
    legal_modes_str = [board.san(move) for move in legal_moves]
    legal_modes_str = [re.sub(r"[+#]", "", move) for move in legal_modes_str]
    regex_pattern = "|".join(re.escape(move) for move in legal_modes_str)
    regex_pattern = f"{regex_pattern}"
    return regex_pattern
```

## Prompting the language model

The prompt corresponds to the current state of the board, so we start with:

```python
prompt = "Let's play Chess. Moves: "

```

We update the prompt at each step so it reflects the state of the board after the previous move.

## Let's play

```python
from outlines import generate

board_state = " "
turn_number = 0
while not board.is_game_over():
    regex_pattern = legal_moves_regex(board)
    structured = generate.regex(model, regex_pattern)(prompt + board_state)
    move = board.parse_san(structured)

    if turn_number % 2 == 0 :  # It's White's turn
        board_state += board.san(move) + " "
    else:
        board_state += board.san(move) + " " + str(turn_number) + "."

    turn_number += 1

    board.push(move)

    print(board_state)
```

Interestingly enough, Phi-2 hates capturing.

```pgn
 e4 e5 1.Nf3 Ne7 3.b4 Nf5 5.Nc3 Ne7 7.Bb5 a6 9.Na4 b6 11.c3 Nec6 13.c4 a5 15.d4 Qg5 17.Nd2 Bb7 19.dxe5
```

*This example was originally authored by [@903124S](https://x.com/903124S) in [this gist](https://gist.github.com/903124/cfbefa24da95e2316e0d5e8ef8ed360d).*
