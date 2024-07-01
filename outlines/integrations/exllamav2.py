"""Make ExllamaV2 compatible with Outlines' structured generation.

 _______________________________
/ Don't want to self-host?       \
\\ Try .json at http://dottxt.co /
 -------------------------------
       \\   ^__^
        \\  (oo)\\_______
            (__)\\       )\\/\
                ||----w |
                ||     ||

Copyright 2024- the Outlines developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import copy
from collections import defaultdict
from typing import DefaultDict, Optional, Type, Union

import torch
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase

from outlines.fsm.guide import Guide, RegexGuide, StopAtEOSGuide
from outlines.fsm.json_schema import build_regex_from_schema
from outlines.generate.generator import is_generation_finished
from outlines.integrations.utils import adapt_tokenizer, convert_json_schema_to_str


class FSMFilter:
    """Bias transformers generation based on a fsm.

    Attributes
    ----------
    fsm
        The finite state machine which is used to bias the logits.
    """

    token_sequence: list[int]
    seq_id: int

    def __init__(self, fsm: Guide):
        """Compile the FSM that drives generation.

        Parameters
        ----------
        fsm
            The fsm of the model.
        """
        self.fsm = fsm
        self._fsm_state: DefaultDict[int, int] = defaultdict(int)
        self.token_sequence = []

    def begin(self, prefix_str: str = "") -> None:
        self._fsm_state = defaultdict(int)
        self.seq_id = hash(tuple([]))

    def feed(self, token: torch.Tensor) -> None:
        int_token = int(token[0][0].numpy())

        last_seq_id = self.seq_id
        self.token_sequence.append(int_token)
        self.seq_id = hash(tuple(self.token_sequence))
        self._fsm_state[self.seq_id] = self.fsm.get_next_state(
            state=self._fsm_state[last_seq_id], token_id=int_token
        )

    def clone(self):
        return copy.deepcopy(self)

    def next(self) -> tuple[set[int], set[int]]:
        allowed_tokens = self.fsm.get_next_instruction(
            state=self._fsm_state[self.seq_id]
        ).tokens
        if allowed_tokens is None:
            allowed_tokens = []
        end_tokens = []
        for token in allowed_tokens:
            next_state = self.fsm.get_next_state(
                state=self._fsm_state[self.seq_id], token_id=token
            )
            if is_generation_finished([self.fsm], [next_state]):
                end_tokens.append(token)
        return set(allowed_tokens), set(end_tokens)


class RegexFilter(FSMFilter):
    """Bias transformers generation based on a regular expression.

    Attributes
    ----------
    fsm
        The finite state machine which is used to bias the logits.
    """

    def __init__(
        self,
        regex_string: str,
        tokenizer: PreTrainedTokenizerBase,
    ):
        """Compile the FSM that drives the regex-structured generation.

        Parameters
        ----------
        tokenizer
            The tokenizer of the model.

        Raises
        ------
        ValueError
            If the `tokenizer` parameter is not a tokenizer.
        """
        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        tokenizer = adapt_tokenizer(tokenizer=tokenizer)
        fsm = RegexGuide(regex_string=regex_string, tokenizer=tokenizer)
        super().__init__(fsm)


class TextFilter(FSMFilter):
    """Bias transformers generation based on a stop at eos text expression.

    Attributes
    ----------
    fsm
        The finite state machine which is used to bias the logits.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
    ):
        """Compile the FSM that drives text generation.

        Parameters
        ----------
        tokenizer
            The tokenizer of the model.

        Raises
        ------
        ValueError
            If the `tokenizer` parameter is not a tokenizer.
        """
        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        tokenizer = adapt_tokenizer(tokenizer=tokenizer)
        fsm = StopAtEOSGuide(tokenizer=tokenizer)
        super().__init__(fsm)


class JSONFilter(RegexFilter):
    """Bias exllamav2 generation based on a JSON schema.

    Attributes
    ----------
    fsm
        The finite state machine which is used to bias the logits.
    """

    def __init__(
        self,
        schema: Union[dict, Type[BaseModel], str],
        tokenizer: PreTrainedTokenizerBase,
        whitespace_pattern: Optional[str] = None,
    ):
        """Compile the FSM that drives the JSON-guided generation.

        Parameters
        ----------
        schema
            A schema that encodes the structure we want the model to generate.
        tokenizer
            The tokenizer of the model.
        whitespace_pattern
            Pattern to use for JSON syntactic whitespace (doesn't impact string
            literals). For example, to allow only a single space or newline with
            `whitespace_pattern=r"[\n ]?"`
        """
        schema_str = convert_json_schema_to_str(json_schema=schema)
        regex_string = build_regex_from_schema(schema_str, whitespace_pattern)
        super().__init__(regex_string=regex_string, tokenizer=tokenizer)


class ChoiceFilter(RegexFilter):
    """Bias exllamav2 generation based on choices.

    Attributes
    ----------
    fsm
        The finite state machine which is used to bias the logits.
    """

    def __init__(
        self,
        choices: list[str],
        tokenizer: PreTrainedTokenizerBase,
    ):
        """Compile the FSM that drives the JSON-guided generation.

        Parameters
        ----------
        schema
            A schema that encodes the structure we want the model to generate.
        tokenizer
            The tokenizer of the model.
        whitespace_pattern
            Pattern to use for JSON syntactic whitespace (doesn't impact string
            literals). For example, to allow only a single space or newline with
            `whitespace_pattern=r"[\n ]?"`
        """
        regex_string = r"(" + r"|".join(choices) + r")"
        super().__init__(regex_string=regex_string, tokenizer=tokenizer)
