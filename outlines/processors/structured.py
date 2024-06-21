"""
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
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Type, Union

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel

from outlines.fsm.guide import CFGGuide, Guide, RegexGuide
from outlines.fsm.json_schema import build_regex_from_schema
from outlines.integrations.utils import convert_json_schema_to_str

from .base_logits_processor import OutlinesLogitsProcessor

if TYPE_CHECKING:
    from outlines.models.tokenizer import Tokenizer


class FSMLogitsProcessor(OutlinesLogitsProcessor):
    """Bias generation using a finite state machine.

    Attributes
    ----------
    tokenizer
        The tokenizer used to convert tokens to ids.
    fsm
        The finite state machine which is used to bias the logits.
    """

    def __init__(self, tokenizer: "Tokenizer", fsm: Guide):
        """A FSM-based logits processor.

        Parameters
        ----------
        tokenizer
            The tokenizer used to convert tokens to ids.
        fsm
            The finite state machine which is used to bias the logits.
        """
        self.tokenizer = tokenizer
        self._fsm_states: Dict[int, int] = {}
        self.fsm: Guide = fsm
        self._is_first_token = True
        self._seq_start_idx: Optional[int] = None

    def process_logits(
        self, input_ids: List[List[int]], logits: torch.Tensor
    ) -> NDArray[np.float32]:
        """Use the FSM to bias the logits before sampling the next token.

        Parameters
        ----------
        input_ids
            The input token ids.
        logits
            The logits.

        Returns
        -------
        torch.Tensor
            The biased logits.
        """
        sequence_states: List[int] = []  # vector of states corresponding to `input_ids`

        if self._is_first_token:
            self._is_first_token = False
            self._seq_start_idx = len(input_ids[0])

            self._fsm_states = {hash(tuple([])): 0}
            sequence_states = [0] * len(input_ids)

        else:
            for seq_ids in input_ids:
                prev_state_key = hash(tuple(seq_ids[self._seq_start_idx : -1]))
                prev_state = self._fsm_states[prev_state_key]

                curr_state_key = hash(tuple(seq_ids[self._seq_start_idx :]))
                curr_state = self.fsm.get_next_state(prev_state, seq_ids[-1])

                self._fsm_states[curr_state_key] = curr_state
                sequence_states.append(curr_state)

        mask = torch.full_like(logits, -math.inf)
        for i, fsm_state in enumerate(sequence_states):
            allowed_tokens = self.fsm.get_next_instruction(fsm_state).tokens
            mask[i, allowed_tokens] = logits[i, allowed_tokens]

        return mask

    def copy(self) -> "FSMLogitsProcessor":
        """Return a copy of the logits processor."""
        return FSMLogitsProcessor(tokenizer=self.tokenizer, fsm=self.fsm.copy())


class RegexLogitsProcessor(FSMLogitsProcessor):
    """Bias generation based on a regular expression.

    Attributes
    ----------
    tokenizer
        The tokenizer used to convert tokens to ids.
    fsm
        The finite state machine which is used to bias the logits.
    """

    def __init__(self, regex_string: str, tokenizer: "Tokenizer"):
        """Compile the FSM that drives the regex-guided generation.

        Parameters
        ----------
        regex_string
            A string that represents a regular expression
        tokenizer
            An Outlines tokenizer
        """
        fsm = RegexGuide(regex_string, tokenizer)
        super().__init__(tokenizer=tokenizer, fsm=fsm)


class JSONLogitsProcessor(RegexLogitsProcessor):
    """Bias generation based on a JSON schema.

    Attributes
    ----------
    tokenizer
        The tokenizer used to convert tokens to ids.
    fsm
        The finite state machine which is used to bias the logits.
    """

    def __init__(
        self,
        schema: Union[dict, Type[BaseModel], str],
        tokenizer: "Tokenizer",
        whitespace_pattern: Optional[str] = None,
    ):
        """Compile the FSM that drives the JSON-guided generation.

        Parameters
        ----------
        schema
            A JSON schema that encodes the structure we want the model to generate.
        tokenizer
            The tokenizer used to convert tokens to ids.
        whitespace_pattern
            Pattern to use for JSON syntactic whitespace (doesn't impact string
            literals). For example, to allow only a single space or newline with
            `whitespace_pattern=r"[\n ]?"`
        """
        schema_str = convert_json_schema_to_str(json_schema=schema)
        regex_string = build_regex_from_schema(schema_str, whitespace_pattern)
        super().__init__(regex_string=regex_string, tokenizer=tokenizer)


class CFGLogitsProcessor(FSMLogitsProcessor):
    """Bias generation based on a context-free grammar.

    Attributes
    ----------
    tokenizer
        The tokenizer used to convert tokens to ids.
    fsm
        The finite state machine which is used to bias the logits.
    """

    def __init__(self, cfg_str: str, tokenizer: "Tokenizer"):
        """Compile the FSM that drives the CFG-guided generation.

        Parameters
        ----------
        cfg_str
            A string that represents a grammar
        tokenizer
            The tokenizer used to convert tokens to ids.
        """
        cfg_automata = CFGGuide(cfg_string=cfg_str, tokenizer=tokenizer)
        super().__init__(tokenizer=tokenizer, fsm=cfg_automata)
