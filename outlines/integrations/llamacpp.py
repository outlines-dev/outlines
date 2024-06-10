"""Make LlamaCpp compatible with Outlines' structured generation.

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
from typing import TYPE_CHECKING, Optional, Type, Union

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel

from outlines.fsm.guide import CFGGuide, Guide, RegexGuide
from outlines.fsm.json_schema import build_regex_from_schema
from outlines.integrations.utils import convert_json_schema_to_str
from outlines.models.llamacpp import LlamaCppTokenizer

if TYPE_CHECKING:
    from llama_cpp import Llama


class LogitsProcessor:
    """Bias LlamaCpp generation using a finite state machine.

    Attributes
    ----------
    tokenizer
        The tokenizer used to convert tokens to ids.
    fsm
        The finite state machine which is used to bias the logits.
    """

    def __init__(self, tokenizer: LlamaCppTokenizer, fsm: Guide):
        """A FSM-based logits processor.

        Parameters
        ----------
        tokenizer
            The tokenizer used to convert tokens to ids.
        fsm
            The finite state machine which is used to bias the logits.
        """
        self.tokenizer = tokenizer
        self._fsm_state = 0
        self.fsm: Guide = fsm
        self._is_first_token = True

    def __call__(
        self, input_ids: NDArray[np.int64], scores: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Use the FSM to bias the logits before sampling the next token.

        Parameters
        ----------
        input_ids
            The input token ids.
        scores
            The logits.

        Returns
        -------
        NDArray[np.float32]
            The biased logits.
        """
        if self._is_first_token:
            self._is_first_token = False
        else:
            last_token = input_ids[-1]
            self._fsm_state = self.fsm.get_next_state(self._fsm_state, last_token)

        allowed_tokens = self.fsm.get_next_instruction(self._fsm_state).tokens

        mask = torch.full((scores.shape[-1],), -math.inf, device="cpu").numpy()
        mask[allowed_tokens] = 0
        biased_scores = scores + mask

        return biased_scores

    def copy(self) -> "LogitsProcessor":
        """Return a copy of the logits processor."""
        return LogitsProcessor(tokenizer=self.tokenizer, fsm=self.fsm.copy())


class RegexLogitsProcessor(LogitsProcessor):
    """Bias LlamaCpp generation based on a regular expression.

    Attributes
    ----------
    tokenizer
        The tokenizer used to convert tokens to ids.
    fsm
        The finite state machine which is used to bias the logits.
    """

    def __init__(self, regex_string: str, llm: "Llama"):
        """Compile the FSM that drives the regex-guided generation.

        Parameters
        ----------
        regex_string
            A string that represents a regular expression
        llm
            The Llama model.
        """
        tokenizer = LlamaCppTokenizer(model=llm)
        fsm = RegexGuide(regex_string, tokenizer)
        super().__init__(tokenizer=tokenizer, fsm=fsm)


class JSONLogitsProcessor(RegexLogitsProcessor):
    """Bias LlamaCpp generation based on a JSON schema.

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
        llm: "Llama",
        whitespace_pattern: Optional[str] = None,
        enable_schema_optimization: bool = False,
    ):
        """Compile the FSM that drives the JSON-guided generation.

        Parameters
        ----------
        schema
            A JSON schema that encodes the structure we want the model to generate.
        llm
            The Llama model.
        whitespace_pattern
            Pattern to use for JSON syntactic whitespace (doesn't impact string
            literals). For example, to allow only a single space or newline with
            `whitespace_pattern=r"[\n ]?"`
        enable_schema_optimization
            If True, this will speed up generation by not requiring optional keys to be
            present in the output. This is especially useful for large schemas with many
            optional keys. Note though that this further restricts the support
            distribution. Thus, it is necessary to remove the optional keys from the
            finetuning dataset as well if needed. Hence, we set this to False by default.
        """
        schema_str = convert_json_schema_to_str(json_schema=schema)
        regex_string = build_regex_from_schema(
            schema_str, whitespace_pattern, enable_schema_optimization
        )
        super().__init__(regex_string=regex_string, llm=llm)


class CFGLogitsProcessor(LogitsProcessor):
    """Bias LlamaCpp generation based on a context-free grammar.

    Attributes
    ----------
    llm
        The Llama model.
    fsm
        The finite state machine which is used to bias the logits.
    """

    def __init__(self, cfg_str: str, llm: "Llama"):
        """Compile the FSM that drives the CFG-guided generation.

        Parameters
        ----------
        cfg_str
            A string that represents a grammar
        llm
            The Llama model.
        """
        tokenizer = LlamaCppTokenizer(model=llm)
        fsm = CFGGuide(cfg_string=cfg_str, tokenizer=tokenizer)
        super().__init__(tokenizer=tokenizer, fsm=fsm)
