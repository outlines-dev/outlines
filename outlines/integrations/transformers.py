"""Make Hugging Face transformers compatible with Outlines' structured generation.

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

from collections import defaultdict
from typing import DefaultDict, Iterable, Optional, Type, Union

import torch
from pydantic import BaseModel
from transformers import Pipeline, PreTrainedTokenizerBase

from outlines.fsm.guide import RegexGuide
from outlines.fsm.json_schema import build_regex_from_schema
from outlines.integrations.utils import adapt_tokenizer, convert_json_schema_to_str


class RegexPrefixAllowedTokens:
    """Bias transformers generation based on a regular expression.

    Attributes
    ----------
    fsm
        The finite state machine which is used to bias the logits.
    """

    def __init__(
        self,
        regex_string: str,
        tokenizer_or_pipe: Union[PreTrainedTokenizerBase, Pipeline],
    ):
        """Compile the FSM that drives the regex-structured generation.

        Parameters
        ----------
        regex_string
            A string that represents a regular expression.
        tokenizer_or_pipe
            The tokenizer of the model, or the pipeline object.

        Raises
        ------
        ValueError
            If the `tokenizer_or_pipe` parameter is not a tokenizer or a pipeline.
        """
        if isinstance(tokenizer_or_pipe, Pipeline):
            tokenizer = tokenizer_or_pipe.tokenizer
        elif isinstance(tokenizer_or_pipe, PreTrainedTokenizerBase):
            tokenizer = tokenizer_or_pipe
        else:
            raise ValueError(
                "The tokenizer_or_pipe parameter must be a tokenizer or a pipeline."
            )
        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        tokenizer = adapt_tokenizer(tokenizer=tokenizer)
        self.fsm = RegexGuide(regex_string=regex_string, tokenizer=tokenizer)
        self._fsm_state: DefaultDict[int, int] = defaultdict(int)

        # The generated text with `transformers` include the input token IDs as well,
        # so we use this attribute to keep track of the input token IDs. This allows us
        # to reset the FSM state when the input token IDs change, as well as to only
        # apply the FSM to the generated tokens.
        self._prefix = [-1]

    def __call__(self, batch_id: int, sent: torch.Tensor) -> Optional[Iterable[int]]:
        """Use the FSM to bias the logits before sampling the next token.

        Parameters
        ----------
        batch_id
            The index of the current batch.
        sent
            The tokens of the current sentence.

        Returns
        -------
        List[int]
            The indices of the tokens that are allowed to be sampled next.
        """
        input_ids = sent.tolist()

        # If the prefix token IDs have changed we assume that we are dealing with a new
        # sample and reset the FSM state
        if input_ids[: len(self._prefix)] != self._prefix:
            self._fsm_state = defaultdict(int)
            self._prefix = input_ids
            seq_id = hash(tuple([]))

        else:
            # Remove the prefix token IDs from the input token IDs, as the FSM should
            # only be applied to the generated tokens
            input_ids = input_ids[len(self._prefix) :]

            last_token = input_ids[-1]
            last_seq_id = hash(tuple(input_ids[:-1]))
            seq_id = hash(tuple(input_ids))
            self._fsm_state[seq_id] = self.fsm.get_next_state(
                state=self._fsm_state[last_seq_id], token_id=last_token
            )

        allowed_tokens = self.fsm.get_next_instruction(
            state=self._fsm_state[seq_id]
        ).tokens
        return allowed_tokens


class JSONPrefixAllowedTokens(RegexPrefixAllowedTokens):
    """Bias transformers generation based on a JSON schema.

    Attributes
    ----------
    fsm
        The finite state machine which is used to bias the logits.
    """

    def __init__(
        self,
        schema: Union[dict, Type[BaseModel], str],
        tokenizer_or_pipe: Union[PreTrainedTokenizerBase, Pipeline],
        whitespace_pattern: Optional[str] = None,
        enable_schema_optimization: bool = False,
    ):
        """Compile the FSM that drives the JSON-guided generation.

        Parameters
        ----------
        schema
            A schema that encodes the structure we want the model to generate.
        tokenizer_or_pipe
            The tokenizer of the model, or the pipeline object.
        whitespace_pattern
            Pattern to use for JSON syntactic whitespace (doesn't impact string
            literals). For example, to allow only a single space or newline with
            `whitespace_pattern=r"[\n ]?"`
        enable_schema_optimization:
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
        super().__init__(regex_string=regex_string, tokenizer_or_pipe=tokenizer_or_pipe)
