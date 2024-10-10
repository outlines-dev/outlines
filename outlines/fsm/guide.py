import collections
import copy
import warnings
from typing import TYPE_CHECKING, Any, Generator, Union

import torch
from lark.indenter import DedentError
from lark.lexer import UnexpectedCharacters, UnexpectedToken
from outlines_core.fsm.guide import Generate
from outlines_core.fsm.guide import Guide as CoreGuide
from outlines_core.fsm.guide import RegexGuide as CoreRegexGuide
from outlines_core.fsm.guide import Write
from outlines_core.fsm.guide import (
    create_states_mapping as uncached_create_states_mapping,
)

from outlines import grammars
from outlines.caching import cache
from outlines.fsm.parsing import PartialLark, PartialParserState

if TYPE_CHECKING:
    from outlines.models.tokenizer import Tokenizer


Instruction = Union[Write, Generate]


class Guide(CoreGuide):
    """Base definition of a generation guide.

    A generation guide defines the behavior of a finite-state machine that guides
    a text generation procedure. Unlike the DFAs built from regular expressions
    guides can also emit a `Write` instructions which tells the model that it can
    append a sequence of tokens (or token word) instead of generating it.

    """

    initial_state: Any


class StopAtEOSGuide(Guide):
    """Guide to generate tokens until the EOS token has been generated."""

    final_state = 1
    start_state = 0  # TODO: remove start_state, use only initial_state
    initial_state = 0

    def __init__(self, tokenizer: "Tokenizer"):
        """Initialize the generation guide.

        model
            The logit generator used to generate the next token.

        """
        self.eos_token_id = tokenizer.eos_token_id
        self.vocabulary = tokenizer.vocabulary.values()

    def get_next_instruction(self, state: int) -> Instruction:
        if self.is_final_state(state):
            return Write([self.eos_token_id])
        return Generate(None)

    def get_next_state(self, state: int, token_id: int) -> int:
        if token_id == self.eos_token_id or state == self.final_state:
            return self.final_state

        return self.initial_state

    def is_final_state(self, state: int):
        return state == self.final_state

    def copy(self):
        return self


@cache()
def cached_create_states_mapping(regex_string, tokenizer, *args, **kwargs):
    return uncached_create_states_mapping(regex_string, tokenizer, *args, **kwargs)


class RegexGuide(CoreRegexGuide):
    """
    Guide to generate text in the language of a regular expression.
    CoreRegexGuide with outlines cache
    """

    @classmethod
    def from_regex(
        cls,
        regex_string: str,
        tokenizer,
        _create_states_mapping=cached_create_states_mapping,
        **kwargs,
    ):
        return super().from_regex(
            regex_string,
            tokenizer,
            _create_states_mapping=_create_states_mapping,
            **kwargs,
        )


CFGState = collections.namedtuple("CFGState", ["parser_state", "prev_token"])


class CFGGuide(Guide):
    """Guide to generate text that is in the language of a context-free Lark grammar."""

    def __init__(self, cfg_string: str, tokenizer):
        """
        Construct the PartialLark parser and set the empty initial_state (PartialParserState)
        """
        warnings.warn(
            "Outlines' public *community-contributed* CFG structured generation is experimental. "
            "Please review https://dottxt-ai.github.io/outlines/latest/reference/generation/cfg#disclaimer"
        )

        self.cfg_string = cfg_string
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.parser = PartialLark(
            cfg_string,
            parser="lalr",
            import_paths=[grammars.GRAMMAR_PATH],
        )
        self.initial_state = CFGState(
            parser_state=self.parser.parse(""), prev_token=None
        )

    def get_next_instruction(self, state: CFGState) -> Instruction:
        """Return the next instruction for guided generation.

        Current lazy approach:
        - For each token in the vocabulary
          - create a copy of the parsers state
          - add the tokens to the parsers input text
          - if valid, add token to returned tokens

        Further refinements are necessary for performant text processing.

        Parameters
        ----------
        state
            The guides current PartialParserState, or None if complete

        Returns
        -------
        A `Generate` instance that contains the model and the allowed token ids.

        """

        if state.parser_state is None:
            return Write(torch.tensor([self.eos_token_id]))

        valid_tokens = list(
            self.iter_valid_token_ids(state, self.tokenizer.vocabulary.values())
        )
        if len(valid_tokens) == 1:
            return Write(torch.tensor(valid_tokens))
        return Generate(torch.tensor(valid_tokens))

    def iter_valid_token_ids(
        self, state: CFGState, candidate_token_ids: list
    ) -> Generator[int, None, None]:
        """
        Iterate over the given token_ids and yield those that are valid for the current parser state.

        Parameters
        ----------
        parser_state
            The current state of the parser, or None if complete.
        token_ids
            The list of token ids to check for validity.

        Yields
        ------
        int
            Valid token ids.
        """
        if state.parser_state is None:
            yield self.eos_token_id
            return

        for token_id in candidate_token_ids:
            if token_id == self.eos_token_id:
                if self.can_terminate_state(state):
                    yield token_id
            else:
                try:
                    self._get_parser_state_token_applied(state, int(token_id))
                    yield token_id
                except (
                    ValueError,
                    EOFError,
                    UnexpectedToken,
                    UnexpectedCharacters,
                    DedentError,
                ):
                    pass

    def get_next_state(self, state: CFGState, token_id: int) -> CFGState:
        """
        Update the state of the guide.
        Decode the token_id, and calculate the new parser_state with the token applied.

        Parameters
        ----------
        state
            The guides current PartialParserState, or None if complete
        token_id
            The id of the token that was just generated.

        Returns
        -------
        The guides new PartialParserState

        """
        if state.parser_state is None or token_id == self.eos_token_id:
            parser_state = None
        else:
            parser_state = self._get_parser_state_token_applied(state, int(token_id))
        return CFGState(parser_state=parser_state, prev_token=token_id)

    def _get_parser_state_token_applied(
        self, state: CFGState, token_id: int
    ) -> PartialParserState:
        """
        Don't mutate `parser_state`, copy to protect

        Get the token string
          - if first token in generation: tokenizer.decode (no leading whitespace)
          - else: normalized (with possibly leading whitespace)

        Don't allow empty ("") tokens, raise ValueError
        """
        parser_state = copy.copy(state.parser_state)  # prevent side effects

        # normalize
        if state.prev_token is None:
            new_token_str = self.tokenizer.decode([token_id])[0]
        else:
            prev_token_str = self.tokenizer.decode([[state.prev_token]])[0]
            combined_token_str = self.tokenizer.decode([[state.prev_token, token_id]])[
                0
            ]
            new_token_str = combined_token_str[len(prev_token_str) :]

        if new_token_str == "":
            raise ValueError("empty next token")

        # update parser with new token
        parser_state.lexer.state.text += new_token_str
        self.parser.parse_from_state(parser_state, is_end=False)

        return parser_state

    def is_final_state(self, state: CFGState) -> bool:
        # TODO: remove this method, use can_terminate_state and must_terminate_state
        # here and in RegexGuide per https://github.com/dottxt-ai/outlines/issues/885
        return self.can_terminate_state(state)

    def can_terminate_state(self, state: CFGState) -> bool:
        """Generation is allowed to terminate"""
        if state.parser_state is not None:
            try:
                copy.copy(state.parser_state).feed_eof()
            except UnexpectedToken:
                return False
        return True

    def must_terminate_state(self, state: CFGState) -> bool:
        """Generation must terminate, no legal continuations"""
        return state.parser_state is None or set(state.parser_state.accepts()).issubset(
            {"$END"}
        )

    def copy(self) -> "CFGGuide":
        """Create a copy of the Guide."""
        return CFGGuide(self.cfg_string, self.tokenizer)
