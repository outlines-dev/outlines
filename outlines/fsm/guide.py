from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Protocol, Tuple, Union

import interegular
import torch
from lark import Lark

from outlines import grammars
from outlines.caching import cache
from outlines.fsm.regex import (
    create_fsm_index_tokenizer,
    make_byte_level_fsm,
    make_deterministic_fsm,
)

if TYPE_CHECKING:
    from outlines.models.tokenizer import Tokenizer


@dataclass(frozen=True)
class Write:
    """Write instruction.

    Attributes
    ----------
    tokens
        The sequence of tokens to be added to the current sequence by the
        generation process.

    """

    tokens: List[int]


@dataclass(frozen=True)
class Generate:
    """Generate instruction

    Attributes
    ----------
    tokens
        The tokens that lead to a valid completion if generated.  A value
        of ``None`` indicates that all tokens are allowed.
    """

    tokens: Optional[List[int]]


Instruction = Union[Write, Generate]


class Guide(Protocol):
    """Base definition of a generation guide.

    A generation guide defines the behavior of a finite-state machine that guides
    a text generation procedure. Unlike the DFAs built from regular expressions
    guides can also emit a `Write` instructions which tells the model that it can
    append a sequence of tokens (or token word) instead of generating it.

    """

    def get_next_instruction(self, state: int) -> Instruction:
        ...

    def get_next_state(self, state: int, token_id: int) -> int:
        ...

    def is_final_state(self, state: int) -> bool:
        ...

    def copy(self) -> "Guide":
        ...


class StopAtEOSGuide(Guide):
    """Guide to generate tokens until the EOS token has been generated."""

    final_state = 1
    start_state = 0

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

        return self.start_state

    def is_final_state(self, state: int):
        return state == self.final_state

    def copy(self):
        return self


@cache()
def create_states_mapping(
    regex_string: str, tokenizer: "Tokenizer"
) -> Tuple[dict, set, set]:
    """Create the variables related to the mapping between states and tokens
    The parameters of the function are used for caching purpose
    """
    regex_pattern = interegular.parse_pattern(regex_string)
    byte_fsm = make_byte_level_fsm(regex_pattern.to_fsm().reduce(), keep_utf8=True)
    regex_fsm, _ = make_deterministic_fsm(byte_fsm)
    states_to_token_maps, empty_token_ids = create_fsm_index_tokenizer(
        regex_fsm, tokenizer
    )

    # We make sure that it is possible to generate strings in the language
    # of the regular expression with the tokens present in the model's
    # vocabulary.
    if not any(
        regex_fsm.finals.intersection(v.values()) for v in states_to_token_maps.values()
    ):
        raise ValueError(
            "The vocabulary does not allow us to build a sequence that matches the input regex"
        )

    return states_to_token_maps, empty_token_ids, regex_fsm.finals


class RegexGuide(Guide):
    """Guide to generate text in the language of a regular expression."""

    initial_state = 0

    def __init__(self, regex_string: str, tokenizer):
        (
            self.states_to_token_maps,
            self.empty_token_ids,
            fsm_finals,
        ) = create_states_mapping(regex_string, tokenizer)
        self.eos_token_id = tokenizer.eos_token_id
        self.final_states = fsm_finals | {-1}

        # cache returned masks token masks
        # this increases performance of the mask substantially
        self.states_to_token_mask = {
            state: torch.tensor(list(next_tokens_to_end_states.keys()))
            for state, next_tokens_to_end_states in self.states_to_token_maps.items()
        }

    def get_next_instruction(self, state: int) -> Instruction:
        """Return the next instruction for guided generation.

        The initialization of the guide builds an index which maps FSM states to a
        map from authorized tokens to the state in which the guide needs to move
        if said token is generated. Therefore the authorized tokens at the
        current state are the keys of the map returned by the value of the index
        for current state.

        If the current state is not contained in the end this means that we are
        in a final state of the guide. We only authorize EOS tokens in the final
        state.

        Parameters
        ----------
        state
            The current state of the guide.

        Returns
        -------
        A `Generate` instance that contains the model and the allowed token ids.

        """
        next_tokens_mask = self.states_to_token_mask.get(state)
        if next_tokens_mask is None:
            return Write(torch.tensor([self.eos_token_id]))

        return Generate(next_tokens_mask)

    def get_next_state(self, state: int, token_id: int) -> int:
        """Update the state of the guide.

        We use the index to determine to which state the guide should transition
        given the token that was just generated.

        Parameters
        ----------
        state
            The current state of the guide.
        token_id
            The id of the token that was just generated.

        Returns
        -------
        The new state of the guide.

        """
        if token_id == self.eos_token_id or state not in self.states_to_token_maps:
            return -1

        last_token_to_end_state = self.states_to_token_maps[state]
        next_state = last_token_to_end_state.get(token_id)
        if next_state is None:
            next_state = -1

        return next_state

    @classmethod
    def from_interegular_fsm(
        cls, interegular_fsm: interegular.fsm.FSM, tokenizer: "Tokenizer"
    ):
        from_interegular_instance = cls.__new__(cls)

        def create_states_mapping_from_interegular_fsm(
            fsm: interegular.fsm.FSM,
        ) -> Tuple[dict, set]:
            """Create the variables related to the mapping between states and tokens
            The parameters of the function are used for caching purpose
            """
            byte_fsm = make_byte_level_fsm(fsm.reduce(), keep_utf8=True)
            regex_fsm, _ = make_deterministic_fsm(byte_fsm)
            states_to_token_maps, empty_token_ids = create_fsm_index_tokenizer(
                regex_fsm, tokenizer
            )

            # We make sure that it is possible to generate strings in the language
            # of the regular expression with the tokens present in the model's
            # vocabulary.
            if not any(
                regex_fsm.finals.intersection(v.values())
                for v in states_to_token_maps.values()
            ):
                raise ValueError(
                    "The vocabulary does not allow us to build a sequence that matches the input regex"
                )

            return states_to_token_maps, empty_token_ids

        (
            from_interegular_instance.states_to_token_maps,
            from_interegular_instance.empty_token_ids,
        ) = create_states_mapping_from_interegular_fsm(interegular_fsm)
        from_interegular_instance.eos_token_id = tokenizer.eos_token_id
        return from_interegular_instance

    def is_final_state(self, state: int) -> bool:
        """Determine whether the current state of the guide is a final state."""
        return state in self.final_states

    def copy(self):
        return self


class CFGGuide(Guide):
    """Guide to generate text that is in the language of a context-free grammar."""

    def __init__(self, cfg_string: str, tokenizer):
        self.cfg_string = cfg_string
        self.tokenizer = tokenizer

        self.parser = Lark(
            cfg_string,
            parser="lalr",
            lexer="contextual",
            propagate_positions=False,
            maybe_placeholders=False,
            regex=True,
            import_paths=[grammars.GRAMMAR_PATH],
        )
        self.terminal_regexps = dict()
        for terminal in self.parser.terminals:
            if terminal.pattern is not None:
                self.terminal_regexps[terminal.name] = terminal.pattern.to_regexp()
        self.terminal_regexps["$END"] = tokenizer.eos_token

        self.generation = ""
        self.reset_state = False
        self.allow_eos = False
        self.regex_fsm: RegexGuide

        self.check_last = False
        self.proposal_last: List[int] = []
        self.regex_fsm_last: RegexGuide

        self.start_state = 0
        self.final_state = -1

    def get_next_instruction(self, state: int) -> Instruction:
        """Generate an instruction for the next step.

        Upon initialization, the CFG incremental parser is used to determine the
        first regex and construct the first FSM to generate the first terminal.

        This FSM is used for proposals until either:

        - The FSM is exhausted, and its only remaining option is the EOS token,
          in which case we feed the generated terminal to the
          CFG incremental parser and allow it to propose the next regex
          corresponding to the next set of valid terminals.
        - The current FSM can be exhausted, but the EOS token is not the only
          remaining option. In this case we allow proposal of current terminal
          extensions, store the current FSM and its state, then also use the CFG
          parser to propose a new regex corresponding to terminating the current
          terminal and starting the next one. The model can then sample from
          either of these sets to determine whether to extend the current
          terminal or terminate it and start the next one.

        The CFG incremental parser is allowed to propose the EOS token from any accepting state,
        and once it is generated, the FSM will continue to always generate the EOS token.

        Parameters
        ----------
        state
            The current state of the FSM.

        Returns
        -------
        A list that contains the tokens to mask.

        """
        if self.is_final_state(state):
            return Write([self.tokenizer.eos_token_id])

        proposal: List[int] = []
        if self.generation != "":
            if self.check_last:
                proposer = self.regex_fsm_last
            else:
                proposer = self.regex_fsm

            instruction = proposer.get_next_instruction(state)

            assert instruction.tokens is not None

            if isinstance(instruction, Write):
                proposal += instruction.tokens
            else:
                proposal += instruction.tokens

            if self.tokenizer.eos_token_id not in proposal:
                return Generate(proposal)

            self.check_last = False
            proposal = [x for x in proposal if x != self.tokenizer.eos_token_id]
            if len(proposal) > 0:
                self.check_last = True
                self.proposal_last = proposal.copy()
                self.regex_fsm_last = proposer

        interactive = self.parser.parse_interactive(self.generation)
        interactive.exhaust_lexer()

        options = {self.terminal_regexps[x] for x in interactive.accepts()}
        # add %ignore terminals
        options |= {self.terminal_regexps[x] for x in self.parser.lexer_conf.ignore}

        if self.terminal_regexps["$END"] in options:
            options.remove(self.terminal_regexps["$END"])
            if len(options) == 0:
                return Write([self.tokenizer.eos_token_id])
            self.allow_eos = True
            options.add("")
            assert len(options) > 1

        regex_string = r"(" + r"|".join([r"(" + x + r")" for x in options]) + r")"
        self.regex_fsm = RegexGuide(regex_string, self.tokenizer)
        self.reset_state = True

        instruction = self.regex_fsm.get_next_instruction(self.start_state)

        assert instruction.tokens is not None

        if isinstance(instruction, Write):
            proposal += instruction.tokens
        else:
            proposal += instruction.tokens

        if self.allow_eos:
            self.allow_eos = False
        else:
            proposal = [x for x in proposal if x != self.tokenizer.eos_token_id]
            assert len(proposal) > 0

        return Generate(proposal)

    def get_next_state(self, state: int, token_id: int) -> int:
        """Update the state of the guide.

        Transitions the underlying regex FSM to its next state.
        If at max tokens or EOS token, transition permanently to the final state.
        Update stored partial generations for subsequent incremental parsing.

        Parameters
        ----------
        state
            The current state of the FSM.
        token_id
            The id of the token that was just generated.

        Returns
        -------
        The new state of the FSM.
        """

        # We need to return the final state when in the final state because we
        # then generate EOS tokens instead of stopping the generation.
        if token_id == self.tokenizer.eos_token_id or state == self.final_state:
            return self.final_state

        self.generation += self.tokenizer.decode([token_id])[0]

        if self.check_last:
            if token_id in self.proposal_last:
                return self.regex_fsm_last.get_next_state(state, token_id)
            self.check_last = False

        if self.reset_state:
            self.reset_state = False
            state = self.start_state

        return self.regex_fsm.get_next_state(state, token_id)

    def is_final_state(self, state: int) -> bool:
        return state == self.final_state

    def copy(self) -> "CFGGuide":
        """Create a copy of the FSM."""
        return CFGGuide(self.cfg_string, self.tokenizer)
