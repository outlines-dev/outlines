from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Protocol, Tuple, Union

import interegular
import torch
from lark import Lark

from outlines import grammars
from outlines.caching import cache
from outlines.fsm.regex import create_fsm_index_tokenizer, make_deterministic_fsm

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
        The tokens that lead to a valid completion if generated.
    """

    tokens: List[int]


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

    def align_prompt_tokens(
        self, token_ids: torch.Tensor, attention_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...


class StopAtEOSGuide(Guide):
    """Guide to generate tokens until the EOS token has been generated."""

    final_state = -1
    start_state = 0

    def __init__(self, tokenizer: "Tokenizer"):
        """Initialize the generation guide.

        model
            The logit generator used to generate the next token.

        """
        self.eos_token_id = tokenizer.eos_token_id
        self.vocabulary = tokenizer.vocabulary
        self.tokenizer = tokenizer
        self.states_to_token_maps = self.create_states_to_tokens_map()

    def create_states_to_tokens_map(self) -> Dict[int, Dict[int, int]]:
        """Create the states_to_tokens_map. All tokens from the starting state lead
        to itself, except for the eos_token that leads to the final state."""
        return {
            self.start_state: {
                token_id: self.start_state
                if token_id != self.eos_token_id
                else self.final_state
                for token_id in self.vocabulary.values()
            }
        }

    def align_prompt_tokens(
        self, token_ids: torch.Tensor, attention_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the states_to_token_maps and return the aligned prompt tokens and attention masks"""
        (
            token_ids,
            attention_masks,
            self.states_to_token_maps,
        ) = align_tokens_states_to_token_maps(
            token_ids, attention_masks, self.vocabulary, self.states_to_token_maps
        )
        return token_ids, attention_masks

    def get_next_instruction(self, state: int) -> Instruction:
        if self.is_final_state(state):
            return Write([self.eos_token_id])

        return Generate(list(self.states_to_token_maps[state].keys()))

    def get_next_state(self, state: int, token_id: int) -> int:
        if self.is_final_state(state):
            return self.final_state

        return self.states_to_token_maps[state][token_id]

    def is_final_state(self, state: int):
        return state == self.final_state

    def copy(self):
        return deepcopy(self)


class RegexGuide(Guide):
    """Guide to generate text in the language of a regular expression."""

    initial_state = 0

    def __init__(self, regex_string: str, tokenizer):
        @cache()
        def create_states_mapping(
            regex_string: str, cacheable_vocabulary: Tuple[Tuple[str, int], ...]
        ) -> Tuple[dict, set, set]:
            """Create the variables related to the mapping between states and tokens
            The parameters of the function are used for caching purpose
            """
            regex_pattern = interegular.parse_pattern(regex_string)
            regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm().reduce())
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

            return states_to_token_maps, empty_token_ids, regex_fsm.finals

        (
            self.states_to_token_maps,
            self.empty_token_ids,
            fsm_finals,
        ) = create_states_mapping(
            regex_string, tuple(sorted(tokenizer.vocabulary.items()))
        )
        self.vocabulary = tokenizer.vocabulary
        self.eos_token_id = tokenizer.eos_token_id
        self.final_states = fsm_finals | {-1}

    def align_prompt_tokens(
        self, token_ids: torch.Tensor, attention_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the states_to_token_maps and return the aligned prompt tokens and attention masks"""
        (
            token_ids,
            attention_masks,
            self.states_to_token_maps,
        ) = align_tokens_states_to_token_maps(
            token_ids, attention_masks, self.vocabulary, self.states_to_token_maps
        )
        return token_ids, attention_masks

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
        next_tokens_to_end_states = self.states_to_token_maps.get(state)
        if next_tokens_to_end_states is None:
            return Write([self.eos_token_id])

        return Generate(list(next_tokens_to_end_states.keys()))

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
        if token_id == self.eos_token_id:
            return -1
        elif state in self.final_states:
            return state

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
            fsm: interegular.fsm.FSM, cacheable_vocabulary: Tuple[Tuple[str, int], ...]
        ) -> Tuple[dict, set]:
            """Create the variables related to the mapping between states and tokens
            The parameters of the function are used for caching purpose
            """
            regex_fsm, _ = make_deterministic_fsm(fsm.reduce())
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
        ) = create_states_mapping_from_interegular_fsm(
            interegular_fsm, tuple(sorted(tokenizer.vocabulary.items()))
        )
        from_interegular_instance.vocabulary = list(tokenizer.vocabulary.values())
        from_interegular_instance.eos_token_id = tokenizer.eos_token_id
        return from_interegular_instance

    def is_final_state(self, state: int) -> bool:
        """Determine whether the current state of the guide is a final state."""
        return state in self.final_states

    def copy(self):
        return deepcopy(self)


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

    def align_prompt_tokens(
        self, token_ids: torch.Tensor, attention_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Not applicable to this type of FSM"""
        return token_ids, attention_masks

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


def align_tokens_states_to_token_maps(
    token_ids: torch.Tensor,
    attention_masks: torch.Tensor,
    vocabulary: Dict[str, int],
    states_to_token_maps: Dict[int, Dict[int, int]],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, Dict[int, int]]]:
    """Apply token alignment to the provided prompt tokens and attention masks given the
    states_to_token_maps of a FSM. Return the updated tokens/maps as well as the updated
    states_to_token_maps"""
    prompt_token_ids = token_ids.tolist()
    crossing_tokens = find_crossing_tokens(prompt_token_ids, vocabulary)
    valid_crossing_tokens = get_crossing_tokens_target_states(
        states_to_token_maps, crossing_tokens, prompt_token_ids, vocabulary
    )
    if not valid_crossing_tokens:
        return token_ids, attention_masks, states_to_token_maps
    (
        states_to_token_maps,
        number_cropped_tokens,
    ) = add_crossing_tokens_states_to_tokens_map(
        states_to_token_maps, prompt_token_ids, valid_crossing_tokens
    )
    return (
        token_ids[:-number_cropped_tokens],
        attention_masks[:-number_cropped_tokens],
        states_to_token_maps,
    )


def find_crossing_tokens(
    token_ids: List[int], vocabulary: Dict[str, int]
) -> Dict[int, List[int]]:
    """Find the tokens that could replace one or more tokens at the end of token_ids
    while conserving the same intial text (and extending it by at least one character).
    Return a dictionary with, for the indexes in the token_ids with matches, the associated crossing tokens.
    """
    reversed_vocabulary = {value: key for key, value in vocabulary.items()}
    len_token_ids = len(token_ids)
    max_length_token_text = max(len(item) for item in vocabulary.keys())
    characters_considered = ""
    crossing_tokens_map = {}

    for index, token_id in enumerate(reversed(token_ids)):
        characters_considered = reversed_vocabulary[token_id] + characters_considered
        if len(characters_considered) >= max_length_token_text:
            break
        crossing_token_ids = [
            token_id
            for text, token_id in vocabulary.items()
            if text.startswith(characters_considered)
            and len(text) > len(characters_considered)
        ]
        if crossing_token_ids:
            crossing_tokens_map[len_token_ids - index - 1] = crossing_token_ids

    return crossing_tokens_map


def get_crossing_tokens_target_states(
    states_to_tokens_map: Dict[int, Dict[int, int]],
    crossing_tokens: Dict[int, List[int]],
    prompt_token_ids: List[int],
    vocabulary: Dict[str, int],
) -> Dict[int, Dict[int, int]]:
    """For each crossing token associated to an index, check that the characters after the boundary
    match the states_to_tokens_map and find the state it would lead to. Return a dict with, for each
    provided indexes, the associated valid tokens with the state they would lead to.
    """
    reversed_vocabulary = {value: key for key, value in vocabulary.items()}
    prompt_token_texts = [
        reversed_vocabulary[token_id] for token_id in prompt_token_ids
    ]

    valid_crossing_tokens: Dict[int, Dict[int, int]] = defaultdict(dict)
    for pos, tokens in crossing_tokens.items():
        for token in tokens:
            is_valid = True
            characters = reversed_vocabulary[token]
            characters_before_border = "".join(prompt_token_texts[pos:])
            characters_after_border = characters[len(characters_before_border) :]
            state = 0
            for char in characters_after_border:
                char_token = vocabulary.get(char)
                try:
                    state = states_to_tokens_map[state][char_token]  # type: ignore
                except KeyError:
                    is_valid = False
                    break
            if is_valid:
                valid_crossing_tokens[pos][token] = state

    return valid_crossing_tokens


def add_crossing_tokens_states_to_tokens_map(
    states_to_tokens_map: Dict[int, Dict[int, int]],
    prompt_token_ids: List[int],
    crossing_tokens_map: Dict[int, Dict[int, int]],
) -> Tuple[Dict[int, Dict[int, int]], int]:
    """Modify the states_to_tokens_map to account for the crossing tokens. This operation modifies
    the starting state of the fsm as we would include some characters at the end of the prompt in
    the states_to_tokens_map.
    Attention! the starting state of the states_to_tokens_map provided must be 0.
    Return the updated states_to_tokens_map and the number of cropped tokens/additional states
    """
    if not crossing_tokens_map:
        return states_to_tokens_map, 0
    first_crossing_token_pos = min(
        [key for key, value in crossing_tokens_map.items() if value]
    )
    number_additional_states = len(prompt_token_ids) - first_crossing_token_pos
    highest_state = max(
        max(states_to_tokens_map.keys()),
        max(max(items.values()) for items in states_to_tokens_map.values()),
    )

    for i in range(number_additional_states):
        # add the tokens that was originally part of the prompt
        if i == number_additional_states - 1:
            states_to_tokens_map[highest_state + 1 + i] = {
                prompt_token_ids[first_crossing_token_pos + i]: 0
            }
        else:
            states_to_tokens_map[highest_state + 1 + i] = {
                prompt_token_ids[first_crossing_token_pos + i]: highest_state + 2 + i
            }
        # add the crossing tokens
        crossing_tokens = crossing_tokens_map.get(first_crossing_token_pos + i)
        if crossing_tokens:
            for token, target_state in crossing_tokens.items():
                states_to_tokens_map[highest_state + 1 + i][token] = target_state

    # set the id of our new initial state to 0
    states_to_tokens_map = swap_state_ids_states_to_tokens_map(
        states_to_tokens_map, highest_state + 1, 0
    )
    return states_to_tokens_map, number_additional_states


def swap_state_ids_states_to_tokens_map(
    states_to_tokens_map: Dict[int, Dict[int, int]],
    first_state_id: int,
    second_state_id: int,
) -> Dict[int, Dict[int, int]]:
    """Swap the id of two states of the states_to_tokens_map while conserving all transitions"""
    first_state_transitions = states_to_tokens_map.pop(first_state_id)
    second_state_transitions = states_to_tokens_map.pop(second_state_id)
    states_to_tokens_map[first_state_id] = second_state_transitions
    states_to_tokens_map[second_state_id] = first_state_transitions

    for transitions in states_to_tokens_map.values():
        for token, target_state_id in list(transitions.items()):
            if target_state_id == first_state_id:
                transitions[token] = second_state_id
            elif target_state_id == second_state_id:
                transitions[token] = first_state_id

    return states_to_tokens_map
