from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
)

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

    final_state: int = -1

    def get_next_instruction(self, state: int) -> Instruction:
        ...

    def get_next_state(self, state: int, token_id: int) -> int:
        ...

    def is_final_state(self, state: int) -> bool:
        ...

    def align_prompt_tokens(self, prompt: str, tokenizer: "Tokenizer") -> str:
        ...

    def copy(self) -> "Guide":
        ...


class StopAtEOSGuide(Guide):
    """Guide to generate tokens until the EOS token has been generated."""

    start_state = 0

    def __init__(self, tokenizer: "Tokenizer"):
        """Initialize the generation guide.

        model
            The logit generator used to generate the next token.

        """
        self.eos_token_id = tokenizer.eos_token_id
        self.vocabulary = tokenizer.vocabulary
        self.states_to_token_maps = self.create_states_to_tokens_map()

    def create_states_to_tokens_map(self) -> Dict[int, Dict[int, int]]:
        """Create the states_to_tokens_map. All tokens lead to the starting
        state, except for the eos_token that leads to the final state."""
        return {
            self.start_state: {
                token_id: self.start_state
                if token_id != self.eos_token_id
                else self.final_state
                for token_id in self.vocabulary.values()
            }
        }

    def align_prompt_tokens(self, prompt: str, tokenizer: "Tokenizer") -> str:
        """Update the states_to_token_maps and return the aligned prompt"""
        token_ids, _ = tokenizer.encode(prompt)
        # possible return types of tokenizers include list, 1d Tensor and 2d Tensor
        if not isinstance(token_ids, list):
            token_ids = token_ids.tolist()
            if isinstance(token_ids[0], list):
                token_ids = token_ids[0]
        (
            aligned_token_ids,
            aligned_states_to_token_maps,
        ) = align_tokens_states_to_token_maps(
            token_ids, self.vocabulary, deepcopy(self.states_to_token_maps)
        )
        # some tokenizer expect a list of lists while others expect a simple list
        aligned_prompt: list
        try:
            aligned_prompt = tokenizer.decode([aligned_token_ids])
        except TypeError:
            aligned_prompt = tokenizer.decode(aligned_token_ids)
        # some models do not accept an empty string as a prompt
        # if token alignement would remove all tokens, do not apply it
        if not aligned_prompt or not aligned_prompt[0]:
            return prompt
        aligned_prompt = aligned_prompt[0]
        print(prompt, token_ids, aligned_token_ids, aligned_prompt)
        self.states_to_token_maps = aligned_states_to_token_maps
        # remove leading whitespace if added by the tokenizer
        if aligned_prompt[0] == " " and prompt[0] != " ":
            return aligned_prompt[1:]
        return aligned_prompt

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


@cache()
def create_states_mapping(
    regex_string: str,
    tokenizer: "Tokenizer",
    regex_parser: Callable[[str], interegular.Pattern] = interegular.parse_pattern,
    frozen_tokens: List[str] = [],
) -> Tuple[Dict[int, Dict[int, int]], Set[int], set]:
    """Create the variables related to the mapping between states and tokens
    The parameters of the function are used for caching purpose.

    Parameters
    ----------
    regex_string: (`str`):
        The regular expression string to generate a states mapping for.
    tokenizer: (`Tokenizer`):
        The model's tokenizer.
    regex_parser: (`Callable[[str], interegular.Pattern]`, *optional*):
        A function that parses a regex string into an `interegular` Pattern object.
    frozen_tokens: (`List[str]`, *optional*):
        A list of tokens that should be kept as-is when expanding the token-level FSM
        into a byte-level FSM. Defaults to an empty list.

    Returns
    -------
    states_to_token_maps: (`Dict[int, Dict[int, int]]`):
        A mapping from states to a mapping from token ids originating from that state
        to the next state to transition to given that token. The structure is as follows:
        (origin_state -> (token_id -> next_state))
    empty_token_ids: (`Set[int]`):
        A set of token ids that correspond to empty strings.
    final_states: (`set`):
        A set of final states in the FSM.
    """
    regex_pattern = regex_parser(regex_string)
    byte_fsm = make_byte_level_fsm(
        regex_pattern.to_fsm().reduce(), keep_utf8=True, frozen_tokens=frozen_tokens
    )
    regex_fsm, _ = make_deterministic_fsm(byte_fsm)
    states_to_token_maps, empty_token_ids = create_fsm_index_tokenizer(
        regex_fsm, tokenizer, frozen_tokens=frozen_tokens
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

    def __init__(self, regex_string: str, tokenizer: "Tokenizer"):
        (
            self.states_to_token_maps,
            self.empty_token_ids,
            fsm_finals,
        ) = create_states_mapping(regex_string, tokenizer)
        self._cache_state_to_token_tensor()
        self.vocabulary = tokenizer.vocabulary
        self.eos_token_id = tokenizer.eos_token_id
        self.final_states = fsm_finals | {-1}

    def align_prompt_tokens(self, prompt: str, tokenizer: "Tokenizer") -> str:
        """Update the states_to_token_maps and return the aligned prompt"""
        token_ids, _ = tokenizer.encode(prompt)
        # possible return types of tokenizers include list, 1d Tensor and 2d Tensor
        if not isinstance(token_ids, list):
            token_ids = token_ids.tolist()
            if isinstance(token_ids[0], list):
                token_ids = token_ids[0]
        (
            aligned_token_ids,
            aligned_states_to_token_maps,
        ) = align_tokens_states_to_token_maps(
            token_ids, self.vocabulary, deepcopy(self.states_to_token_maps)
        )
        # some tokenizer expect a list of lists while others expect a simple list
        aligned_prompt: str
        try:
            aligned_prompt = tokenizer.decode([aligned_token_ids])[0]
        except TypeError:
            aligned_prompt = tokenizer.decode(aligned_token_ids)[0]
        # some models do not accept an empty string as a prompt
        # if token alignement would remove all tokens, do not apply it
        if not aligned_prompt:
            return prompt
        self.states_to_token_maps = aligned_states_to_token_maps
        self._cache_state_to_token_tensor()
        # remove leading whitespace if added by the tokenizer
        if aligned_prompt[0] == " " and prompt[0] != " ":
            return aligned_prompt[1:]
        return aligned_prompt

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
        from_interegular_instance.vocabulary = tokenizer.vocabulary
        from_interegular_instance.eos_token_id = tokenizer.eos_token_id
        from_interegular_instance._cache_state_to_token_tensor()
        return from_interegular_instance

    def _cache_state_to_token_tensor(self):
        """
        cache state -> token int tensor
        this increases performance of mask construction substantially
        """
        self.states_to_token_mask = {
            state: torch.tensor(list(next_tokens_to_end_states.keys()))
            for state, next_tokens_to_end_states in self.states_to_token_maps.items()
        }

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

    def align_prompt_tokens(self, prompt: str, tokenizer: "Tokenizer") -> str:
        """Not applicable to this type of Guide"""
        return prompt

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


def align_tokens_states_to_token_maps(
    token_ids: List[int],
    vocabulary: Dict[str, int],
    states_to_token_maps: Dict[int, Dict[int, int]],
) -> Tuple[List[int], Dict[int, Dict[int, int]]]:
    """Apply token alignment to the provided prompt tokens and attention masks given the
    states_to_token_maps of a FSM. Return the updated tokens/maps as well as the updated
    states_to_token_maps"""
    crossing_tokens = find_crossing_tokens(token_ids, vocabulary)
    valid_crossing_tokens = get_crossing_tokens_target_states(
        states_to_token_maps, crossing_tokens, token_ids, vocabulary
    )
    if not valid_crossing_tokens:
        return token_ids, states_to_token_maps
    (
        states_to_token_maps,
        number_cropped_tokens,
    ) = add_crossing_tokens_states_to_tokens_map(
        states_to_token_maps, token_ids, valid_crossing_tokens
    )
    return (
        token_ids[:-number_cropped_tokens],
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
