import collections
import copy
import warnings
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
)

import interegular
import torch
from lark.indenter import DedentError
from lark.lexer import UnexpectedCharacters, UnexpectedToken

from outlines import grammars
from outlines.caching import cache
from outlines.fsm.parsing import PartialLark, PartialParserState
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

    initial_state: Any

    def get_next_instruction(self, state: Any) -> Instruction:
        ...

    def get_next_state(self, state: Any, token_id: int) -> Any:
        ...

    def is_final_state(self, state: Any) -> bool:
        ...

    def copy(self) -> "Guide":
        ...

    def accepts(self, token_ids: List[int], state=None) -> bool:
        """
        Determine whether the sequence, `token_ids`, is accepted by the Guide.
        `token_ids` doesn't need to complete the guide to be accepted.
        """
        try:
            self.derive(token_ids, state)
            return True
        except ValueError:
            return False

    def derive(self, token_ids: List[int], state=None) -> Union["Guide", None]:
        """
        TODO: Docstring
        """
        if state is None:
            state = self.initial_state
        for token_id in token_ids:
            instruction = self.get_next_instruction(state)

            # determine if token_id allowed by instruction
            if isinstance(instruction, Write):
                raise NotImplementedError("TODO")
            elif isinstance(instruction, Generate):
                if (
                    instruction.tokens is not None
                    and token_id not in instruction.tokens
                ):
                    raise ValueError("Cannot advance state with provided token_ids")
            else:
                raise TypeError(f"Expected instruction, got {instruction}")

            # advance state
            state = self.get_next_state(state, token_id)

        return state


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
        self.eos_token_id = tokenizer.eos_token_id
        self.final_states = fsm_finals | {-1}
        self._cache_state_to_token_tensor()

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
        return self


CFGState = collections.namedtuple("CFGState", ["parser_state", "prev_token"])


class CFGGuide(Guide):
    """Guide to generate text that is in the language of a context-free Lark grammar."""

    def __init__(self, cfg_string: str, tokenizer):
        """
        Construct the PartialLark parser and set the empty initial_state (PartialParserState)
        """
        warnings.warn(
            "Outlines' public *community-contributed* CFG structured generation is experimental. "
            "Please review https://dottxt-ai.github.io/outlines/reference/cfg#disclaimer"
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


@cache()
def build_vocab_prefix_map(tokenizer: "Tokenizer") -> Dict[str, Set[Tuple[str, Tuple]]]:
    """Build a map from token prefix to Set[Tuple[suffix, aligment_token_id, suffix_token_ids]]"""

    # precompute the token ids of all vocab suffixes
    suffixes = list(
        {tok[i:] for tok in tokenizer.vocabulary for i in range(1, len(tok))}
    )
    encoded_suffixes, _ = tokenizer.encode(suffixes)
    encoded_suffixes = [
        [tok for tok in seq_ids if tok != tokenizer.pad_token_id]
        for seq_ids in encoded_suffixes.tolist()
    ]
    suffix_map = dict(zip(suffixes, map(tuple, encoded_suffixes)))
    suffix_map[""] = tuple()

    # compute prefix-suffix map for all tokens, s.t. prefix + suffix = token
    prefix_map = collections.defaultdict(set)
    for token, token_id in tokenizer.vocabulary.items():
        for i in range(1, len(token) + 1):
            prefix_map[token[:i]].add((token[i:], suffix_map[token[i:]]))
    return prefix_map


AlignmentGuideState = collections.namedtuple(
    "AlignmentGuideState", ["legal_path_map", "child_guide_state"]
)


class AlignmentGuide(Guide):
    def __init__(
        self, prompt: str, tokenizer: "Tokenizer", child_guide: Optional[Guide] = None
    ):
        """
        Initialize the AlignmentGuide with a prompt, tokenizer, and an optional child guide.

        Parameters
        ----------
        prompt : str
            The prompt text to be aligned with the generated tokens.
        tokenizer : Tokenizer
            Tokenizer used to align the prompt.
        child_guide : Guide, optional
            A guide to take control after alignment is complete. None -> Unconstrained after alignment
        """
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.child_guide = child_guide

        alignment_seqs, child_guide_ids = self._get_alignment_sequences(
            prompt, tokenizer, child_guide
        )
        alignment_prompt_ids, common_prompt_len = self._get_longest_common_prompt_ids(
            alignment_seqs
        )

        self.alignment_prompt = self.tokenizer.decode(
            [alignment_seqs[0, :common_prompt_len]]
        )[0]

        # calculate map of alignment_prompt continuation tokens -> child_guide advancement tokens
        legal_paths = [
            tuple([t for t in seq if t != tokenizer.pad_token_id])
            for seq in alignment_seqs[:, common_prompt_len:].tolist()
        ]
        legal_path_map = dict(zip(legal_paths, child_guide_ids))

        self.initial_state = AlignmentGuideState(
            legal_path_map=legal_path_map, child_guide_state=None
        )

    @staticmethod
    def _get_alignment_sequences(
        prompt: str, tokenizer: "Tokenizer", child_guide: Optional[Guide] = None
    ):
        """
        Calculate all possible sequences which are valid with a prompt + child_guide
        E.g. prompt="hello wo", child guide accepts "rld" -> tokenization ["hello", "world"] is valid

        Returns tuple of (alignment_seqs, child_guide_ids) of same length
        - alignment_seqs:
            All token sequences which can represent `prompt` + start of generation. The last token
            must represent the end of the prompt can extend beyond the prompt to start generation.
            Sequences are only included if the start of generation portion is legal with child guide.
        - child_guide_ids:
            Token to send to the child guide to simulate the start of generation. In the example above
            "world" is the last alignment seq token, therefore we must advance the state of the child
            guide with the tokenization of "rld" in order to continue generation with the child guide.
        """
        guide_accepts: Dict[
            Tuple[int], bool
        ] = {}  # cache of suffix acceptance for child_guide.accepts()

        # prompts with alignment tokens at end
        aligned_prompt_completions: List[str] = []
        # tokens to feed child guide once alignment completes
        child_guide_ids: List[Tuple] = []

        # compute alignment seqs which are valid with prompt and child guide
        for prefix, alignment_details in build_vocab_prefix_map(tokenizer).items():
            if prompt.endswith(prefix):
                for suffix, suffix_ids in alignment_details:
                    if child_guide is None:
                        aligned_prompt_completions.append(prompt + suffix)
                        child_guide_ids.append(tuple())
                    elif guide_accepts.setdefault(
                        suffix_ids, child_guide.accepts(suffix_ids)
                    ):
                        aligned_prompt_completions.append(prompt + suffix)
                        child_guide_ids.append(suffix_ids)

        alignment_seqs, _ = tokenizer.encode(aligned_prompt_completions)
        return alignment_seqs, child_guide_ids

    @staticmethod
    def _get_longest_common_prompt_ids(alignment_seqs):
        """
        Among all candidate prompt alignment seqs, get the longest shared prefix and their length
        """
        # get longest common prefix among alignment sequences, which will form our alignment prompt
        common = (
            (alignment_seqs.unsqueeze(1) == alignment_seqs.unsqueeze(0))
            .all(0)
            .cumprod(1)
        )
        common_len = common.sum(1).max().item()
        return alignment_seqs[0, :common_len], common_len

    def get_next_instruction(self, state: AlignmentGuideState) -> Instruction:
        """
        Return the next set of valid tokens for generation based on the current state.

        If alignment hasn't completed:
            tokens which continue one of the candidate alignment paths are legal
        If alignment has completed:
            get instruction from the child guide
        """
        if state.legal_path_map is not None:
            return Generate(
                sorted({token_ids[0] for token_ids in state.legal_path_map.keys()})
            )
        elif self.child_guide is None:
            return Generate(None)
        else:
            return self.child_guide.get_next_instruction(state.child_guide_state)

    def get_next_state(
        self, state: AlignmentGuideState, token_id: int
    ) -> AlignmentGuideState:
        """
        Get AlignmentGuideState advanced by token ID.

        If alignment has completed:
            get instruction from the child guide
        If alignment hasn't completed:
            Filter out alignment paths which don't start with token_id
            Remove First token from remaining paths
        If advancing the state completes alignment:
            Advance the child_guide state
        """
        if state.legal_path_map is None:
            if self.child_guide is not None:
                return AlignmentGuideState(
                    legal_path_map=None,
                    child_guide_state=self.child_guide.get_next_state(
                        state.child_guide_state, token_id
                    ),
                )
            else:
                return AlignmentGuideState(None, None)
        else:
            next_state_legal_path_map = {
                key[1:]: value
                for key, value in state.legal_path_map.items()
                if key[0] == token_id
            }
            # if none remaining, advance the child guide
            if not any(next_state_legal_path_map):
                if self.child_guide is not None:
                    child_guide_advancement_ids = next(
                        iter(next_state_legal_path_map.values())
                    )
                    return AlignmentGuideState(
                        legal_path_map=None,
                        child_guide_state=self.child_guide.derive(
                            child_guide_advancement_ids, state.child_guide_state
                        ),
                    )
                else:
                    return AlignmentGuideState(None, None)

            # if paths remaining, return advanced legal_path_map
            else:
                return AlignmentGuideState(
                    legal_path_map=next_state_legal_path_map,
                    child_guide_state=state.child_guide_state,
                )

    def is_final_state(self, state: AlignmentGuideState) -> bool:
        if state.legal_path_map is not None:
            return False
        elif self.child_guide is None:
            return True
        else:
            return self.child_guide.is_final_state(state.child_guide_state)

    def copy(self):
        """AlignmentGuide isn't mutated"""
        return self
