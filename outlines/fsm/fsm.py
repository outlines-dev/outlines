import functools
from typing import TYPE_CHECKING, List, NewType, Protocol, Tuple

import interegular
from lark import Lark

# from outlines.fsm.parsing import PartialLark
from outlines import grammars
from outlines.caching import cache
from outlines.fsm.regex import create_fsm_index_tokenizer, make_deterministic_fsm

if TYPE_CHECKING:
    from outlines.models.tokenizer import Tokenizer

FSMState = NewType("FSMState", int)


class FSM(Protocol):
    first_state: FSMState = FSMState(0)
    final_state: FSMState = FSMState(-1)

    def is_final_state(self, state: FSMState) -> bool:
        """Determine whether the current state of the FSM is a final state."""
        return state == self.final_state

    def allowed_token_ids(self, state: FSMState) -> List[int]:
        ...

    def next_state(self, state: FSMState, token_id: int) -> FSMState:
        ...

    def copy(self) -> "FSM":
        ...


class StopAtEosFSM(FSM):
    """FSM to generate text until EOS has been generated."""

    def __init__(self, tokenizer: "Tokenizer"):
        self.eos_token_id = tokenizer.eos_token_id
        self.vocabulary = tokenizer.vocabulary.values()

    def allowed_token_ids(self, state: FSMState) -> List[int]:
        """Generate a list of allowed tokens for the next step.

        When in the initial state we allow every token to be generated.
        In the final state the only allowed token is `stop_token_id`.

        Parameters
        ----------
        state
            The current state of the FSM.

        Returns
        -------
        A list that contains the tokens to mask.

        """
        if self.is_final_state(state):
            return [self.eos_token_id]
        return list(self.vocabulary)

    def next_state(self, state: FSMState, token_id: int) -> FSMState:
        """Update the state of the FSM.

        The FSM stays in the initial state `0` unless the specified stop token
        has been generated or the maximum number of tokens has been reached. In
        which case the FSM moves to the final state `-1`.

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
        if token_id == self.eos_token_id or state == self.final_state:
            return self.final_state

        return self.first_state

    def copy(self) -> "StopAtEosFSM":
        """Create a copy of the FSM."""
        return self


@functools.lru_cache(maxsize=1024)
class RegexFSM(FSM):
    """FSM to generate text that is in the language of a regular expression."""

    def __init__(self, regex_string: str, tokenizer):
        @cache()
        def create_states_mapping(
            regex_string: str, cacheable_vocabulary: Tuple[Tuple[str, int], ...]
        ) -> Tuple[dict, set]:
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

            return states_to_token_maps, empty_token_ids

        self.states_to_token_maps, self.empty_token_ids = create_states_mapping(
            regex_string, tuple(sorted(tokenizer.vocabulary.items()))
        )
        self.vocabulary = list(tokenizer.vocabulary.values())
        self.eos_token_id = tokenizer.eos_token_id

    def allowed_token_ids(self, state: FSMState) -> List[int]:
        """Generate a list of allowed tokens for the next step.

        The initialization of the FSM builds an index which maps FSM states to a
        map from authorized tokens to the state in which the FSM needs to move
        if said token is generated. Therefore the authorized tokens at the
        current state are the keys of the map returned by the value of the index
        for current state.

        If the current state is not contained in the end this means that we are
        in a final state of the FSM. We only authorize EOS tokens in the final
        state.

        Parameters
        ----------
        state
            The current state of the FSM.

        Returns
        -------
        A list that contains the tokens to mask.

        """
        next_tokens_to_end_states = self.states_to_token_maps.get(state)

        if next_tokens_to_end_states is None:
            return [self.eos_token_id]
        else:
            return list(next_tokens_to_end_states.keys())

    def next_state(self, state: FSMState, token_id: int) -> FSMState:
        """Update the state of the FSM.

        We use the index to determine to which state the FSM should transition
        given the token that was just generated.

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
        if token_id == self.eos_token_id or state == self.final_state:
            return self.final_state

        last_token_to_end_state = self.states_to_token_maps[state]
        next_state = last_token_to_end_state.get(token_id)
        if next_state is None:
            return self.final_state

        return FSMState(next_state)

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

    def copy(self) -> "RegexFSM":
        """Create a copy of the FSM."""
        return self


class CFGFSM(FSM):
    """FSM to generate text that is in the language of a context-free grammar."""

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
        self.regex_fsm: RegexFSM

        self.check_last = False
        self.proposal_last: List[int] = []
        self.regex_fsm_last: RegexFSM

    def allowed_token_ids(self, state: FSMState) -> List[int]:
        """Generate a list of allowed tokens for the next step.

        Upon initialization, the CFG incremental parser is used to determine the
        first regex and construct the first FSM to generate the first terminal.

        This FSM is used for proposals until either:

        - The FSM is exhausted, and its only remaining option is the EOS
          token, in which case we feed the generated terminal to the
          CFG incremental parser and allow it to propose the next regex
          corresponding to the next set of valid terminals.
        - The current FSM can be exhausted, but the EOS token is not the only
          remaining option. In this case we allow proposal of current terminal extensions,
          store the current FSM and its state, then also use the CFG parser
          to propose a new regex corresponding to terminating the current terminal
          and starting the next one. The model can then sample from either of these sets
          to determine whether to extend the current terminal or terminate it and start the next one.

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
            return [self.tokenizer.eos_token_id]

        proposal = []
        if self.generation != "":
            if self.check_last:
                proposer = self.regex_fsm_last
            else:
                proposer = self.regex_fsm
            proposal += proposer.allowed_token_ids(state)
            if self.tokenizer.eos_token_id not in proposal:
                return proposal
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
                return [self.tokenizer.eos_token_id]
            self.allow_eos = True
            options.add("")
            assert len(options) > 1

        regex_string = r"(" + r"|".join([r"(" + x + r")" for x in options]) + r")"
        self.regex_fsm = RegexFSM(regex_string, self.tokenizer)
        self.reset_state = True

        proposal += self.regex_fsm.allowed_token_ids(self.first_state)
        if self.allow_eos:
            self.allow_eos = False
        else:
            proposal = [x for x in proposal if x != self.tokenizer.eos_token_id]
            assert len(proposal) > 0
        return proposal

    def next_state(self, state: FSMState, token_id: int) -> FSMState:
        """Update the state of the FSM.

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
        if token_id == self.tokenizer.eos_token_id or state == self.final_state:
            return self.final_state

        self.generation += self.tokenizer.decode([token_id])[0]

        if self.check_last:
            if token_id in self.proposal_last:
                return self.regex_fsm_last.next_state(state, token_id)
            self.check_last = False

        if self.reset_state:
            self.reset_state = False
            state = self.first_state

        return self.regex_fsm.next_state(state, token_id)

    def copy(self) -> "CFGFSM":
        """Create a copy of the FSM."""
        return CFGFSM(self.cfg_string, self.tokenizer)
