from copy import copy, deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import (
    Any,
    Dict,
    FrozenSet,
    Generator,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import interegular
from interegular.fsm import FSM, Alphabet, OblivionError
from interegular.patterns import Unsupported
from lark import Lark, Token
from lark.common import LexerConf, ParserConf
from lark.exceptions import LexError, UnexpectedInput
from lark.indenter import Indenter
from lark.lexer import (
    BasicLexer,
    ContextualLexer,
    LexerState,
    LexerThread,
    Scanner,
    UnexpectedCharacters,
    UnexpectedToken,
    _create_unless,
)
from lark.parser_frontends import (
    ParsingFrontend,
    PostLexConnector,
    _validate_frontend_args,
)
from lark.parsers.lalr_analysis import (
    Action,
    IntParseTable,
    LALR_Analyzer,
    ParseTable,
    Shift,
)
from lark.parsers.lalr_interactive_parser import InteractiveParser
from lark.parsers.lalr_parser import LALR_Parser, ParseConf, ParserState, _Parser
from outlines_core.fsm.regex import (
    BetterFSM,
    get_token_transition_keys,
    make_deterministic_fsm,
)

PartialParseState = Tuple[str, int]
ParseStateType = Union[int, FrozenSet]


@dataclass
class PartialTerminalInfo:
    priority: int
    terminal_name: str
    can_transition: bool
    is_final: bool


@dataclass
class PartialTokensInfo:
    fsm_state_seq: Tuple[int, ...]
    is_not_finished: bool
    terminals_and_info: Tuple[PartialTerminalInfo, ...]
    final_terminals_and_info: Tuple[PartialTerminalInfo, ...]


class PartialParserConf(ParserConf):
    __serialize_fields__ = (
        "rules",
        "start",
        "parser_type",
        "deterministic",
        "use_value_stack",
    )

    def __init__(self, rules, callbacks, start, deterministic, use_value_stack):
        super().__init__(rules, callbacks, start)
        self.deterministic = deterministic
        self.use_value_stack = use_value_stack


class PartialLark(Lark):
    __serialize_fields__ = (
        "parser",
        "rules",
        "options",
        "deterministic",
        "use_value_stack",
    )

    def __init__(self, grammar, **options):
        # TODO: Could've extended `LarkOptions`, but all these extensions are
        # already way too much (and brittle).  This library really needs a
        # complete refactoring.
        self.deterministic = options.pop("deterministic", False)
        self.use_value_stack = options.pop("use_value_stack", False)
        options["regex"] = True
        super().__init__(grammar, **options)
        assert self.options.parser == "lalr"

    def _build_lexer(self, dont_ignore: bool = False) -> "PartialBasicLexer":
        lexer_conf = self.lexer_conf
        if dont_ignore:
            from copy import copy

            lexer_conf = copy(lexer_conf)
            lexer_conf.ignore = ()

        return PartialBasicLexer(lexer_conf)

    def _build_parser(self) -> "PartialParsingFrontend":
        self._prepare_callbacks()
        _validate_frontend_args(self.options.parser, self.options.lexer)
        parser_conf = PartialParserConf(
            self.rules,
            self._callbacks,
            self.options.start,
            self.deterministic,
            self.use_value_stack,
        )

        # This is `_construct_parsing_frontend` expanded/inlined
        parser_type = self.options.parser
        lexer_type = self.options.lexer
        lexer_conf = self.lexer_conf

        assert isinstance(lexer_conf, LexerConf)
        assert isinstance(parser_conf, ParserConf)
        parser_conf.parser_type = parser_type
        self.lexer_conf.lexer_type = lexer_type
        return PartialParsingFrontend(lexer_conf, parser_conf, self.options)

    def __repr__(self):
        return "{}(open({!r}), parser={!r}, lexer={!r}, ...)".format(
            type(self).__name__,
            self.source_path,
            self.options.parser,
            self.options.lexer,
        )

    def parse_from_state(self, parse_state: "PartialParseState", is_end=False):
        return self.parser.parser.parser.parse_from_state(parse_state, is_end=is_end)


class PartialLexerThread(LexerThread):
    def __copy__(self):
        return type(self)(copy(self.lexer), copy(self.state))

    def __repr__(self):
        return f"{type(self).__name__}(lexer={self.lexer!r}, state={self.state!r})"


class PartialPostLexConnector(PostLexConnector):
    def __copy__(self):
        return type(self)(self.lexer, copy(self.postlexer))

    def __repr__(self):
        return (
            f"{type(self).__name__}(lexer={self.lexer!r}, postlexer={self.postlexer!r})"
        )


class PartialParsingFrontend(ParsingFrontend):
    def __init__(self, lexer_conf, parser_conf, options, parser=None):
        assert parser_conf.parser_type == "lalr"

        options._plugins["LALR_Parser"] = PartialLALRParser
        options._plugins["BasicLexer"] = PartialBasicLexer
        options._plugins["ContextualLexer"] = PartialContextualLexer
        options._plugins["LexerThread"] = PartialLexerThread

        super().__init__(lexer_conf, parser_conf, options, parser=parser)

        if lexer_conf.postlex:
            self.lexer = PartialPostLexConnector(self.lexer.lexer, lexer_conf.postlex)

        self._termset_fsm_info = None
        self._symbols_to_states: Optional[
            Dict[str, Set[Tuple[ParseStateType, Action]]]
        ] = None
        self._reverse_shifts: Optional[
            Dict[ParseStateType, Dict[str, Set[ParseStateType]]]
        ] = None
        # self._state_transition_map: Optional[
        #     Dict[Tuple[ParseStateType, str], Set[ParseStateType]]
        # ] = None

    def _compute_maps(
        self,
    ):
        """Compute state transition and symbols-to-states maps."""
        self._reverse_shifts = {}
        self._symbols_to_states = {}

        parse_table = self.parser.parser.parse_table

        for from_state, symbols_to_ops in parse_table.states.items():
            for symbol, op in symbols_to_ops.items():
                if op[0] == Shift:
                    symbols_to_from_states = self._reverse_shifts.setdefault(op[1], {})
                    symbols_to_from_states.setdefault(symbol, set()).add(from_state)
                self._symbols_to_states.setdefault(symbol, set()).add((from_state, op))

        # # TODO: This approach is very wasteful.
        # context_lexer = get_contextual_lexer(self)
        # self._state_transition_map = {}
        #
        # for from_state, transitions in parse_table.states.items():
        #     for symbol, action in transitions.items():
        #         # TODO: Filter non-terminals
        #         if symbol not in context_lexer.root_lexer.terminals_by_name:
        #             continue
        #
        #         if action[0] is Shift:
        #             self._state_transition_map.setdefault(
        #                 (from_state, symbol), set()
        #             ).add(action[1])
        #             continue
        #
        #         antecedent_state_seqs = parse_to_terminal(self, [(from_state,)], symbol)
        #
        #         for antecedent_state_seq in antecedent_state_seqs:
        #             antecedent_state = antecedent_state_seq[-1]
        #             self._state_transition_map.setdefault(
        #                 (from_state, symbol), set()
        #             ).add(antecedent_state)

    def _compute_termset_fsm_info(self):
        """Collect and return information about terminal symbol sets and their FSMs.

        Terminal symbol sets (or "termsets") are ordered sequences of terminal
        symbols that are used by each parser state.  Associated with each is a
        collection of FSMs for each terminal and a single parse state FSM that is
        the union of each terminal's FSM.

        This constructs a list of tuples containing the termset, the set of
        parse states that use the termsets, parse state FSMs, and information
        mapping the components of the parse state FSMs to their terminal symbol
        FSMs.

        """
        context_lexer = get_contextual_lexer(self)
        termsets_to_fsms = {}
        termsets_to_parse_states: Dict[Tuple[str, ...], Set[ParseStateType]] = {}
        for parse_state, lexer in context_lexer.lexers.items():
            scanner = lexer.scanner
            key = tuple(term.name for term in scanner.terminals)
            termsets_to_fsms[key] = (scanner.fsm, scanner.fsms_to_trans_finals)
            termsets_to_parse_states.setdefault(key, set()).add(parse_state)

        self._termset_fsm_info = [
            (
                termset,
                frozenset(termsets_to_parse_states[termset]),
                fsm,
                fsms_to_trans_finals,
            )
            for termset, (fsm, fsms_to_trans_finals) in termsets_to_fsms.items()
        ]

    @property
    def termset_fsm_info(self):
        if self._termset_fsm_info is None:
            self._compute_termset_fsm_info()
        return self._termset_fsm_info

    @property
    def symbols_to_states(self):
        if self._symbols_to_states is None:
            self._compute_maps()
        return self._symbols_to_states

    @property
    def reverse_shifts(self):
        if self._reverse_shifts is None:
            self._compute_maps()
        return self._reverse_shifts

    # @property
    # def state_transition_map(self):
    #     if self._state_transition_map is None:
    #         self._compute_maps()
    #     return self._state_transition_map


class PartialLALRParser(LALR_Parser):
    def __init__(self, parser_conf, debug=False, strict=False):
        analysis = LALR_Analyzer(
            parser_conf, debug=debug if not parser_conf.deterministic else True
        )
        analysis.compute_lalr()
        callbacks = parser_conf.callbacks

        self.parser_conf = parser_conf
        self._parse_table = analysis.parse_table

        if parser_conf.deterministic:
            old_to_new = {}

            def to_tuple(v):
                new = old_to_new.get(v)
                if new is None:
                    new = tuple(sorted(v, key=lambda y: str(y)))
                    old_to_new[v] = new
                return new

            enum = sorted(
                self._parse_table.states.keys(),
                key=lambda x: str(sorted(x, key=lambda y: str(y))),
            )

            new_states = {}
            for s in enum:
                transitions = {
                    term: op if op[0] is not Shift else (op[0], to_tuple(op[1]))
                    for term, op in self._parse_table.states[s].items()
                }
                new_states[to_tuple(s)] = transitions

            self._parse_table = type(self._parse_table)(
                new_states,
                {k: to_tuple(v) for k, v in self._parse_table.start_states.items()},
                {k: to_tuple(v) for k, v in self._parse_table.end_states.items()},
            )

            if not debug:
                self._parse_table = IntParseTable.from_ParseTable(self._parse_table)
                self.states_to_rulesets = dict(
                    zip(self._parse_table.states.keys(), new_states.keys())
                )

        self.parser = PartialParser(
            self._parse_table,
            callbacks,
            debug,
            use_value_stack=parser_conf.use_value_stack,
        )

    @classmethod
    def deserialize(cls, data, memo, callbacks, debug=False):
        inst = cls.__new__(cls)
        inst._parse_table = ParseTable.deserialize(data, memo)
        inst.parser = PartialParser(inst._parse_table, callbacks, debug)
        return inst


class PartialParserState(ParserState):
    __slots__ = "use_value_stack"

    def __init__(
        self,
        parse_conf,
        lexer,
        state_stack=None,
        value_stack=None,
        use_value_stack=False,
    ):
        super().__init__(
            parse_conf, lexer, state_stack=state_stack, value_stack=value_stack
        )
        self.use_value_stack = use_value_stack

    def feed_token(self, token, is_end=False):
        if token.type == "partial":
            # If none of the potential terminals can transition, we need to know now
            current_state = self.state_stack[-1]
            current_lexer = get_contextual_lexer(self.lexer).lexers[current_state]

            # We have to feed the token and determine whether or not at least
            # one terminal is consistent with the stack; otherwise, we'll miss
            # invalid REDUCE cases.
            # TODO: We should track separate parses conditional on possible
            # token/symbol types, then we can coherently reuse the following
            # results instead of recomputing it later.
            can_transition = False
            for terminal_info in token.value.terminals_and_info:
                if terminal_info.terminal_name not in current_lexer.ignore_types:
                    test_token = Token.new_borrow_pos(
                        terminal_info.terminal_name, "", token
                    )

                    stack = copy(self.state_stack)
                    try:
                        self.feed_token_no_stack(test_token, is_end=is_end)
                        can_transition = True
                        break
                    except UnexpectedToken:
                        continue
                    finally:
                        self.state_stack = stack
                else:
                    can_transition = True

            if not can_transition:
                expected = {
                    s
                    for s in self.parse_conf.states[current_state].keys()
                    if s.isupper()
                }
                raise UnexpectedToken(
                    token, expected, state=self, interactive_parser=None
                )

        elif self.use_value_stack:
            super().feed_token(token, is_end=is_end)
        else:
            self.feed_token_no_stack(token, is_end=is_end)

    def feed_token_no_stack(self, token, is_end=False):
        """
        This is a copy of `ParserState.feed_token` with all the value stack
        steps removed.  Since we're not exactly parsing in order to obtain a
        CST or anything similar, we can avoid the growing expense of tracking
        the parse tree.
        """
        state_stack = self.state_stack
        states = self.parse_conf.states
        end_state = self.parse_conf.end_state

        while True:
            state = state_stack[-1]
            try:
                action, arg = states[state][token.type]
            except KeyError:
                expected = {s for s in states[state].keys() if s.isupper()}
                raise UnexpectedToken(
                    token, expected, state=self, interactive_parser=None
                )

            assert arg != end_state

            if action is Shift:
                # shift once and return
                assert not is_end
                state_stack.append(arg)
                return
            else:
                # reduce+shift as many times as necessary
                rule = arg
                size = len(rule.expansion)
                if size:
                    del state_stack[-size:]

                _action, new_state = states[state_stack[-1]][rule.origin.name]
                assert _action is Shift
                state_stack.append(new_state)

                if is_end and state_stack[-1] == end_state:
                    return

    def feed_eof(self):
        last_token = self.lexer.state.last_token

        if last_token is None:
            eof_token = self.lexer._Token("$END", "", 0, 1, 1)
        else:
            eof_token = Token.new_borrow_pos("$END", "", last_token)

        new_token_is_legal = (
            last_token is None
            or last_token.type != "partial"
            or any(ti.is_final for ti in last_token.value.terminals_and_info)
        )
        if new_token_is_legal:
            self.feed_token(eof_token, is_end=True)
        else:
            raise UnexpectedToken(eof_token, [], state=self, interactive_parser=None)

    def choices(self):
        return self.parse_conf.parse_table.states[self.position]

    def accepts(self):
        """
        Adapted from https://github.com/lark-parser/lark/blob/be542c2ff6d968817df019b8bf03f37b3111c08c/lark/parsers/lalr_interactive_parser.py#L95
        Returns the set of possible tokens that will advance the parser into a new valid state.
        """
        accepts = set()
        conf_no_callbacks = copy(self.parse_conf)
        # We don't want to call callbacks here since those might have arbitrary side effects
        # and are unnecessarily slow.
        conf_no_callbacks.callbacks = {}
        for t in self.choices():
            if t.isupper():  # is terminal?
                new_state = copy(self)
                new_state.parse_conf = conf_no_callbacks
                try:
                    new_state.feed_token(new_state.lexer._Token(t, ""))
                except UnexpectedToken:
                    pass
                else:
                    accepts.add(t)
        return accepts

    def __copy__(self):
        return type(self)(
            self.parse_conf,
            copy(self.lexer),
            copy(self.state_stack),
            deepcopy(self.value_stack),
            use_value_stack=self.use_value_stack,
        )

    def __repr__(self):
        return f"{type(self).__name__}(lexer={self.lexer!r}, state_stack={self.state_stack!r})"


class PartialParser(_Parser):
    def __init__(self, parse_table, callbacks, debug=False, use_value_stack=False):
        super().__init__(parse_table, callbacks, debug=debug)
        self.use_value_stack = use_value_stack

    def parse(
        self, lexer, start, value_stack=None, state_stack=None, start_interactive=False
    ):
        parse_conf = ParseConf(self.parse_table, self.callbacks, start)
        parser_state = PartialParserState(
            parse_conf, copy(lexer), state_stack, value_stack, self.use_value_stack
        )
        if start_interactive:
            return InteractiveParser(self, parser_state, parser_state.lexer)
        return self.parse_from_state(parser_state)

    def parse_from_state(self, state, last_token=None, is_end=False):
        try:
            token = last_token
            for token in state.lexer.lex(state):
                state.feed_token(token)

            if is_end and (not token or token.type != "partial"):
                state.feed_eof()

            return state
        except UnexpectedInput as e:
            try:
                e.interactive_parser = InteractiveParser(self, state, state.lexer)
            except NameError:
                pass
            raise e
        except Exception:
            if self.debug:
                print("")
                print("STATE STACK DUMP")
                print("----------------")
                for i, s in enumerate(state.state_stack):
                    print("%d)" % i, s)
                print("")

            raise


class PartialScanner(Scanner):
    @classmethod
    @lru_cache
    def construct_terminal_fsm(cls, terminal):
        # TODO: This should really be done at the lexer/parser level so that
        # the lifetime of these objects is tied to the parser itself.
        regex_str = terminal.pattern.to_regexp()
        pattern = interegular.parse_pattern(regex_str)
        fsm, _ = make_deterministic_fsm(pattern.to_fsm().reduce())
        return fsm, pattern.prefix_postfix

    def __init__(self, terminals, g_regex_flags, re_, use_bytes, match_whole=False):
        self.terminals = terminals
        self.g_regex_flags = g_regex_flags
        self.use_bytes = use_bytes
        self.match_whole = match_whole
        self.allowed_types = {t.name for t in self.terminals}
        self._mres = None

        fsms = []
        for t in self.terminals:
            fsm, prefix_postfix = self.construct_terminal_fsm(t)

            # TODO FIXME: We don't support this right now.
            assert prefix_postfix == (0, 0)

            fsms.append(fsm)

        self.fsm, self.fsms_to_trans_finals = fsm_union(fsms)

    def get_terminals_info(
        self, fsm_state_seq
    ) -> Tuple[Tuple[PartialTerminalInfo, ...], Tuple[PartialTerminalInfo, ...]]:
        """Get the possible terminal symbols for an FSM state sequence."""
        terminals_and_info: Tuple[PartialTerminalInfo, ...] = ()
        final_terminals_and_info: Tuple[PartialTerminalInfo, ...] = ()
        for i, (fsm_id, fsm_reads_more, in_final) in enumerate(
            get_sub_fsms_from_seq(fsm_state_seq, self.fsms_to_trans_finals)
        ):
            terminal_name = self.terminals[fsm_id].name
            info = PartialTerminalInfo(i, terminal_name, fsm_reads_more, in_final)
            terminals_and_info += (info,)
            if in_final:
                final_terminals_and_info += (info,)

        return terminals_and_info, final_terminals_and_info

    def match(self, text, pos, last_fsm_state_seq: Optional[Tuple[int, ...]] = None):
        """Determine an FSM match over `text` starting at `pos` and continuing `last_fsm_state_seq`."""

        start_pos = pos

        if last_fsm_state_seq:
            assert len(last_fsm_state_seq) > 1
            start_pos += len(last_fsm_state_seq) - 1
            start_state = last_fsm_state_seq[-1]
        else:
            start_state = self.fsm.initial

        text_part = text[start_pos:]

        text_transitions = get_token_transition_keys(
            self.fsm.fsm_info.alphabet_symbol_mapping,
            self.fsm.fsm_info.alphabet_anything_value,
            text_part,
        )

        state_seq = walk_fsm(
            self.fsm,
            text_transitions,
            start_state,
            full_match=self.match_whole,
        )

        if not state_seq:
            return None

        if last_fsm_state_seq:
            res = last_fsm_state_seq + tuple(state_seq)
        else:
            res = (start_state,) + tuple(state_seq)

        return res


class PartialContextualLexer(ContextualLexer):
    def __init__(self, conf: "LexerConf", states, always_accept=()):
        terminals = list(conf.terminals)
        terminals_by_name = conf.terminals_by_name

        trad_conf = copy(conf)
        trad_conf.terminals = terminals

        lexer_by_symbols: Dict = {}
        self.lexers = {}
        for state, accepts in states.items():
            key = frozenset(accepts)
            try:
                lexer = lexer_by_symbols[key]
            except KeyError:
                accepts = set(accepts) | set(conf.ignore) | set(always_accept)
                lexer_conf = copy(trad_conf)
                lexer_conf.terminals = [
                    terminals_by_name[n] for n in accepts if n in terminals_by_name
                ]
                if not lexer_conf.terminals:
                    continue
                lexer = PartialBasicLexer(lexer_conf)
                lexer_by_symbols[key] = lexer

            self.lexers[state] = lexer

        assert trad_conf.terminals is terminals
        self.root_lexer = PartialBasicLexer(trad_conf)

    def lex(self, lexer_state: LexerState, parser_state: Any) -> Iterator[Token]:
        try:
            while True:
                lexer = self.lexers[parser_state.position]
                next_tok = lexer.next_token(lexer_state, parser_state)
                yield next_tok
        except EOFError:
            pass
        except KeyError:
            if len(lexer_state.text) > lexer_state.line_ctr.char_pos:
                raise UnexpectedCharacters(
                    lexer_state.text,
                    lexer_state.line_ctr.char_pos,
                    lexer_state.line_ctr.line,
                    lexer_state.line_ctr.column,
                    allowed=False,
                    token_history=lexer_state.last_token and [lexer_state.last_token],
                    state=parser_state,
                    terminals_by_name=self.root_lexer.terminals,
                )


class PartialBasicLexer(BasicLexer):
    def __init__(self, conf: "LexerConf"):
        super().__init__(conf)
        # Eagerly construct the scanner
        self._build_scanner()

    def _build_scanner(self):
        # This seems incredibly convoluted: `lark` creates callback-triggered
        # nested scanners for regex-defined terminals that overlap with
        # string-defined terminals when both types of terminals have the same
        # priority.  Unless I'm missing something important, why not simply
        # reorder the terminals so that the string-defined ones come before the
        # regex-defined ones?
        terminals, self.callback = _create_unless(
            self.terminals, self.g_regex_flags, self.re, self.use_bytes
        )

        # We can't let people arbitrarily mess with the scanning process.
        assert not self.user_callbacks
        # for type_, f in self.user_callbacks.items():
        #     if type_ in self.callback:
        #         # Already a callback there, probably UnlessCallback
        #         self.callback[type_] = CallChain(
        #             self.callback[type_], f, lambda t: t.type == type_
        #         )
        #     else:
        #         self.callback[type_] = f

        # We used the "callback" results to reorder the terminals (see the
        # comments above).
        for terminal_name, callback in self.callback.items():
            terminal = self.terminals_by_name[terminal_name]
            for sub_terminal in callback.scanner.terminals:
                self.terminals.remove(sub_terminal)
                idx = self.terminals.index(terminal)
                self.terminals.insert(idx, sub_terminal)

        self._scanner = PartialScanner(
            self.terminals, self.g_regex_flags, self.re, self.use_bytes
        )

    def match(self, text, pos, last_fsm_state_seq=None):
        return self.scanner.match(text, pos, last_fsm_state_seq)

    def next_token(self, lex_state: LexerState, parser_state: Any = None) -> Token:
        last_token = lex_state.last_token

        last_fsm_state_seq = None
        if last_token and last_token.type == "partial":
            # Continue from last partial lexer state
            last_fsm_state_seq = last_token.value.fsm_state_seq

        line_ctr = lex_state.line_ctr
        end_pos = line_ctr.char_pos + (
            len(last_fsm_state_seq) - 1 if last_fsm_state_seq else 0
        )
        while end_pos < len(lex_state.text):
            res = self.match(lex_state.text, line_ctr.char_pos, last_fsm_state_seq)

            if not res:
                if (
                    not last_fsm_state_seq
                    or last_fsm_state_seq[-1] not in self.scanner.fsm.finals
                ):
                    allowed = self.scanner.allowed_types - self.ignore_types
                    if not allowed:
                        allowed = {"<END-OF-FILE>"}
                    raise UnexpectedCharacters(
                        lex_state.text,
                        line_ctr.char_pos,
                        line_ctr.line,
                        line_ctr.column,
                        allowed=allowed,
                        token_history=lex_state.last_token and [lex_state.last_token],
                        state=parser_state,
                        terminals_by_name=self.terminals_by_name,
                    )

                # The partial match might be complete now
                fsm_state_seq = last_token.value.fsm_state_seq
                terminals_and_info = last_token.value.terminals_and_info
                final_terminals_and_info = last_token.value.final_terminals_and_info
            else:
                fsm_state_seq = res
                (
                    terminals_and_info,
                    final_terminals_and_info,
                ) = self.scanner.get_terminals_info(fsm_state_seq)

            priority_terminal_info = (
                final_terminals_and_info[0]
                if final_terminals_and_info
                else terminals_and_info[0]
            )

            is_not_finished = (
                not priority_terminal_info.is_final
                or priority_terminal_info.can_transition
                or len(terminals_and_info) > 1
            )

            start_pos = line_ctr.char_pos
            end_pos = start_pos + len(fsm_state_seq) - 1

            if end_pos >= len(lex_state.text) and is_not_finished:
                type_name = "partial"
                token_value = PartialTokensInfo(
                    fsm_state_seq,
                    is_not_finished,
                    terminals_and_info,
                    final_terminals_and_info,
                )
                # Don't update the line counter states until we've finished
                value = ""
            else:
                type_name = priority_terminal_info.terminal_name
                # The token value should contain all partial scan parts in this
                # case
                value = token_value = lex_state.text[start_pos:end_pos]

            assert isinstance(self.callback, Dict)

            if type_name not in self.ignore_types:
                t = Token(
                    type_name,
                    token_value,
                    line_ctr.char_pos,
                    line_ctr.line,
                    line_ctr.column,
                )

                line_ctr.feed(value, type_name in self.newline_types)

                t.end_line = line_ctr.line
                t.end_column = line_ctr.column
                t.end_pos = line_ctr.char_pos
                if t.type in self.callback:
                    t = self.callback[t.type](t)
                    if not isinstance(t, Token):
                        raise LexError(
                            "Callbacks must return a token (returned %r)" % t
                        )
                lex_state.last_token = t
                return t

            if type_name in self.callback:
                t2 = Token(
                    type_name, value, line_ctr.char_pos, line_ctr.line, line_ctr.column
                )
                self.callback[type_name](t2)

            line_ctr.feed(value, type_name in self.newline_types)

            last_fsm_state_seq = None

        raise EOFError(self)


class PartialIndenter(Indenter):
    """An `Indenter` that doesn't reset its state every time `process` is called."""

    def process(self, stream):
        return self._process(stream)

    def _process(self, stream):
        for token in stream:
            # These were previously *after* the `yield`, but that makes the
            # state tracking unnecessarily convoluted.
            if token.type in self.OPEN_PAREN_types:
                self.paren_level += 1
            elif token.type in self.CLOSE_PAREN_types:
                self.paren_level -= 1
                if self.paren_level < 0:
                    raise UnexpectedToken(token, [])

            if token.type == self.NL_type:
                yield from self.handle_NL(token)
            else:
                yield token

        # TODO: What do we want to do here?
        # while len(self.indent_level) > 1:
        #     self.indent_level.pop()
        #     yield Token(self.DEDENT_type, "")

    def accepts_token_type(self, token_type):
        if token_type in self.CLOSE_PAREN_types and self.paren_level - 1 < 0:
            return False

        # TODO:
        # if token_type == self.NL_type and self.paren_level == 0:
        #     ...
        #     return False

        return True

    def __copy__(self):
        res = type(self)()
        res.paren_level = self.paren_level
        res.indent_level = copy(self.indent_level)
        return res

    def __repr__(self):
        return f"{type(self).__name__}(paren_level={self.paren_level!r}, indent_level={self.indent_level!r})"


class PartialPythonIndenter(PartialIndenter):
    NL_type = "_NEWLINE"
    OPEN_PAREN_types = ["LPAR", "LSQB", "LBRACE"]
    CLOSE_PAREN_types = ["RPAR", "RSQB", "RBRACE"]
    INDENT_type = "_INDENT"
    DEDENT_type = "_DEDENT"
    tab_len = 8


def get_contextual_lexer(x: Union[PartialLexerThread, PartialParsingFrontend]):
    if isinstance(x.lexer, ContextualLexer):
        return x.lexer
    else:
        return x.lexer.lexer


def terminals_to_fsms(lp: PartialLark) -> Dict[str, FSM]:
    """Construct a ``dict`` mapping terminal symbol names to their finite state machines."""

    symbol_names_and_fsms = {}
    for terminal in lp.terminals:
        pattern = interegular.parse_pattern(terminal.pattern.to_regexp())
        # TODO: Use `pyparser.terminals[0].pattern.flags`?
        try:
            fsm, _ = make_deterministic_fsm(pattern.to_fsm().reduce())
        except Unsupported:
            fsm = None

        symbol_names_and_fsms[terminal.name] = fsm

    return symbol_names_and_fsms


def fsm_union(
    fsms: Sequence[FSM],
) -> Tuple[FSM, Dict[int, Tuple[Set[Tuple[int, int]], Set[int], Dict[int, Set[int]]]]]:
    """Construct an FSM representing the union of the FSMs in `fsms`.

    This is an updated version of `interegular.fsm.FSM.union` made to return an
    extra map of component FSMs to the sets of state transitions that
    correspond to them in the new FSM.

    """

    alphabet, new_to_old = Alphabet.union(*[fsm.alphabet for fsm in fsms])

    indexed_fsms = tuple(enumerate(fsms))

    initial = {i: fsm.initial for (i, fsm) in indexed_fsms}

    # Dedicated function accepting a "superset" and returning the next
    # "superset" obtained by following this transition in the new FSM
    def follow(current_state, new_transition: int):
        next = {}
        for i, f in indexed_fsms:
            old_transition = new_to_old[i][new_transition]
            if (
                i in current_state
                and current_state[i] in f.map
                and old_transition in f.map[current_state[i]]
            ):
                next[i] = f.map[current_state[i]][old_transition]
        if not next:
            raise OblivionError
        return next

    states = [initial]
    finals: Set[int] = set()
    map: Dict[int, Dict[int, int]] = {}

    # Map component FSMs to their new state-to-state transitions, finals, and a
    # map translating component FSM states to aggregate FSM states
    fsms_to_trans_finals: Dict[
        int, Tuple[Set[Tuple[int, int]], Set[int], Dict[int, Set[int]]]
    ] = {}

    i = 0
    while i < len(states):
        state = states[i]

        # Add to the finals of the aggregate FSM whenever we hit a final in a
        # component FSM
        if any(state.get(j, -1) in fsm.finals for (j, fsm) in indexed_fsms):
            finals.add(i)

        # Compute the map for this state
        map[i] = {}
        for transition in alphabet.by_transition:
            try:
                next = follow(state, transition)
            except OblivionError:
                # Reached an oblivion state; don't list it
                continue
            else:
                try:
                    # TODO: Seems like this could--and should--be avoided
                    j = states.index(next)
                except ValueError:
                    j = len(states)
                    states.append(next)

                map[i][transition] = j

                for fsm_id, fsm_state in next.items():
                    (
                        fsm_transitions,
                        fsm_finals,
                        fsm_old_to_new,
                    ) = fsms_to_trans_finals.setdefault(fsm_id, (set(), set(), {}))
                    old_from = state[fsm_id]
                    old_to = fsm_state
                    fsm_old_to_new.setdefault(old_from, set()).add(i)
                    fsm_old_to_new.setdefault(old_to, set()).add(j)
                    fsm_transitions.add((i, j))
                    if fsm_state in fsms[fsm_id].finals:
                        fsm_finals.add(j)

        i += 1

    fsm = FSM(
        alphabet=alphabet,
        states=range(len(states)),
        initial=0,
        finals=finals,
        map=map,
        __no_validation__=True,
    )

    fsm, old_to_new_states = make_deterministic_fsm(fsm)
    _fsms_to_trans_finals = {
        fsm_id: (
            {(old_to_new_states[s1], old_to_new_states[s2]) for s1, s2 in transitions},
            {old_to_new_states[s] for s in finals},
            {
                old_state: {old_to_new_states[new_state] for new_state in new_states}
                for old_state, new_states in old_to_new.items()
            },
        )
        for fsm_id, (transitions, finals, old_to_new) in sorted(
            fsms_to_trans_finals.items(), key=lambda x: x[0]
        )
    }

    return (
        fsm,
        _fsms_to_trans_finals,
    )


def get_sub_fsms_from_seq(
    state_seq: Sequence[int],
    fsms_to_trans_finals: Dict[
        int, Tuple[Set[Tuple[int, int]], Set[int], Dict[int, Set[int]]]
    ],
) -> Generator[Tuple[int, bool, bool], None, None]:
    """Get the indices of the sub-FSMs in `fsm` that could have matched the state sequence `state_seq`.

    Parameters
    ----------
    state_seq
        A state sequence.
    fsms_to_trans_finals
        A map from FSM indices to tuples containing sets of their state transitions
        and sets of the final/accept states.

    Returns
    -------
    A generator returning tuples containing each sub-FSM index (in the order
    they were union-ed to construct `fsm`) and booleans indicating whether or
    not there is another valid transition from the last state in the sequence
    for the associated sub-FSM (i.e. if the FSM can continue
    accepting/matching) and whether or not the sequence ends in a final state
    of the sub-FSM.
    """
    state_seq_transitions = set(zip(state_seq[:-1], state_seq[1:]))
    last_fsm_state = state_seq[-1]
    yield from (
        (
            # The sub-FMS index
            fsm_idx,
            # Is there another possible transition in this sub-FSM?
            any(last_fsm_state == from_s for (from_s, to_s) in transitions),
            # Is this sub-FSM in a final state?
            state_seq[-1] in finals,
        )
        for fsm_idx, (transitions, finals, _) in fsms_to_trans_finals.items()
        if state_seq_transitions.issubset(transitions)
    )


def walk_fsm(
    fsm: BetterFSM,
    token_transition_keys: Sequence[int],
    start_state: int,
    full_match: bool = True,
) -> List[int]:
    fsm_finals = fsm.finals

    state = start_state
    accepted_states: List[int] = []
    last_final_idx: int = 0

    fsm_transitions = fsm.flat_transition_map

    # Iterate over token transition key sequence. The transition key
    # sequence represents the FSM traversal rules of the tokens symbols.
    for i, trans_key in enumerate(token_transition_keys):
        new_state = fsm_transitions.get((state, trans_key))

        if new_state is None:
            if not full_match and last_final_idx > 0:
                return accepted_states[:last_final_idx]

            return []

        state = new_state

        if state in fsm_finals:
            last_final_idx = i + 1

        accepted_states.append(state)

    if full_match and last_final_idx - 1 != i:
        return []

    return accepted_states
