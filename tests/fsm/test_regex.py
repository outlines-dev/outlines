import interegular
import numba
import numpy as np
import pytest
from transformers import AutoTokenizer

from outlines.fsm.regex import (
    _walk_fsm,
    create_fsm_index_end_to_end,
    create_fsm_index_tokenizer,
    fsm_union,
    get_sub_fsms_from_seq,
    get_token_transition_keys,
    get_vocabulary_transition_keys,
    make_byte_level_better_fsm,
    make_byte_level_fsm,
    make_deterministic_fsm,
    reduced_vocabulary,
    walk_fsm,
)
from outlines.models.transformers import TransformerTokenizer
from outlines.models.vllm import adapt_tokenizer


def identity(s):
    return s


def to_bytes(s):
    return [chr(b) if b < 0x80 else f"\x00{b:02X}" for b in s.encode("utf-8")]


def merge_symbols(byte_hexs):
    return "".join(["\x00" + b if len(b) == 2 else b for b in byte_hexs])


def token_str_to_trans_key(fsm, input_string):
    return get_token_transition_keys(
        fsm.fsm_info.alphabet_symbol_mapping,
        fsm.fsm_info.alphabet_anything_value,
        input_string,
    )


def walk_fsm_from_token_str(
    fsm,
    input_string: str,
    start_state: int,
    full_match: bool = True,
):
    return walk_fsm(
        fsm,
        token_str_to_trans_key(fsm, input_string),
        start_state,
        full_match,
    )


def walk_fsm_from_token_str_numba(
    fsm,
    input_string: str,
    start_state: int,
    full_match: bool = True,
):
    return _walk_fsm(
        fsm.fsm_info.transitions,
        fsm.fsm_info.initial,
        fsm.fsm_info.finals,
        token_str_to_trans_key(fsm, input_string),
        start_state,
        full_match=full_match,
    )


@pytest.mark.parametrize(
    "function",
    [
        walk_fsm_from_token_str,
        walk_fsm_from_token_str_numba,
    ],
)
def test_walk_fsm(function):
    regex_pattern = interegular.parse_pattern("0|[1-9][2-9]*")
    regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm().reduce())

    res = tuple(function(regex_fsm, "0", regex_fsm.initial, full_match=True))
    assert res == (1,)

    res = tuple(function(regex_fsm, "00", regex_fsm.initial, full_match=False))
    assert res == (1,)

    res = tuple(function(regex_fsm, "!", regex_fsm.initial, full_match=True))
    assert res == tuple()

    res = tuple(function(regex_fsm, "00", regex_fsm.initial, full_match=True))
    assert res == tuple()

    # This should fail, because state `1` reads nothing
    res = tuple(function(regex_fsm, "0", 1, full_match=True))
    assert res == tuple()

    regex_pattern = interegular.parse_pattern("0|[1-9][2-9]+")
    regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm().reduce())

    res = tuple(function(regex_fsm, "1", regex_fsm.initial, full_match=True))
    assert res == tuple()

    res = tuple(function(regex_fsm, "1", regex_fsm.initial, full_match=False))
    assert res == (2,)

    res = tuple(function(regex_fsm, "12", regex_fsm.initial, full_match=True))
    assert res == (2, 3)

    pattern = interegular.parse_pattern(r"(?:[^\W\d]\w*|[\t \x0c]+)")
    fsm, _ = make_deterministic_fsm(pattern.to_fsm().reduce())

    res = tuple(function(fsm, "x ", fsm.initial, full_match=False))
    assert res == (2,)

    start_state = list(fsm.finals)[0]
    res = tuple(function(fsm, "!", start_state, full_match=False))
    assert res == tuple()


@pytest.mark.parametrize(
    "function",
    [
        walk_fsm_from_token_str,
        walk_fsm_from_token_str_numba,
    ],
)
@pytest.mark.parametrize(
    "transform",
    [
        identity,
        to_bytes,
    ],
)
def test_walk_fsm_multi_bytes(function, transform):
    regex_pattern = interegular.parse_pattern("😂|[😇-😍][😈-😍]*")
    str_regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm().reduce())
    regex_fsm = make_byte_level_better_fsm(str_regex_fsm, keep_utf8=True)

    res = tuple(
        function(
            regex_fsm, merge_symbols(transform("😂")), regex_fsm.initial, full_match=True
        )
    )
    assert res[-1:] == (1,)

    res = tuple(
        function(
            regex_fsm,
            merge_symbols(transform("😂😂")),
            regex_fsm.initial,
            full_match=False,
        )
    )
    assert res[-1:] == (1,)

    res = tuple(
        function(
            regex_fsm, merge_symbols(transform("!")), regex_fsm.initial, full_match=True
        )
    )
    assert res == tuple()

    res = tuple(
        function(
            regex_fsm,
            merge_symbols(transform("😂😂")),
            regex_fsm.initial,
            full_match=True,
        )
    )
    assert res == tuple()


def test_get_sub_fsms_from_seq():
    name_pattern = interegular.parse_pattern(r"[^\W\d]\w*")
    name_fsm, _ = make_deterministic_fsm(name_pattern.to_fsm().reduce())

    def_pattern = interegular.parse_pattern("def")
    def_fsm, _ = make_deterministic_fsm(def_pattern.to_fsm().reduce())

    match_pattern = interegular.parse_pattern("match")
    match_fsm, _ = make_deterministic_fsm(match_pattern.to_fsm().reduce())

    peq_pattern = interegular.parse_pattern(r"\+=")
    peq_fsm, _ = make_deterministic_fsm(peq_pattern.to_fsm().reduce())

    plus_pattern = interegular.parse_pattern(r"\+")
    plus_fsm, _ = make_deterministic_fsm(plus_pattern.to_fsm().reduce())

    fsms = [def_fsm, match_fsm, name_fsm, peq_fsm, plus_fsm]

    fsm, fsms_to_trans_finals = fsm_union(fsms)

    assert fsms_to_trans_finals == {
        0: ({(0, 3), (3, 9), (9, 10)}, {10}, {0: {0}, 1: {3}, 2: {9}, 3: {10}}),
        1: (
            {(0, 4), (4, 5), (5, 6), (6, 7), (7, 8)},
            {8},
            {0: {0}, 1: {4}, 2: {5}, 3: {6}, 4: {7}, 5: {8}},
        ),
        2: (
            {
                (0, 2),
                (0, 3),
                (0, 4),
                (2, 2),
                (3, 2),
                (3, 9),
                (4, 2),
                (4, 5),
                (5, 2),
                (5, 6),
                (6, 2),
                (6, 7),
                (7, 2),
                (7, 8),
                (8, 2),
                (9, 2),
                (9, 10),
                (10, 2),
            },
            {2, 3, 4, 5, 6, 7, 8, 9, 10},
            {0: {0}, 1: {2, 3, 4, 5, 6, 7, 8, 9, 10}},
        ),
        3: ({(0, 1), (1, 11)}, {11}, {0: {0}, 1: {1}, 2: {11}}),
        4: ({(0, 1)}, {1}, {0: {0}, 1: {1}}),
    }

    assert not fsm.accepts("1a")
    assert fsm.accepts("a1")
    assert fsm.accepts("def")
    assert fsm.accepts("match")
    assert fsm.accepts("+=")
    assert fsm.accepts("+")

    state_seq = walk_fsm_from_token_str(fsm, "def", fsm.initial)
    state_seq.insert(0, fsm.fsm_info.initial)

    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(0, False, True), (2, True, True)]

    # Make sure the old-to-new state map is correct
    def_state_seq = walk_fsm_from_token_str(def_fsm, "def", fsm.initial)
    def_state_seq.insert(0, fsm.fsm_info.initial)

    def_old_to_new_states = fsms_to_trans_finals[0][2]
    assert all(
        new_state in def_old_to_new_states[old_state]
        for old_state, new_state in zip(def_state_seq, state_seq)
    )

    state_seq = walk_fsm_from_token_str(fsm, "ef", fsm.initial)
    state_seq.insert(0, fsm.initial)

    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(2, True, True)]

    name_state_seq = walk_fsm_from_token_str(name_fsm, "ef", fsm.initial)
    name_state_seq.insert(0, fsm.initial)

    name_old_to_new_states = fsms_to_trans_finals[2][2]
    assert all(
        new_state in name_old_to_new_states[old_state]
        for old_state, new_state in zip(name_state_seq, state_seq)
    )

    state_seq = walk_fsm_from_token_str(fsm, "match", fsm.initial)
    state_seq.insert(0, fsm.initial)

    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(1, False, True), (2, True, True)]

    match_state_seq = walk_fsm_from_token_str(match_fsm, "match", fsm.initial)
    match_state_seq.insert(0, fsm.initial)

    match_old_to_new_states = fsms_to_trans_finals[1][2]
    assert all(
        new_state in match_old_to_new_states[old_state]
        for old_state, new_state in zip(match_state_seq, state_seq)
    )

    state_seq = walk_fsm_from_token_str(fsm, "defa", fsm.initial)
    state_seq.insert(0, fsm.initial)

    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(2, True, True)]

    state_seq = walk_fsm_from_token_str(fsm, "de", fsm.initial)
    state_seq.insert(0, fsm.initial)

    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(0, True, False), (2, True, True)]

    state_seq = walk_fsm_from_token_str(fsm, "+", fsm.initial, False)
    state_seq.insert(0, fsm.initial)

    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(3, True, False), (4, False, True)]

    state_seq = walk_fsm_from_token_str(fsm, "+=", fsm.initial)
    state_seq.insert(0, fsm.initial)

    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(3, False, True)]

    # Test some overlapping patterns
    join_fsms = [
        interegular.parse_pattern(r"JOIN").to_fsm().reduce(),
        interegular.parse_pattern(r"JOIN LEFT").to_fsm().reduce(),
    ]
    fsm, fsms_to_trans_finals = fsm_union(join_fsms)

    # Matching "OI"
    state_seq = [1, 2, 3]
    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(0, True, False), (1, True, False)]

    # Matching "N"
    state_seq = [3, 4]
    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(0, False, True), (1, True, False)]

    # Matching " "
    state_seq = [4, 5]
    res = list(get_sub_fsms_from_seq(state_seq, fsms_to_trans_finals))
    assert res == [(1, True, False)]


def test_create_fsm_index_end_to_end():
    regex_str = "0|[1-9][0-9]*"

    regex_pattern = interegular.parse_pattern(regex_str)
    regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm().reduce())

    vocabulary = {
        "blah": numba.typed.List([0]),
        "1a": numba.typed.List([1]),
        "2": numba.typed.List([2]),
        "0": numba.typed.List([3]),
        "<EOS>": numba.typed.List([4]),
    }

    vocabulary_nb = numba.typed.List.empty_list(
        numba.types.Tuple(
            (
                numba.types.unicode_type,
                numba.int64[:],
            )
        )
    )
    for token_tuple, token_ids in vocabulary.items():
        token = merge_symbols(token_tuple)
        token_ids_np = np.fromiter(token_ids, dtype=np.dtype("int64"))
        vocabulary_nb.append((token, token_ids_np))

    res = create_fsm_index_end_to_end(regex_fsm.fsm_info, vocabulary_nb)

    assert res == {0: {(2, 2), (3, 1)}, 2: {(2, 2), (3, 2)}}


def test_create_fsm_index_end_to_end_multi_byte():
    regex_str = "😇| [😈-😍][😇-😎]*"

    regex_pattern = interegular.parse_pattern(regex_str)
    regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm().reduce())
    byte_fsm = make_byte_level_better_fsm(regex_fsm, keep_utf8=True)

    vocabulary = {
        "blah": numba.typed.List([0]),
        "😈a": numba.typed.List([1]),
        "😇": numba.typed.List([2]),
        "😍": numba.typed.List([3]),
        merge_symbols(("F0", "9F", "98", "8D")): numba.typed.List([4]),  # '😍'
        " 😍": numba.typed.List([5]),
        merge_symbols((" ", "F0", "9F", "98", "8D")): numba.typed.List([6]),  # ' 😍'
        merge_symbols((" ", "F0", "9F", "98")): numba.typed.List(
            [7]
        ),  # ' 😍' incomplete
        "<EOS>": numba.typed.List([8]),
    }

    vocabulary_nb = numba.typed.List.empty_list(
        numba.types.Tuple(
            (
                numba.types.unicode_type,
                numba.int64[:],
            )
        )
    )
    for token_tuple, token_ids in vocabulary.items():
        token_tuple_np = merge_symbols(token_tuple)
        token_ids_np = np.fromiter(token_ids, dtype=np.dtype("int64"))
        vocabulary_nb.append((token_tuple_np, token_ids_np))

    res = create_fsm_index_end_to_end(byte_fsm.fsm_info, vocabulary_nb)

    assert res == {0: {(5, 3), (6, 3), (7, 7), (2, 2)}, 3: {(2, 3), (3, 3), (4, 3)}}


@pytest.mark.parametrize(
    "hf_tokenizer_uri",
    [
        "gpt2",
        "microsoft/phi-2",
        "Qwen/Qwen1.5-0.5B-Chat",
        "NousResearch/Hermes-2-Pro-Llama-3-8B",
    ],
)
def test_create_fsm_index_tokenizer(hf_tokenizer_uri):
    # The combined regular expressions of a lexer state in a Python grammar
    regex_str = "(?:(?:[0-9](?:(?:_)?[0-9])*(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*|(?:[0-9](?:(?:_)?[0-9])*\\.(?:[0-9](?:(?:_)?[0-9])*)?|\\.[0-9](?:(?:_)?[0-9])*)(?:(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*)?)|[0-9](?:(?:_)?[0-9])*)(?:J|j)|(?:[0-9](?:(?:_)?[0-9])*(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*|(?:[0-9](?:(?:_)?[0-9])*\\.(?:[0-9](?:(?:_)?[0-9])*)?|\\.[0-9](?:(?:_)?[0-9])*)(?:(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*)?)|0(?:x|X)(?:(?:_)?(?:[0-9]|[a-f]|[A-F]))+|0(?:b|B)(?:(?:_)?[0-1])+|0(?:o|O)(?:(?:_)?[0-7])+|(?:(?i:([ubf]?r?|r[ubf])('([^\\\\']|.)*?'))|(?i:([ubf]?r?|r[ubf])(\"([^\\\"]|.)*?\")))|(?:(?:\r?\n[\t ]*|#[^\n]*))+|[1-9](?:(?:_)?[0-9])*|\\\\[\t \x0c]*\r?\n|continue|nonlocal|assert|global|import|lambda|return|async|await|break|class|False|match|raise|while|yield|case|from|None|pass|True|with|def|del|for|not|try|if|[^\\W\\d]\\w*|#[^\n]*|[\t \x0c]+|\\.\\.\\.|@|\\{|\\(|\\[|\\-|\\+|\\*|\\~"

    regex_pattern = interegular.parse_pattern(regex_str)
    # Not reduced, so that there are many states
    regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm())
    bytes_fsm = make_byte_level_better_fsm(regex_fsm, keep_utf8=True)

    num_fsm_states = len(regex_fsm.states)
    assert num_fsm_states == 220

    num_bytes_fsm_states = len(bytes_fsm.states)
    assert num_bytes_fsm_states == 235

    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_uri)
    tokenizer = TransformerTokenizer(tokenizer)

    states_to_token_subsets, empty_token_ids = create_fsm_index_tokenizer(
        bytes_fsm, tokenizer
    )

    assert not empty_token_ids
    assert len(states_to_token_subsets) / num_fsm_states > 0.94


@pytest.mark.parametrize(
    "regex,string,should_accept",
    [
        ("[a-c]+", "😀", False),
        ("[^a-c]+", "😀", True),
        ("😀+", "😀😀😀", True),
        ("😀+", "a", False),
        ("[😀-😍]{2}", "😈😈", True),
        ("[😀-😍]{2}", "aa", False),
        ("[^😀-😍]{2}", "aa", True),
        ("[^😀-😍]{2}", "😈😈", False),
        ("[^😀-😍]{2}", "😎😎", True),
        ("[^😀-😍]{2}", "😎😓", True),
        ("[^😀-😍]{2}", "😎😈", False),
        ("[😀-🙌]{2}", "😎😈", True),
        ("[^😀-🙌]{2}", "😎😈", False),
        ("[^😀-🙌]{2}", "🙏🙏", True),
        ("[^😀-🙌]{2}", "🙏😎", False),
    ],
)
def test_make_byte_level_fsm(regex, string, should_accept):
    str_fsm = interegular.parse_pattern(regex).to_fsm()
    str_accepts = str_fsm.accepts(string)
    assert str_accepts == should_accept

    byte_fsm = make_byte_level_fsm(str_fsm)
    byte_accepts = byte_fsm.accepts(to_bytes(string))  # type: ignore
    assert byte_accepts == str_accepts

    mix_fsm = make_byte_level_fsm(str_fsm, keep_utf8=True)
    mix_accepts = mix_fsm.accepts(to_bytes(string))  # type: ignore
    assert mix_accepts == str_accepts

    mix_accepts_utf8 = mix_fsm.accepts(string)  # type: ignore
    assert mix_accepts_utf8 == str_accepts

    def advance(fsm, state, seq):
        for symbol in seq:
            if state is None:
                return None
            key = fsm.alphabet[symbol]
            state = fsm.map[state].get(key)
        return state

    # verify each state along the pattern
    str_state = str_fsm.initial
    byte_state = byte_fsm.initial
    mix_state = byte_fsm.initial
    for symbol in string:
        str_state = advance(str_fsm, str_state, symbol)
        byte_state = advance(byte_fsm, byte_state, to_bytes(symbol))
        mix_state_utf8 = advance(mix_fsm, mix_state, symbol)
        mix_state = advance(mix_fsm, mix_state, to_bytes(symbol))
        assert byte_state == str_state
        assert mix_state == str_state
        assert mix_state_utf8 == str_state


@pytest.mark.skip(reason="Only for local profiling")
def test_regex_index_performance():
    from line_profiler import LineProfiler  # type: ignore [import]

    regex_str = "(?:(?:[0-9](?:(?:_)?[0-9])*(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*|(?:[0-9](?:(?:_)?[0-9])*\\.(?:[0-9](?:(?:_)?[0-9])*)?|\\.[0-9](?:(?:_)?[0-9])*)(?:(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*)?)|[0-9](?:(?:_)?[0-9])*)(?:J|j)|(?:[0-9](?:(?:_)?[0-9])*(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*|(?:[0-9](?:(?:_)?[0-9])*\\.(?:[0-9](?:(?:_)?[0-9])*)?|\\.[0-9](?:(?:_)?[0-9])*)(?:(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*)?)|0(?:x|X)(?:(?:_)?(?:[0-9]|[a-f]|[A-F]))+|0(?:b|B)(?:(?:_)?[0-1])+|0(?:o|O)(?:(?:_)?[0-7])+|(?:(?i:([ubf]?r?|r[ubf])('([^\\\\']|.)*?'))|(?i:([ubf]?r?|r[ubf])(\"([^\\\"]|.)*?\")))|(?:(?:\r?\n[\t ]*|#[^\n]*))+|[1-9](?:(?:_)?[0-9])*|\\\\[\t \x0c]*\r?\n|continue|nonlocal|assert|global|import|lambda|return|async|await|break|class|False|match|raise|while|yield|case|from|None|pass|True|with|def|del|for|not|try|if|[^\\W\\d]\\w*|#[^\n]*|[\t \x0c]+|\\.\\.\\.|@|\\{|\\(|\\[|\\-|\\+|\\*|\\~"

    regex_pattern = interegular.parse_pattern(regex_str)
    # Not reduced, so that there are many states
    regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm())

    num_fsm_states = len(regex_fsm.states)
    assert num_fsm_states == 220

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer = TransformerTokenizer(tokenizer)

    # Pre-compile Numba functions
    res, _ = create_fsm_index_tokenizer(regex_fsm, tokenizer)
    assert len(res) > 1

    profiler = LineProfiler(create_fsm_index_end_to_end)

    profiler.runctx(
        "create_fsm_index_tokenizer(regex_fsm, tokenizer)",
        globals(),
        locals(),
    )
    profiler.dump_stats("line-profiler-create_fsm_index.pkl")
    profiler.print_stats(output_unit=1e-3, summarize=True, stripzeros=True)


@pytest.mark.skip(reason="Only for local profiling")
def test_json_index_performance():
    import json
    from enum import Enum

    from line_profiler import LineProfiler  # type: ignore [import]
    from pydantic import BaseModel, constr

    import outlines

    class Weapon(str, Enum):
        sword = "sword"
        axe = "axe"
        mace = "mace"
        spear = "spear"
        bow = "bow"
        crossbow = "crossbow"

    class Armor(str, Enum):
        leather = "leather"
        chainmail = "chainmail"
        plate = "plate"

    class Character(BaseModel):
        name: constr(max_length=10)
        # TODO: Add support for conint
        age: int  # conint(int, ge=18, le=100)
        armor: Armor
        weapon: Weapon
        # TODO: Add support for conint
        strength: int  # conint(int, ge=0, le=100)

    model = outlines.models.transformers("gpt2", device="cuda")
    json_schema = json.dumps(Character.model_json_schema())

    def build_regex():
        regex_str = outlines.index.json_schema.build_regex_from_object(json_schema)
        outlines.generate.regex(model, regex_str)

    profiler = LineProfiler(create_fsm_index_end_to_end)
    profiler.add_function(create_fsm_index_tokenizer)
    profiler.add_function(outlines.index.index.RegexFSM.__init__)

    profiler.runctx(
        "build_regex()",
        globals(),
        locals(),
    )
    profiler.dump_stats("line-profiler-build-json-regex.pkl")
    profiler.print_stats(output_unit=1e-3, summarize=True, stripzeros=True)


def test_token_trans_keys_identical():
    """assert two tokens w/ identical behavior wrt FSM have same trans key seq"""

    class MockTokenizer:
        vocabulary = {"a": 1, "b": 2, "z": 3, "eos": 4}
        special_tokens = {"eos"}
        eos_token_id = 4

        def convert_token_to_string(self, token):
            return token

    tokenizer = MockTokenizer()

    pattern = r"z[ab]z"
    regex_pattern = interegular.parse_pattern(pattern)
    interegular_fsm = regex_pattern.to_fsm().reduce()
    regex_fsm, _ = make_deterministic_fsm(interegular_fsm)
    vocabulary, _ = reduced_vocabulary(tokenizer)
    token_trans_keys = get_vocabulary_transition_keys(
        regex_fsm.fsm_info.alphabet_symbol_mapping,
        regex_fsm.fsm_info.alphabet_anything_value,
        vocabulary,
        numba.typed.List.empty_list(numba.types.unicode_type),
    )

    token_str_to_tranition_keys = {
        token_str: trans_key_seq
        for (token_str, _), trans_key_seq in zip(vocabulary, token_trans_keys)
    }
    # `a` and `b` both are workable, but `z` has distinct transition rules
    assert interegular_fsm.accepts("zaz")
    assert interegular_fsm.accepts("zbz")
    assert (token_str_to_tranition_keys["a"] == token_str_to_tranition_keys["b"]).all()
    assert not (
        token_str_to_tranition_keys["a"] == token_str_to_tranition_keys["z"]
    ).all()


def test_token_trans_keys_walk_fsm():
    """assert _walk_fsm works using transition keys"""

    class MockTokenizer:
        vocabulary = {"ab": 1, "ac": 2, "az": 3, "eos": 4}
        special_tokens = {"eos"}
        eos_token_id = 4

        def convert_token_to_string(self, token):
            return token

    tokenizer = MockTokenizer()

    pattern = r"a[bc]z"
    regex_pattern = interegular.parse_pattern(pattern)
    interegular_fsm = regex_pattern.to_fsm().reduce()
    regex_fsm, _ = make_deterministic_fsm(interegular_fsm)
    vocabulary, _ = reduced_vocabulary(tokenizer)
    token_trans_keys = get_vocabulary_transition_keys(
        regex_fsm.fsm_info.alphabet_symbol_mapping,
        regex_fsm.fsm_info.alphabet_anything_value,
        vocabulary,
        numba.typed.List.empty_list(numba.types.unicode_type),
    )

    token_str_trans_key_seq = {
        token_str: trans_key_seq
        for (token_str, _), trans_key_seq in zip(vocabulary, token_trans_keys)
    }

    # verify initial state valid only for "ab" and "ac" using transition key seq
    token_acceptance = {"ab": True, "ac": True, "az": False}
    for token, should_accept in token_acceptance.items():
        token_trans_key_seq = token_str_trans_key_seq[token]
        state_seq = _walk_fsm(
            regex_fsm.fsm_info.transitions,
            regex_fsm.fsm_info.initial,
            regex_fsm.fsm_info.finals,
            token_trans_key_seq,
            regex_fsm.fsm_info.initial,
            False,
        )
        is_accepted = len(state_seq) >= len(token_trans_key_seq)
        assert should_accept == is_accepted


def test_numba_leading_null_byte_UnicodeCharSeq_remains_broken():
    """Assert numba UnicodeCharSeq w/ leading \x00 is still broken"""
    # EXPLANATION:
    # https://github.com/outlines-dev/outlines/pull/930#issuecomment-2143535968

    # from https://github.com/numba/numba/issues/9542
    d = numba.typed.typeddict.Dict.empty(numba.types.UnicodeCharSeq(1), numba.int64)
    d["一"] = 10  # \xe4\xb8\x80
    with pytest.raises(KeyError):
        str(d)

    # most characters are fine, but "\x00" is converted to ""
    l = np.fromiter(["\x99", "\x00"], dtype=np.dtype("U2"))
    assert str(l[0]) == "\x99"  # fine
    assert str(l[1]) == ""  # 1-byte null converted to 0-bytes


@pytest.mark.parametrize("input_key", ["一", "\x00"])
def test_numba_leading_null_byte_unicode_type_sane(input_key):
    """Assert numba unicode_type w/ leading \x00 is working"""
    # EXPLANATION:
    # https://github.com/outlines-dev/outlines/pull/930#issuecomment-2143535968

    # from https://github.com/numba/numba/issues/9542
    d = numba.typed.typeddict.Dict.empty(numba.types.unicode_type, numba.int64)
    d["一"] = 10  # \xe4\xb8\x80
    str(d)  # assert successfully interprets


@pytest.mark.parametrize(
    "rare_token",
    [
        "�",
        "��",
        "�.",
        "�..",
        "▁�",
        "▁▁�",
        "▁�.",
        "▁�.",
        "▁▁�..",
    ],
)
def test_reduced_vocabulary_with_rare_tokens(rare_token):
    """Assert reduced_vocabulary works with rare tokens.

    See [1] and [2] for context.

    [1]: https://github.com/outlines-dev/outlines/pull/763
    [2]: https://github.com/outlines-dev/outlines/pull/948
    """
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer = adapt_tokenizer(tokenizer=tokenizer)
    tokenizer.vocabulary[rare_token] = max(tokenizer.vocabulary.values()) + 1
    reduced_vocabulary(tokenizer)
