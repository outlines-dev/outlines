import pytest
import torch

from outlines.fsm.guide import (
    CFGGuide,
    Generate,
    RegexGuide,
    StopAtEOSGuide,
    Write,
    add_crossing_tokens_states_to_tokens_map,
    align_tokens_states_to_token_maps,
    find_crossing_tokens,
    get_crossing_tokens_target_states,
    swap_state_ids_states_to_tokens_map,
)


def assert_expected_tensor_ids(tensor, ids):
    assert len(tensor) == len(ids)
    norm_tensor = sorted(map(int, tensor))
    norm_ids = sorted(map(int, tensor))
    assert norm_tensor == norm_ids, (norm_tensor, norm_ids)


def test_stop_at_eos():
    class MockTokenizer:
        vocabulary = {"a": 1, "eos": 2}
        eos_token_id = 2

    fsm = StopAtEOSGuide(MockTokenizer())

    instruction = fsm.get_next_instruction(fsm.start_state)
    assert isinstance(instruction, Generate)
    assert instruction.tokens == [1, 2]

    instruction = fsm.get_next_instruction(fsm.final_state)
    assert isinstance(instruction, Write)
    assert instruction.tokens == [2]

    assert fsm.get_next_state(fsm.start_state, 2) == fsm.final_state
    assert fsm.get_next_state(fsm.start_state, 1) == fsm.start_state
    assert fsm.is_final_state(fsm.start_state) is False
    assert fsm.is_final_state(fsm.final_state) is True


def test_stop_at_eos_align_prompt_tokens():
    class MockTokenizer:
        vocabulary = {"a": 1, "ab": 2, "b": 3, "eos": 4}
        eos_token_id = 4

        def encode(self, prompt):
            return torch.Tensor([[(self.vocabulary[char]) for char in prompt]]), None

        def decode(self, token_ids):
            reversed_vocabulary = {value: key for key, value in self.vocabulary.items()}
            return [
                "".join([reversed_vocabulary[int(token_id)] for token_id in token_ids])
            ]

    tokenizer = MockTokenizer()

    # with crossing tokens
    fsm = StopAtEOSGuide(tokenizer)
    aligned_prompt = fsm.align_prompt_tokens("ba", tokenizer)
    assert aligned_prompt == "b"
    assert fsm.states_to_token_maps == {0: {1: 1, 2: 1}, 1: {1: 1, 2: 1, 3: 1, 4: -1}}

    # no crossing tokens
    fsm = StopAtEOSGuide(tokenizer)
    aligned_prompt = fsm.align_prompt_tokens("bb", tokenizer)
    assert aligned_prompt == "bb"
    assert fsm.states_to_token_maps == {0: {1: 0, 2: 0, 3: 0, 4: -1}}

    # all prompt tokens would be removed -> no prompt alignment
    fsm = StopAtEOSGuide(tokenizer)
    aligned_prompt = fsm.align_prompt_tokens("a", tokenizer)
    assert aligned_prompt == "a"
    assert fsm.states_to_token_maps == {0: {1: 0, 2: 0, 3: 0, 4: -1}}


def test_regex_vocabulary_error():
    class MockTokenizer:
        vocabulary = {"a": 1}
        special_tokens = {"eos"}
        eos_token_id = 3

        def convert_token_to_string(self, token):
            return token

    regex_str = "[1-9]"

    with pytest.raises(ValueError, match="The vocabulary"):
        RegexGuide(regex_str, MockTokenizer())


def test_regex():
    class MockTokenizer:
        vocabulary = {"1": 1, "a": 2, "eos": 3}
        special_tokens = {"eos"}
        eos_token_id = 3

        def convert_token_to_string(self, token):
            return token

    regex_str = "[1-9]"
    tokenizer = MockTokenizer()
    fsm = RegexGuide(regex_str, tokenizer)

    assert fsm.states_to_token_maps == {0: {1: 1}}

    instruction = fsm.get_next_instruction(0)
    assert isinstance(instruction, Generate)
    assert_expected_tensor_ids(instruction.tokens, [1])

    assert fsm.get_next_state(state=0, token_id=1) == 1
    assert fsm.get_next_state(state=0, token_id=tokenizer.eos_token_id) == -1

    assert fsm.is_final_state(0) is False

    for state in fsm.final_states:
        assert fsm.is_final_state(state) is True


def test_regex_multi_byte_llama_like():
    class MockTokenizer:
        vocabulary = {
            "1": 1,
            "a": 2,
            "eos": 3,
            "😍": 4,
            "<0xF0>": 5,
            "<0x9F>": 6,
            "<0x98>": 7,
            "<0x88>": 8,  # 😈
            "\ufffd": 9,
            "\ufffd\ufffd": 10,
        }
        special_tokens = {"eos"}
        eos_token_id = 3

        def convert_token_to_string(self, token):
            if token[0] == "<":
                return "\ufffd"
            return token

    regex_str = "[😁-😎]"
    tokenizer = MockTokenizer()
    fsm = RegexGuide(regex_str, tokenizer)

    assert fsm.states_to_token_maps == {
        0: {5: 1, 4: 2},
        1: {6: 3},
        3: {7: 4},
        4: {8: 2},
    }

    instruction = fsm.get_next_instruction(0)
    assert isinstance(instruction, Generate)
    assert_expected_tensor_ids(instruction.tokens, [5, 4])

    assert fsm.get_next_state(state=0, token_id=5) == 1
    assert fsm.get_next_state(state=0, token_id=tokenizer.eos_token_id) == -1

    assert fsm.is_final_state(0) is False

    for state in fsm.final_states:
        assert fsm.is_final_state(state) is True


def test_regex_multi_byte_gpt2_like():
    class MockTokenizer:
        vocabulary = {
            "1": 1,
            "a": 2,
            "eos": 3,
            "😍": 4,
            " ": 5,
            "\ufffd": 6,
            "\ufffd\ufffd": 7,
            "ðŁĺ": 8,
            "Ī": 9,  # '😈'
            "Ġð": 10,
            "ŁĺĪ": 11,  # ' 😈'
        }
        special_tokens = {"eos"}
        eos_token_id = 3

        def convert_token_to_string(self, token):
            if self.vocabulary[token] >= 8:
                return "\ufffd"
            return token

    regex_str = " [😁-😎]"
    tokenizer = MockTokenizer()
    fsm = RegexGuide(regex_str, tokenizer)

    assert fsm.states_to_token_maps == {
        0: {5: 1, 10: 2},
        1: {8: 5, 4: 3},
        2: {11: 3},
        5: {9: 3},
    }

    instruction = fsm.get_next_instruction(0)
    assert isinstance(instruction, Generate)
    assert_expected_tensor_ids(instruction.tokens, [5, 10])

    assert fsm.get_next_state(state=0, token_id=5) == 1
    assert fsm.get_next_state(state=0, token_id=tokenizer.eos_token_id) == -1

    assert fsm.is_final_state(0) is False

    for state in fsm.final_states:
        assert fsm.is_final_state(state) is True


def test_regex_align_prompt_tokens():
    class MockTokenizer:
        vocabulary = {"1": 1, "2": 2, "12": 3, "eos": 4}
        special_tokens = {"eos"}
        eos_token_id = 4

        def convert_token_to_string(self, token):
            return token

        def encode(self, prompt):
            return torch.Tensor([[self.vocabulary[char] for char in prompt]]), None

        def decode(self, token_ids):
            reversed_vocabulary = {value: key for key, value in self.vocabulary.items()}
            return [
                "".join([reversed_vocabulary[int(token_id)] for token_id in token_ids])
            ]

    regex_str = "[1-9]"
    tokenizer = MockTokenizer()

    # with crossing tokens
    fsm = RegexGuide(regex_str, tokenizer)
    aligned_prompt = fsm.align_prompt_tokens("11", tokenizer)
    assert aligned_prompt == "1"
    assert fsm.states_to_token_maps == {0: {1: 2, 3: 1}, 2: {1: 1, 2: 1}}

    # no crossing tokens
    fsm = RegexGuide(regex_str, tokenizer)
    aligned_prompt = fsm.align_prompt_tokens("22", tokenizer)
    assert aligned_prompt == "22"
    assert fsm.states_to_token_maps == {0: {1: 1, 2: 1}}

    # all prompt tokens would be removed -> no prompt alignment
    fsm = RegexGuide(regex_str, tokenizer)
    aligned_prompt = fsm.align_prompt_tokens("1", tokenizer)
    assert aligned_prompt == "1"
    assert fsm.states_to_token_maps == {0: {1: 1, 2: 1}}


def test_regex_final_state():
    """Make sure that the FSM stays in the final state as we keep generating"""

    class MockTokenizer:
        vocabulary = {"`": 101, ".": 102, "\n": 103, "eos": 104}
        special_tokens = {"eos"}
        eos_token_id = 104

        def convert_token_to_string(self, token):
            return token

    regex_str = r"`\n(\.\n)?`\n"
    tokenizer = MockTokenizer()
    fsm = RegexGuide(regex_str, tokenizer)

    state = fsm.get_next_state(state=4, token_id=103)
    assert state == 5
    assert fsm.is_final_state(state)

    state = fsm.get_next_state(state=5, token_id=103)
    assert fsm.is_final_state(state)


def test_cfg():
    class MockTokenizer:
        vocabulary = {"{": 1, "}": 2, "[": 3, "]": 4, "eos": 5}
        special_tokens = {"eos"}
        eos_token = "eos"
        eos_token_id = 5

        def convert_token_to_string(self, token):
            return token

        @property
        def inverse_vocabulary(self):
            return {v: k for k, v in self.vocabulary.items()}

        def decode(self, token_ids):
            return [self.inverse_vocabulary[t] for t in token_ids]

    cfg_str = """
        start: expr
        expr: "{" expr "}" | "[" expr "]" |
    """
    tokenizer = MockTokenizer()
    fsm = CFGGuide(cfg_str, tokenizer)

    instruction = fsm.get_next_instruction(fsm.start_state)
    assert isinstance(instruction, Generate)
    assert_expected_tensor_ids(instruction.tokens, [1, 3, 5])
    state = fsm.get_next_state(state=fsm.start_state, token_id=1)
    assert fsm.generation == "{"
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Generate)
    assert_expected_tensor_ids(instruction.tokens, [1, 2, 3])
    state = fsm.get_next_state(state=state, token_id=3)
    assert fsm.generation == "{["
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Generate)
    assert_expected_tensor_ids(instruction.tokens, [1, 3, 4])
    state = fsm.get_next_state(state=state, token_id=4)
    assert fsm.generation == "{[]"
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Generate)
    assert_expected_tensor_ids(instruction.tokens, [2])
    state = fsm.get_next_state(state=state, token_id=2)
    assert fsm.generation == "{[]}"
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Write)
    assert_expected_tensor_ids(instruction.tokens, [5])
    state = fsm.get_next_state(state=state, token_id=5)
    assert fsm.generation == "{[]}"
    assert fsm.is_final_state(state)


def test_cfg_early_termination():
    class MockTokenizer:
        vocabulary = {"(": 1, ")": 2, "eos": 3}
        special_tokens = {"eos"}
        eos_token = "eos"
        eos_token_id = 3

        def convert_token_to_string(self, token):
            return token

        @property
        def inverse_vocabulary(self):
            return {v: k for k, v in self.vocabulary.items()}

        def decode(self, token_ids):
            return [self.inverse_vocabulary[t] for t in token_ids]

    cfg_str = """
        start: expr+
        expr: "(" subexpr ")"
        subexpr: expr |
    """
    tokenizer = MockTokenizer()
    fsm = CFGGuide(cfg_str, tokenizer)

    instruction = fsm.get_next_instruction(fsm.start_state)
    assert isinstance(instruction, Generate)
    assert_expected_tensor_ids(instruction.tokens, [1])
    state = fsm.get_next_state(state=fsm.start_state, token_id=1)
    assert fsm.generation == "("
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Generate)
    assert_expected_tensor_ids(instruction.tokens, [1, 2])
    state = fsm.get_next_state(state=state, token_id=2)
    assert fsm.generation == "()"
    assert not fsm.is_final_state(state)

    # possible to continue or terminate
    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Generate)
    assert_expected_tensor_ids(instruction.tokens, [1, 3])
    state = fsm.get_next_state(state=state, token_id=3)  # feed eos
    assert fsm.generation == "()"
    assert fsm.is_final_state(state)

    # once eos generated, can only terminate
    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Write)
    assert_expected_tensor_ids(instruction.tokens, [3])


def test_cfg_ignore_directive():
    class MockTokenizer:
        vocabulary = {"a": 1, " ": 2, "eos": 3}
        special_tokens = {"eos"}
        eos_token = "eos"
        eos_token_id = 3

        def convert_token_to_string(self, token):
            return token

        @property
        def inverse_vocabulary(self):
            return {v: k for k, v in self.vocabulary.items()}

        def decode(self, token_ids):
            return [self.inverse_vocabulary[t] for t in token_ids]

    cfg_str = """
        start: LETTER+
        LETTER: "a"
        WS: " "
        %ignore WS
    """
    tokenizer = MockTokenizer()
    fsm = CFGGuide(cfg_str, tokenizer)

    state = 0

    instruction = fsm.get_next_instruction(0)
    assert isinstance(instruction, Generate)
    assert_expected_tensor_ids(instruction.tokens, [1, 2])
    state = fsm.get_next_state(state=0, token_id=2)
    assert fsm.generation == " "
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(0)
    assert isinstance(instruction, Generate)
    assert_expected_tensor_ids(instruction.tokens, [1, 2])
    state = fsm.get_next_state(state=0, token_id=1)
    assert fsm.generation == " a"
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Generate)
    assert_expected_tensor_ids(instruction.tokens, [1, 2, 3])
    state = fsm.get_next_state(state=state, token_id=2)
    assert fsm.generation == " a "
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Generate)
    assert_expected_tensor_ids(instruction.tokens, [1, 2, 3])
    state = fsm.get_next_state(state=state, token_id=2)
    assert fsm.generation == " a  "
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Generate)
    assert_expected_tensor_ids(instruction.tokens, [1, 2, 3])
    state = fsm.get_next_state(state=state, token_id=1)
    assert fsm.generation == " a  a"
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Generate)
    assert_expected_tensor_ids(instruction.tokens, [1, 2, 3])
    state = fsm.get_next_state(state=state, token_id=3)
    assert fsm.generation == " a  a"
    assert fsm.is_final_state(state)

    # once eos generated, can only terminate
    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Write)
    assert_expected_tensor_ids(instruction.tokens, [3])


def test_cfg_multitoken_terminal():
    class MockTokenizer:
        vocabulary = {"a": 1, "b": 2, "eos": 3}
        special_tokens = {"eos"}
        eos_token = "eos"
        eos_token_id = 3

        def convert_token_to_string(self, token):
            return token

        @property
        def inverse_vocabulary(self):
            return {v: k for k, v in self.vocabulary.items()}

        def decode(self, token_ids):
            return [self.inverse_vocabulary[t] for t in token_ids]

    cfg_str = """
        start: S
        S: "aa" | "bb"
    """
    tokenizer = MockTokenizer()
    fsm = CFGGuide(cfg_str, tokenizer)

    instruction = fsm.get_next_instruction(fsm.start_state)
    assert isinstance(instruction, Generate)
    assert_expected_tensor_ids(instruction.tokens, [1, 2])
    assert fsm.reset_state  # starting new regex
    state = fsm.get_next_state(state=fsm.start_state, token_id=1)
    assert fsm.generation == "a"
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Generate)
    assert_expected_tensor_ids(instruction.tokens, [1])
    assert not fsm.reset_state  # continuing current regex
    state = fsm.get_next_state(state=state, token_id=1)
    assert fsm.generation == "aa"
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Write)
    assert_expected_tensor_ids(instruction.tokens, [3])
    assert not fsm.reset_state  # completing current regex
    state = fsm.get_next_state(state=state, token_id=3)
    assert fsm.generation == "aa"
    assert fsm.is_final_state(state)


def test_cfg_allow_both_extend_and_shift_terminal():
    class MockTokenizer:
        vocabulary = {"(": 1, ")": 2, "a": 3, "eos": 4}
        special_tokens = {"eos"}
        eos_token = "eos"
        eos_token_id = 4

        def convert_token_to_string(self, token):
            return token

        @property
        def inverse_vocabulary(self):
            return {v: k for k, v in self.vocabulary.items()}

        def decode(self, token_ids):
            return [self.inverse_vocabulary[t] for t in token_ids]

    cfg_str = """
        start: s
        s: "(" s ")" | /a+/
    """
    tokenizer = MockTokenizer()
    fsm = CFGGuide(cfg_str, tokenizer)

    instruction = fsm.get_next_instruction(fsm.start_state)
    assert isinstance(instruction, Generate)
    assert_expected_tensor_ids(instruction.tokens, [1, 3])
    state = fsm.get_next_state(state=fsm.start_state, token_id=1)
    assert fsm.generation == "("
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Generate)
    assert_expected_tensor_ids(instruction.tokens, [1, 3])
    state = fsm.get_next_state(state=state, token_id=3)
    assert fsm.generation == "(a"
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Generate)
    assert_expected_tensor_ids(instruction.tokens, [2, 3])
    state = fsm.get_next_state(state=state, token_id=3)
    assert fsm.generation == "(aa"
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Generate)
    assert_expected_tensor_ids(instruction.tokens, [2, 3])
    state = fsm.get_next_state(state=state, token_id=2)
    assert fsm.generation == "(aa)"
    assert not fsm.is_final_state(state)

    instruction = fsm.get_next_instruction(state)
    assert isinstance(instruction, Write)
    assert_expected_tensor_ids(instruction.tokens, [4])
    state = fsm.get_next_state(state=state, token_id=4)
    assert fsm.generation == "(aa)"
    assert fsm.is_final_state(state)


@pytest.mark.parametrize(
    "token_ids,vocabulary,expected_output",
    [
        # Several possible crossing tokens for the last prompt token
        ([1, 2], {"a": 1, "ab": 2, "abc": 3, "abcd": 4}, {1: [3, 4]}),
        # Several possible crossing tokens for the one before last prompt token
        ([1, 2, 3], {"a": 1, "b": 2, "c": 3, "bcd": 4, "bcde": 5}, {1: [4, 5]}),
        # Several possible crossing tokens for several different tokens of the prompt
        (
            [1, 2, 3],
            {"a": 1, "b": 2, "c": 3, "cd": 4, "cde": 5, "bcd": 6, "bcde": 7},
            {1: [6, 7], 2: [4, 5]},
        ),
        # No crossing token found
        ([1, 2], {"a": 1, "b": 2, "c": 3, "cd": 4}, {}),
    ],
)
def test_find_crossing_tokens(token_ids, vocabulary, expected_output):
    assert find_crossing_tokens(token_ids, vocabulary) == expected_output


@pytest.mark.parametrize(
    "states_to_tokens_map,crossing_tokens,prompt_token_ids,vocabulary,expected_output",
    [
        # Only some of the crossing tokens are valid, several different target states
        (
            {
                0: {8: 1, 10: 1, 11: -1},
                1: {10: -1},
            },
            {1: [6, 7], 2: [4, 5]},
            [1, 2, 3],
            {
                "a": 1,
                "b": 2,
                "c": 3,
                "cd": 4,
                "cde": 5,
                "bcd": 6,
                "bcdf": 7,
                "d": 8,
                "e": 9,
                "f": 10,
                "df": 11,
            },
            {1: {6: 1, 7: -1}, 2: {4: 1}},
        ),
        # No valid crossing tokens
        (
            {
                0: {9: 1},
                1: {8: 2, 11: -1},
                2: {10: -1},
            },
            {1: [6, 7], 2: [4, 5]},
            [1, 2, 3],
            {
                "a": 1,
                "b": 2,
                "c": 3,
                "cd": 4,
                "cde": 5,
                "bcd": 6,
                "bcdf": 7,
                "d": 8,
                "e": 9,
                "f": 10,
                "df": 11,
            },
            {},
        ),
    ],
)
def test_get_crossing_tokens_target_states(
    states_to_tokens_map, crossing_tokens, prompt_token_ids, vocabulary, expected_output
):
    assert (
        get_crossing_tokens_target_states(
            states_to_tokens_map, crossing_tokens, prompt_token_ids, vocabulary
        )
        == expected_output
    )


@pytest.mark.parametrize(
    "states_to_tokens_map,first_state_id,second_state_id,expected_output",
    [
        (
            {
                0: {10: 1, 11: 2, 12: -1},
                1: {12: 2, 14: -1},
                2: {15: 2, 16: 0, 17: -1},
                3: {18: 0, 19: 1, 20: 2},
            },
            0,
            3,
            {
                3: {10: 1, 11: 2, 12: -1},
                1: {12: 2, 14: -1},
                2: {15: 2, 16: 3, 17: -1},
                0: {18: 3, 19: 1, 20: 2},
            },
        )
    ],
)
def test_swap_state_ids_states_to_tokens_map(
    states_to_tokens_map, first_state_id, second_state_id, expected_output
):
    assert (
        swap_state_ids_states_to_tokens_map(
            states_to_tokens_map, first_state_id, second_state_id
        )
        == expected_output
    )


def test_swap_state_ids_states_to_tokens_map_key_error():
    with pytest.raises(KeyError):
        swap_state_ids_states_to_tokens_map({0: {1: 1}, 1: {2: -1}}, 0, 2)


@pytest.mark.parametrize(
    "states_to_tokens_map,prompt_token_ids,crossing_tokens_map,expected_output",
    [
        # Add several new states to states_to_tokens_map
        (
            {
                0: {10: 1, 11: 2, 12: -1},
                1: {12: 2, 14: -1},
                2: {15: 2, 16: 0, 17: -1},
                3: {18: 0, 19: 1, 20: 2},
            },
            [6, 7, 8],
            {
                1: {20: 1, 21: 2},
                2: {22: 1, 23: 3},
            },
            (
                {
                    4: {10: 1, 11: 2, 12: -1},
                    1: {12: 2, 14: -1},
                    2: {15: 2, 16: 4, 17: -1},
                    3: {18: 4, 19: 1, 20: 2},
                    0: {7: 5, 20: 1, 21: 2},
                    5: {8: 4, 22: 1, 23: 3},
                },
                2,
            ),
        ),
        # No crossing tokens, unchanged states_to_tokens_map
        ({0: {1: -1, 2: -1}}, [5, 6, 7, 8], {}, ({0: {1: -1, 2: -1}}, 0)),
    ],
)
def test_add_crossing_tokens_states_to_tokens_map(
    states_to_tokens_map, prompt_token_ids, crossing_tokens_map, expected_output
):
    assert (
        add_crossing_tokens_states_to_tokens_map(
            states_to_tokens_map, prompt_token_ids, crossing_tokens_map
        )
        == expected_output
    )


@pytest.mark.parametrize(
    "token_ids,vocabulary,states_to_token_maps,expected_output",
    [
        (
            [1, 2, 3],
            {
                "a": 1,
                "b": 2,
                "c": 3,
                "cd": 4,
                "cde": 5,
                "bcd": 6,
                "bcdf": 7,
                "d": 8,
                "e": 9,
                "f": 10,
                "df": 11,
            },
            {
                0: {8: 1, 10: 1, 11: -1},
                1: {10: -1},
            },
            (
                [1],
                {
                    2: {8: 1, 10: 1, 11: -1},
                    1: {10: -1},
                    0: {2: 3, 6: 1, 7: -1},
                    3: {3: 2, 4: 1},
                },
            ),
        )
    ],
)
def test_align_tokens_states_to_token_maps(
    token_ids, vocabulary, states_to_token_maps, expected_output
):
    assert (
        align_tokens_states_to_token_maps(token_ids, vocabulary, states_to_token_maps)
        == expected_output
    )
