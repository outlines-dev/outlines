import pytest

from outlines.fsm.guide import CFGGuide, Generate, RegexGuide, StopAtEOSGuide, Write


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
    assert instruction.tokens is None

    instruction = fsm.get_next_instruction(fsm.final_state)
    assert isinstance(instruction, Write)
    assert instruction.tokens == [2]

    assert fsm.get_next_state(fsm.start_state, 2) == fsm.final_state
    assert fsm.get_next_state(fsm.start_state, 1) == fsm.start_state
    assert fsm.is_final_state(fsm.start_state) is False
    assert fsm.is_final_state(fsm.final_state) is True


def test_regex_vocabulary_error():
    class MockTokenizer:
        vocabulary = {"a": 1}
        special_tokens = {"eos"}
        eos_token_id = 3

        def convert_token_to_string(self, token):
            return token

    regex_str = "[1-9]"

    with pytest.raises(ValueError, match="The vocabulary"):
        RegexGuide.from_regex(regex_str, MockTokenizer())


def test_regex():
    class MockTokenizer:
        vocabulary = {"1": 1, "a": 2, "eos": 3}
        special_tokens = {"eos"}
        eos_token_id = 3

        def convert_token_to_string(self, token):
            return token

    regex_str = "[1-9]"
    tokenizer = MockTokenizer()
    fsm = RegexGuide.from_regex(regex_str, tokenizer)

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
    fsm = RegexGuide.from_regex(regex_str, tokenizer)

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
    fsm = RegexGuide.from_regex(regex_str, tokenizer)

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
    fsm = RegexGuide.from_regex(regex_str, tokenizer)

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
            if isinstance(token_ids[0], list):
                return [
                    "".join(map(self.inverse_vocabulary.get, token_ids_sublist))
                    for token_ids_sublist in token_ids
                ]
            return [self.inverse_vocabulary[token_id] for token_id in token_ids]

    cfg_str = """
        start: expr
        expr: "{" expr "}" | "[" expr "]" |
    """
    tokenizer = MockTokenizer()

    guide = CFGGuide(cfg_str, tokenizer)

    assert_expected_tensor_ids(
        guide.get_next_instruction(guide.initial_state).tokens, [1, 3, 5]
    )
    state = guide.get_next_state(guide.initial_state, token_id=1)
    assert not guide.must_terminate_state(state)
    assert not guide.can_terminate_state(state)

    assert_expected_tensor_ids(guide.get_next_instruction(state).tokens, [1, 2, 3])
    state = guide.get_next_state(state, token_id=3)
    assert not guide.must_terminate_state(state)
    assert not guide.can_terminate_state(state)

    assert_expected_tensor_ids(guide.get_next_instruction(state).tokens, [1, 3, 4])
    state = guide.get_next_state(state, token_id=4)
    assert not guide.must_terminate_state(state)
    assert not guide.can_terminate_state(state)

    assert_expected_tensor_ids(guide.get_next_instruction(state).tokens, [2])
    state = guide.get_next_state(state, token_id=2)
    assert guide.must_terminate_state(state)
    assert guide.can_terminate_state(state)

    assert_expected_tensor_ids(guide.get_next_instruction(state).tokens, [5])
    state = guide.get_next_state(state, token_id=5)
    assert guide.must_terminate_state(state)
    assert guide.can_terminate_state(state)


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
            if isinstance(token_ids[0], list):
                return [
                    "".join(map(self.inverse_vocabulary.get, token_ids_sublist))
                    for token_ids_sublist in token_ids
                ]
            return [self.inverse_vocabulary[token_id] for token_id in token_ids]

    cfg_str = """
        start: expr+
        expr: "(" subexpr ")"
        subexpr: expr |
    """
    tokenizer = MockTokenizer()

    guide = CFGGuide(cfg_str, tokenizer)

    assert_expected_tensor_ids(
        guide.get_next_instruction(guide.initial_state).tokens, [1]
    )
    state = guide.get_next_state(guide.initial_state, token_id=1)
    assert not guide.must_terminate_state(state)
    assert not guide.can_terminate_state(state)

    assert_expected_tensor_ids(guide.get_next_instruction(state).tokens, [1, 2])
    state = guide.get_next_state(state, token_id=2)
    assert not guide.must_terminate_state(state)
    assert guide.can_terminate_state(state)

    # possible to continue or terminate
    assert_expected_tensor_ids(guide.get_next_instruction(state).tokens, [1, 3])
    state = guide.get_next_state(state, token_id=3)  # feed eos
    assert guide.must_terminate_state(state)
    assert guide.can_terminate_state(state)

    # once eos generated, can only terminate
    assert_expected_tensor_ids(guide.get_next_instruction(state).tokens, [3])


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
            if isinstance(token_ids[0], list):
                return [
                    "".join(map(self.inverse_vocabulary.get, token_ids_sublist))
                    for token_ids_sublist in token_ids
                ]
            return [self.inverse_vocabulary[token_id] for token_id in token_ids]

    cfg_str = """
        start: LETTER+
        LETTER: "a"
        WS: " "
        %ignore WS
    """
    tokenizer = MockTokenizer()

    guide = CFGGuide(cfg_str, tokenizer)

    state = guide.initial_state

    assert_expected_tensor_ids(guide.get_next_instruction(state).tokens, [1, 2])
    state = guide.get_next_state(state, token_id=2)
    assert not guide.must_terminate_state(state)
    assert not guide.can_terminate_state(state)

    assert_expected_tensor_ids(guide.get_next_instruction(state).tokens, [1, 2])
    state = guide.get_next_state(state, token_id=1)
    assert not guide.must_terminate_state(state)
    assert guide.can_terminate_state(state)

    assert_expected_tensor_ids(guide.get_next_instruction(state).tokens, [1, 2, 3])
    state = guide.get_next_state(state, token_id=2)
    assert not guide.must_terminate_state(state)
    assert guide.can_terminate_state(state)

    assert_expected_tensor_ids(guide.get_next_instruction(state).tokens, [1, 2, 3])
    state = guide.get_next_state(state, token_id=2)
    assert not guide.must_terminate_state(state)
    assert guide.can_terminate_state(state)

    assert_expected_tensor_ids(guide.get_next_instruction(state).tokens, [1, 2, 3])
    state = guide.get_next_state(state, token_id=1)
    assert not guide.must_terminate_state(state)
    assert guide.can_terminate_state(state)

    assert_expected_tensor_ids(guide.get_next_instruction(state).tokens, [1, 2, 3])
    state = guide.get_next_state(state, token_id=3)
    assert guide.must_terminate_state(state)
    assert guide.can_terminate_state(state)

    # once eos generated, can only terminate
    assert_expected_tensor_ids(guide.get_next_instruction(state).tokens, [3])


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
            if isinstance(token_ids[0], list):
                return [
                    "".join(map(self.inverse_vocabulary.get, token_ids_sublist))
                    for token_ids_sublist in token_ids
                ]
            return [self.inverse_vocabulary[token_id] for token_id in token_ids]

    cfg_str = """
        start: S
        S: "aa" | "bb"
    """
    tokenizer = MockTokenizer()

    guide = CFGGuide(cfg_str, tokenizer)

    assert_expected_tensor_ids(
        guide.get_next_instruction(guide.initial_state).tokens, [1, 2]
    )
    state = guide.get_next_state(guide.initial_state, token_id=1)
    assert not guide.must_terminate_state(state)
    assert not guide.can_terminate_state(state)

    assert_expected_tensor_ids(guide.get_next_instruction(state).tokens, [1])
    state = guide.get_next_state(state, token_id=1)
    assert guide.must_terminate_state(state)
    assert guide.can_terminate_state(state)

    assert_expected_tensor_ids(guide.get_next_instruction(state).tokens, [3])
    state = guide.get_next_state(state, token_id=3)
    assert guide.must_terminate_state(state)
    assert guide.can_terminate_state(state)


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
            if isinstance(token_ids[0], list):
                return [
                    "".join(map(self.inverse_vocabulary.get, token_ids_sublist))
                    for token_ids_sublist in token_ids
                ]
            return [self.inverse_vocabulary[token_id] for token_id in token_ids]

    cfg_str = """
        start: s
        s: "(" s ")" | /a+/
    """
    tokenizer = MockTokenizer()

    guide = CFGGuide(cfg_str, tokenizer)

    assert_expected_tensor_ids(
        guide.get_next_instruction(guide.initial_state).tokens, [1, 3]
    )
    state = guide.get_next_state(guide.initial_state, token_id=1)
    assert not guide.must_terminate_state(state)
    assert not guide.can_terminate_state(state)

    assert_expected_tensor_ids(guide.get_next_instruction(state).tokens, [1, 3])
    state = guide.get_next_state(state, token_id=3)
    assert not guide.must_terminate_state(state)
    assert not guide.can_terminate_state(state)

    assert_expected_tensor_ids(guide.get_next_instruction(state).tokens, [2, 3])
    state = guide.get_next_state(state, token_id=3)
    assert not guide.must_terminate_state(state)
    assert not guide.can_terminate_state(state)

    assert_expected_tensor_ids(guide.get_next_instruction(state).tokens, [2, 3])
    state = guide.get_next_state(state, token_id=2)

    assert guide.must_terminate_state(state)
    assert guide.can_terminate_state(state)

    assert_expected_tensor_ids(guide.get_next_instruction(state).tokens, [4])
    state = guide.get_next_state(state, token_id=4)
    assert guide.must_terminate_state(state)
    assert guide.can_terminate_state(state)
