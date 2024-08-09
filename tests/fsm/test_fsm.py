import pytest

from outlines.fsm.fsm import RegexFSM, StopAtEosFSM


def assert_expected_tensor_ids(tensor, ids):
    assert len(tensor) == len(ids)
    norm_tensor = sorted(map(int, tensor))
    norm_ids = sorted(map(int, tensor))
    assert norm_tensor == norm_ids, (norm_tensor, norm_ids)


def test_stop_at_eos():
    class MockTokenizer:
        vocabulary = {"a": 1, "eos": 2}
        eos_token_id = 2

    with pytest.warns(UserWarning):
        fsm = StopAtEosFSM(MockTokenizer())

    assert fsm.allowed_token_ids(fsm.start_state) is None
    assert fsm.allowed_token_ids(fsm.final_state) == [2]
    assert fsm.next_state(fsm.start_state, 2) == fsm.final_state
    assert fsm.next_state(fsm.start_state, 1) == fsm.start_state
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
        RegexFSM(regex_str, MockTokenizer())


def test_regex():
    class MockTokenizer:
        vocabulary = {"1": 1, "a": 2, "eos": 3}
        special_tokens = {"eos"}
        eos_token_id = 3

        def convert_token_to_string(self, token):
            return token

    regex_str = "[1-9]"
    tokenizer = MockTokenizer()

    with pytest.warns(UserWarning):
        fsm = RegexFSM(regex_str, tokenizer)

    assert fsm.states_to_token_maps == {0: {1: 1}}
    assert_expected_tensor_ids(fsm.allowed_token_ids(state=0), [1])
    assert fsm.next_state(state=0, token_id=1) == 1
    assert fsm.next_state(state=0, token_id=tokenizer.eos_token_id) == -1

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

    with pytest.warns(UserWarning):
        fsm = RegexFSM(regex_str, tokenizer)

    state = fsm.next_state(state=4, token_id=103)
    assert state == 5
    assert fsm.is_final_state(state)

    state = fsm.next_state(state=5, token_id=103)
    assert fsm.is_final_state(state)
