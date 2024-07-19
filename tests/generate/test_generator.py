import math
from typing import Generator

import pytest
import torch

from outlines.fsm.guide import Generate
from outlines.generate.api import SequenceGenerator
from outlines.generate.generator import (
    bias_logits,
    get_allowed_tokens,
    get_next_fsm_states,
    is_generation_finished,
    sequence_generator,
    update_attention_masks,
    update_token_ids,
)


def test_sequence_generator_class():
    class MockFSM:
        first_state = 0

        def align_prompt_tokens(self, prompt):
            return prompt

        def get_next_state(self, state, next_token_ids):
            return 4

        def get_next_instruction(self, *_):
            return Generate([4])

        def is_final_state(self, _):
            return True

        def copy(self):
            return self

    class MockTokenizer:
        def encode(self, _):
            # Input: "test"
            return torch.tensor([[0, 1, 2, 3]]), torch.tensor([[1, 1, 1, 1]])

        def decode(self, tokens):
            return ["".join(["testx"[int(i)] for i in tokens[0]])]

    class MockModel:
        def __init__(self):
            self.tokenizer = MockTokenizer()

        def __call__(*_):
            return torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.float), None

    class sampler:
        def __init__(self):
            self.samples = 1

        def __call__(self, biased_logits, *_):
            return torch.argmax(biased_logits, keepdims=True), torch.tensor([0]), None

    # Stream
    generator = SequenceGenerator(MockFSM(), MockModel(), sampler(), "cpu")
    assert generator.device == "cpu"
    assert isinstance(generator.tokenizer, MockTokenizer)
    assert isinstance(generator.fsm, MockFSM)

    sequence = generator.stream("test")
    assert isinstance(sequence, Generator)

    next(sequence)

    with pytest.raises(StopIteration):
        next(sequence)

    # Call
    generator = SequenceGenerator(MockFSM(), MockModel(), sampler(), "cpu")
    result = generator("test")
    assert result == "x"


def test_sequence_generator_1d_single_iteration():
    class MockFSM:
        def align_prompt_tokens(self, prompt):
            return prompt

        def get_next_state(self, state, next_token_ids):
            return 0

        def get_next_instruction(self, _):
            return Generate([0, 1, 2, 3])

        def is_final_state(self, _):
            return True

        def copy(self):
            return self

    class MockTokenizer:
        def encode(self, _):
            return torch.tensor([[0, 1, 2, 3]]), torch.tensor([[1, 1, 1, 1]])

        def decode(self, x):
            return x

    class MockModel:
        def __init__(self):
            self.tokenizer = MockTokenizer()

        def __call__(*_):
            return torch.tensor([[0, 1, 2, 3]], dtype=torch.float), None

    def sampler(biased_logits, *_):
        return torch.argmax(biased_logits, keepdims=True), torch.tensor([0]), None

    token_ids, attention_mask = (
        torch.tensor([[0, 1, 2, 3]]),
        torch.tensor([[1, 1, 1, 1]]),
    )
    init_fsm_states = [0]
    sequence = sequence_generator(
        MockModel(),
        sampler,
        [MockFSM()],
        token_ids,
        torch.tensor([0]),
        attention_mask,
        init_fsm_states,
        rng=torch.Generator(),
    )
    result = next(sequence)

    assert torch.equal(result.token_ids, torch.tensor([[0, 1, 2, 3, 3]]))
    assert torch.equal(result.logits, torch.tensor([[0, 1, 2, 3]]))

    with pytest.raises(StopIteration):
        next(sequence)


def test_sequence_generator_1d_several_iterations():
    class MockFSM:
        def align_prompt_tokens(self, prompt):
            return prompt

        def get_next_state(self, state, next_token_ids):
            return state + 1

        def get_next_instruction(self, _):
            return Generate([0, 1, 2, 3])

        def is_final_state(self, state):
            if state < 2:
                return False
            else:
                return True

        def copy(self):
            return self

    class MockTokenizer:
        def encode(self, _):
            return torch.tensor([[0, 1, 2, 3]]), torch.tensor([[1, 1, 1, 1]])

        def decode(self, x):
            return x

    class MockModel:
        def __init__(self):
            self.tokenizer = MockTokenizer()

        def __call__(*_):
            return torch.tensor([[0, 1, 2, 3]], dtype=torch.float), None

    def sampler(biased_logits, *_):
        return torch.argmax(biased_logits, keepdims=True), torch.tensor([0]), None

    token_ids, attention_mask = (
        torch.tensor([[0, 1, 2, 3]]),
        torch.tensor([[1, 1, 1, 1]]),
    )
    init_fsm_states = [0]
    sequence = sequence_generator(
        MockModel(),
        sampler,
        [MockFSM()],
        token_ids,
        torch.tensor([0]),
        attention_mask,
        init_fsm_states,
        rng=torch.Generator(),
    )

    result = next(sequence)
    assert torch.equal(result.token_ids, torch.tensor([[0, 1, 2, 3, 3]]))
    assert torch.equal(result.logits, torch.tensor([[0, 1, 2, 3]]))

    result = next(sequence)
    assert torch.equal(result.token_ids, torch.tensor([[0, 1, 2, 3, 3, 3]]))
    assert torch.equal(result.logits, torch.tensor([[0, 1, 2, 3]]))

    with pytest.raises(StopIteration):
        next(sequence)


def test_sequence_generator_2d_single_iteration():
    class MockFSM:
        def align_prompt_tokens(self, prompt):
            return prompt

        def get_next_state(self, state, next_token_ids):
            return 0

        def get_next_instruction(self, _):
            return Generate([0, 1, 2, 3])

        def is_final_state(self, _):
            return True

        def copy(self):
            return self

    class MockTokenizer:
        def encode(self, _):
            return torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]), torch.tensor(
                [[1, 1, 1, 1], [1, 1, 1, 1]]
            )

        def decode(self, x):
            return x

    class MockModel:
        def __init__(self):
            self.tokenizer = MockTokenizer()

        def __call__(*_):
            return torch.tensor([[0, 1, 2, 3], [4, 5, 7, 6]], dtype=torch.float), None

    def sampler(biased_logits, *_):
        return (
            torch.argmax(biased_logits, keepdims=True, dim=-1),
            torch.tensor([0, 1]),
            None,
        )

    token_ids, attention_mask = (
        torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]),
        torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]]),
    )
    init_fsm_states = [0, 0]
    fsms = [MockFSM(), MockFSM()]
    sequence = sequence_generator(
        MockModel(),
        sampler,
        fsms,
        token_ids,
        torch.tensor([0, 0]),
        attention_mask,
        init_fsm_states,
        rng=torch.Generator(),
    )

    result = next(sequence)
    assert torch.equal(
        result.token_ids, torch.tensor([[0, 1, 2, 3, 3], [4, 5, 6, 7, 2]])
    )
    assert torch.equal(
        result.logits, torch.tensor([[0, 1, 2, 3], [4, 5, 7, 6]], dtype=torch.float)
    )

    with pytest.raises(StopIteration):
        next(sequence)


def test_sequence_generator_2d_several_iterations():
    class MockFSM:
        def align_prompt_tokens(self, prompt):
            return prompt

        def get_next_state(self, state, next_token_ids):
            return state + 1

        def get_next_instruction(self, _):
            return Generate([0, 1, 2, 3])

        def is_final_state(self, state):
            if state < 2:
                return False
            else:
                return True

        def copy(self):
            return self

    class MockTokenizer:
        def encode(self, _):
            return torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]), torch.tensor(
                [[1, 1, 1, 1], [1, 1, 1, 1]]
            )

        def decode(self, x):
            return x

    class MockModel:
        def __init__(self):
            self.tokenizer = MockTokenizer()

        def __call__(*_):
            return torch.tensor([[0, 1, 2, 3], [4, 5, 7, 6]], dtype=torch.float), None

    def sampler(biased_logits, *_):
        return (
            torch.argmax(biased_logits, keepdims=True, dim=-1),
            torch.tensor([0, 1]),
            None,
        )

    token_ids, attention_mask = (
        torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]),
        torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]]),
    )
    init_fsm_states = [0, 0]
    fsms = [MockFSM(), MockFSM()]
    sequence = sequence_generator(
        MockModel(),
        sampler,
        fsms,
        token_ids,
        torch.tensor([0, 0]),
        attention_mask,
        init_fsm_states,
        rng=torch.Generator(),
    )

    result = next(sequence)
    assert torch.equal(
        result.token_ids, torch.tensor([[0, 1, 2, 3, 3], [4, 5, 6, 7, 2]])
    )
    assert torch.equal(
        result.logits, torch.tensor([[0, 1, 2, 3], [4, 5, 7, 6]], dtype=torch.float)
    )

    result = next(sequence)
    assert torch.equal(
        result.token_ids, torch.tensor([[0, 1, 2, 3, 3, 3], [4, 5, 6, 7, 2, 2]])
    )
    assert torch.equal(
        result.logits, torch.tensor([[0, 1, 2, 3], [4, 5, 7, 6]], dtype=torch.float)
    )

    with pytest.raises(StopIteration):
        next(sequence)


def test_get_next_fsm_states():
    class MockFSM:
        def get_next_state(self, state, next_token_ids):
            return 0

        def copy(self):
            return self

    result = get_next_fsm_states([MockFSM()], [0], torch.tensor([[0]]))
    assert result == [0]

    result = get_next_fsm_states(
        [MockFSM(), MockFSM()], [0, 0], torch.tensor([[0], [0]])
    )
    assert result == [0, 0]


def test_get_get_next_instructions():
    class MockFSM:
        def get_next_instruction(self, _):
            return Generate([1, 2, 3, 4])

    result = get_allowed_tokens([MockFSM()], [0])
    assert result == [[1, 2, 3, 4]]

    result = get_allowed_tokens([MockFSM(), MockFSM()], [0, 1])
    assert result == [[1, 2, 3, 4], [1, 2, 3, 4]]


def test_is_generation_finished():
    class MockFSMFinished:
        def is_final_state(self, _):
            return True

    result = is_generation_finished([MockFSMFinished(), MockFSMFinished()], [1, 1])
    assert result is True

    class MockFSMNotFinished:
        def is_final_state(self, state):
            if state == 0:
                return False
            else:
                return True

    result = is_generation_finished(
        [MockFSMNotFinished(), MockFSMNotFinished()], [0, 1]
    )
    assert result is False


@pytest.mark.parametrize(
    "token_ids,next_token_ids,ancestors,expected_result",
    [
        (
            torch.tensor([[1]]),
            torch.tensor([[2]]),
            torch.tensor([0]),
            torch.tensor([[1, 2]]),
        ),
        (
            torch.tensor([[1], [2]]),
            torch.tensor([[3], [4]]),
            torch.tensor([0, 1]),
            torch.tensor([[1, 3], [2, 4]]),
        ),
        (
            torch.tensor([[1], [2]]),
            torch.tensor([[3], [4]]),
            torch.tensor([0, 0]),
            torch.tensor([[1, 3], [1, 4]]),
        ),
        (
            torch.tensor([[1], [2]]),
            torch.tensor([[3], [4]]),
            torch.tensor([1, 0]),
            torch.tensor([[2, 3], [1, 4]]),
        ),
        (
            torch.tensor([[1, 2], [3, 5]]),
            torch.tensor([[3], [4]]),
            torch.tensor([1, 0]),
            torch.tensor([[3, 5, 3], [1, 2, 4]]),
        ),
    ],
)
def test_update_token_ids(token_ids, next_token_ids, ancestors, expected_result):
    result = update_token_ids(token_ids, next_token_ids, ancestors)
    assert torch.equal(result, expected_result)


@pytest.mark.parametrize(
    "attention_masks,ancestors,expected_result",
    [
        (
            torch.tensor([[1, 1]], dtype=torch.float),
            torch.tensor([0]),
            torch.tensor([[1, 1, 1]], dtype=torch.float),
        ),
        (
            torch.tensor([[1, 1], [1, 1]], dtype=torch.float),
            torch.tensor([0, 0]),
            torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.float),
        ),
        (
            torch.tensor([[0, 1], [1, 1]], dtype=torch.float),
            torch.tensor([0, 1]),
            torch.tensor([[0, 1, 1], [1, 1, 1]], dtype=torch.float),
        ),
        (
            torch.tensor([[0, 1], [1, 1]], dtype=torch.float),
            torch.tensor([1, 0]),
            torch.tensor([[1, 1, 1], [0, 1, 1]], dtype=torch.float),
        ),
        (
            torch.tensor([[0, 1], [1, 1]], dtype=torch.float),
            torch.tensor([1, 1]),
            torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.float),
        ),
    ],
)
def test_expand_attention_masks(attention_masks, ancestors, expected_result):
    result = update_attention_masks(attention_masks, ancestors)
    assert torch.equal(result, expected_result)


@pytest.mark.parametrize(
    "logits,indices_to_mask,expected",
    [
        (
            torch.tensor([[1, 2, 3, 4]], dtype=torch.float),
            [[0, 1, 2, 3]],
            torch.tensor([[1, 2, 3, 4]], dtype=torch.float),
        ),
        (
            torch.tensor([[1, 2, 3, 4]], dtype=torch.float),
            [[0, 2, 3]],
            torch.tensor([[1, -math.inf, 3, 4]], dtype=torch.float),
        ),
        (
            torch.tensor([[1, 2, 3, 4]], dtype=torch.float),
            [[0, 2]],
            torch.tensor([[1, -math.inf, 3, -math.inf]], dtype=torch.float),
        ),
        (
            torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float),
            [[1, 2], [0, 1]],
            torch.tensor([[-math.inf, 2, 3], [4, 5, -math.inf]], dtype=torch.float),
        ),
        (
            torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float),
            [[0, 2], [1]],
            torch.tensor(
                [[1, -math.inf, 3], [-math.inf, 5, -math.inf]], dtype=torch.float
            ),
        ),
    ],
)
def test_bias_logits(logits, indices_to_mask, expected):
    masked_logits = bias_logits(logits, indices_to_mask)
    assert torch.equal(masked_logits, expected)
