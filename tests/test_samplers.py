import math

import pytest
import torch

from outlines.samplers import (
    BeamSearchSampler,
    GreedySampler,
    MultinomialSampler,
    ParticleFilter,
    beam_search,
    greedy,
    keep_top_k_logits,
    keep_top_p_logits,
    multinomial,
    multinomial_resampling,
    rescale_logits,
)


def compute_logprobs(logits):
    return torch.nn.functional.log_softmax(logits, dim=-1)


def test_aliases():
    assert greedy == GreedySampler
    assert multinomial == MultinomialSampler
    assert beam_search == BeamSearchSampler


def test_greedy():
    sampler = GreedySampler()
    logits = torch.tensor([[1.0, 2.0, 5.0]])
    weights = torch.tensor([0])
    next_token_ids, ancestors, weights = sampler(None, logits, weights, None)

    logprobs = compute_logprobs(logits)
    assert next_token_ids.equal(torch.tensor([[2]]))
    assert ancestors.equal(torch.tensor([0]))
    assert weights.equal(logprobs[..., 2])

    sampler = GreedySampler()
    logits = torch.tensor([[10.0, 0.0, 3.0], [-math.inf, 2.0, 5.0]])
    weights = torch.tensor([0, 0])
    next_token_ids, ancestors, weights = sampler(None, logits, weights, None)

    logprobs = compute_logprobs(logits)
    assert next_token_ids.equal(torch.tensor([[0], [2]]))
    assert ancestors.equal(torch.tensor([0, 1]))
    assert weights.equal(torch.tensor([logprobs[0, 0], logprobs[1, 2]]))


def test_multinomial():
    rng = torch.Generator()
    rng.manual_seed(239)

    sampler = MultinomialSampler()
    logits = torch.tensor([[1.0, 4.0, 5.0]])
    weights = torch.tensor([0])
    next_token_ids, ancestors, weights = sampler(None, logits, weights, rng)

    logprobs = compute_logprobs(logits)
    assert next_token_ids.equal(torch.tensor([[2]]))
    assert ancestors.equal(torch.tensor([0]))
    assert weights.equal(torch.tensor([logprobs[:, 2]]))

    sampler = MultinomialSampler()
    logits = torch.tensor([[10.0, 0.0, 9.0], [-math.inf, 4.0, 5.0]])
    weights = torch.tensor([0, 0])
    next_token_ids, ancestors, weights = sampler(None, logits, weights, rng)

    logprobs = compute_logprobs(logits)
    assert next_token_ids.equal(torch.tensor([[0], [2]]))
    assert ancestors.equal(torch.tensor([0, 1]))
    assert weights.equal(torch.tensor([logprobs[0, 0], logprobs[1, 2]]))


def test_multinomial_init():
    sampler = MultinomialSampler()
    assert sampler.logits_processors == []

    sampler = MultinomialSampler(3)
    assert sampler.logits_processors == []

    sampler = MultinomialSampler(top_k=1)
    assert len(sampler.logits_processors) == 1

    sampler = MultinomialSampler(top_p=0.95)
    assert len(sampler.logits_processors) == 1

    sampler = MultinomialSampler(top_k=1, top_p=0.95)
    assert len(sampler.logits_processors) == 1

    sampler = MultinomialSampler(temperature=1.0)
    assert len(sampler.logits_processors) == 1

    sampler = MultinomialSampler(top_k=1, temperature=1.0)
    assert len(sampler.logits_processors) == 2

    sampler = MultinomialSampler(top_p=0.95, temperature=1.0)
    assert len(sampler.logits_processors) == 2


def test_top_k():
    with pytest.raises(ValueError, match="`k` must be a strictly"):
        keep_top_k_logits(-1)

    with pytest.raises(ValueError, match="`k` must be a strictly"):
        keep_top_k_logits(0.1)

    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    logits_processor = keep_top_k_logits(1)
    result = logits_processor(logits)
    assert result.equal(torch.tensor([[-math.inf, -math.inf, -math.inf, 4.0]]))

    logits_processor = keep_top_k_logits(10)
    result = logits_processor(logits)
    assert result.equal(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    logits_processor = keep_top_k_logits(2)
    result = logits_processor(logits)
    assert result.equal(
        torch.tensor(
            [[-math.inf, -math.inf, 3.0, 4.0], [-math.inf, -math.inf, 7.0, 8.0]]
        )
    )


def test_top_p():
    with pytest.raises(ValueError, match="`p` must be a floating point"):
        keep_top_p_logits(-0.1)

    with pytest.raises(ValueError, match="`p` must be a floating point"):
        keep_top_p_logits(0.0)

    with pytest.raises(ValueError, match="`p` must be a floating point"):
        keep_top_p_logits(1.1)

    logits = torch.tensor([[1.0, 1.01, 1.02, 4.0]])

    logits_processor = keep_top_p_logits(0.1)
    result = logits_processor(logits)
    assert result.equal(torch.tensor([[-math.inf, -math.inf, -math.inf, 4.0]]))

    logits_processor = keep_top_p_logits(0.95)
    result = logits_processor(logits)
    assert result.equal(torch.tensor([[-math.inf, 1.01, 1.02, 4.0]]))

    logits_processor = keep_top_p_logits(1.0)
    result = logits_processor(logits)
    assert result.equal(torch.tensor([[1.0, 1.01, 1.02, 4.0]]))

    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

    logits_processor = keep_top_p_logits(0.1)
    result = logits_processor(logits)
    assert result.equal(
        torch.tensor(
            [
                [-math.inf, -math.inf, -math.inf, 4.0],
                [-math.inf, -math.inf, -math.inf, 8.0],
            ]
        )
    )

    logits_processor = keep_top_p_logits(0.95)
    result = logits_processor(logits)
    assert result.equal(
        torch.tensor(
            [
                [-math.inf, 2.0, 3.0, 4.0],
                [-math.inf, 6.0, 7.0, 8.0],
            ]
        )
    )

    logits_processor = keep_top_p_logits(1.0)
    result = logits_processor(logits)
    assert result.equal(
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ]
        )
    )


def test_rescale():
    with pytest.raises(ValueError, match="`temperature` must"):
        rescale_logits(1)

    with pytest.raises(ValueError, match="`temperature` must"):
        rescale_logits(-0.1)

    with pytest.raises(ValueError, match="Please use the greedy sampler"):
        rescale_logits(0.0)


def test_beam_search():
    # Two beams, single sequence
    sampler = BeamSearchSampler(2)
    logits = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    init_weights = torch.tensor([0, 1.0])
    next_token_ids, ancestors, weights = sampler(None, logits, init_weights, None)

    logprobs = compute_logprobs(logits)
    assert next_token_ids.equal(torch.tensor([[0], [1]]))
    assert ancestors.equal(torch.tensor([1, 0]))
    assert weights.equal(
        torch.tensor([init_weights[1] + logprobs[1][0], logprobs[0][1]])
    )

    # Make sure that initial samples are different
    sampler = BeamSearchSampler(2)
    logits = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    init_weights = torch.tensor([0, 0])
    next_token_ids, ancestors, weights = sampler(None, logits, init_weights, None)

    logprobs = compute_logprobs(logits)
    assert next_token_ids.equal(torch.tensor([[1], [0]]))
    assert ancestors.equal(torch.tensor([0, 0]))
    assert weights.equal(torch.tensor([logprobs[0][1], logprobs[0][0]]))

    # One beam, batch of two sequences. Reduces to Greedy Search.
    sampler = BeamSearchSampler(1)
    logits = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    weights = torch.tensor([0, 0])
    next_token_ids, ancestors, weights = sampler(None, logits, weights, None)

    logprobs = compute_logprobs(logits)
    assert next_token_ids.equal(torch.tensor([[1], [0]]))
    assert ancestors.equal(torch.tensor([0, 1]))
    assert weights.equal(torch.tensor([logprobs[0][1], logprobs[1][0]]))

    # Two beams, batch of two sequences
    sampler = BeamSearchSampler(2)
    logits = torch.tensor([[0.0, 1.0], [2.0, 0.0], [3.0, 2.0], [0.0, 1.0]])
    init_weights = torch.tensor([0, 0, 2.0, 0])
    next_token_ids, ancestors, weights = sampler(None, logits, init_weights, None)

    logprobs = compute_logprobs(logits)
    assert next_token_ids.equal(torch.tensor([[0], [1], [0], [1]]))
    assert ancestors.equal(torch.tensor([1, 0, 2, 2]))
    assert weights.equal(
        torch.tensor(
            [
                logprobs[1][0],
                logprobs[0][1],
                init_weights[2] + logprobs[2][0],
                init_weights[2] + logprobs[2][1],
            ]
        )
    )


def test_multinomial_resampling():
    rng = torch.Generator()
    rng.manual_seed(239)

    weights = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    num_samples = 2
    weights, indices = multinomial_resampling(rng, weights, num_samples)
    assert indices.equal(torch.tensor([[2, 0], [2, 1]]))
    assert weights.equal(torch.tensor([[3.0, 1.0], [6.0, 5.0]]))

    weights = torch.tensor([[-math.inf, 1.0]])
    num_samples = 1
    weights, indices = multinomial_resampling(rng, weights, num_samples)
    assert indices.equal(torch.tensor([[1]]))
    assert indices.equal(torch.tensor([[1.0]]))


def test_particle_filter():
    # If masked_logits = logits then equal probability
    rng = torch.Generator()
    rng.manual_seed(239)

    sampler = ParticleFilter(2)
    logits = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    biased_logits = logits
    init_weights = None
    next_token_ids, ancestors, weights = sampler(
        logits, biased_logits, init_weights, rng
    )
    assert next_token_ids.equal(torch.tensor([[0], [1]]))
    assert ancestors.equal(torch.tensor([0, 1]))
    assert weights.equal(torch.tensor([0, 0]))

    rng = torch.Generator()
    rng.manual_seed(239)

    sampler = ParticleFilter(1)
    logits = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    biased_logits = logits
    init_weights = None
    next_token_ids, ancestors, weights = sampler(
        logits, biased_logits, init_weights, rng
    )
    assert next_token_ids.equal(torch.tensor([[0], [1]]))
    assert ancestors.equal(torch.tensor([0, 1]))
    assert weights.equal(torch.tensor([0, 0]))

    rng = torch.Generator()
    rng.manual_seed(239)

    sampler = ParticleFilter(2)
    logits = torch.tensor([[0.0, 1.0], [2.0, 0.0], [3.0, 2.0], [0.0, 1.0]])
    biased_logits = torch.tensor([[0.0, 1.0], [2.0, 0.0], [3.0, 2.0], [0.0, 1.0]])
    init_weights = None
    next_token_ids, ancestors, weights = sampler(
        logits, biased_logits, init_weights, rng
    )

    assert next_token_ids.equal(torch.tensor([[0], [1], [1], [0]]))
    assert ancestors.equal(torch.tensor([0, 1, 2, 3]))

    # If probability of masked logits is really low for one sequence, then only
    # the other one should be resampled.
    rng = torch.Generator()
    rng.manual_seed(239)

    sampler = ParticleFilter(2)
    logits = torch.tensor([[1.0, 10.0, 10.0], [10.0, 1.0, 1.0]])
    biased_logits = torch.tensor([[-math.inf, 1.0, 1.0], [-math.inf, 1.0, 1.0]])
    init_weights = None
    next_token_ids, ancestors, weights = sampler(
        logits, biased_logits, init_weights, rng
    )
    assert next_token_ids.equal(torch.tensor([[1], [2]]))
    assert ancestors.equal(torch.tensor([0, 0]))
