import pytest

from vllm.sampling_params import SamplingParams

import outlines.models as models
import outlines.generate as generate
import outlines.samplers as samplers


@pytest.fixture(scope="module")
def model():
    return models.vllm("gpt2", gpu_memory_utilization=0.5)


@pytest.mark.parametrize(
    "generator_type,params",
    ((generate.text, []), (generate.regex, ("[0-9]",)))
)
def test_outlines_generation_api(model, generator_type, params):
    generator = generator_type(model, *params)

    res = generator("test")
    assert isinstance(res, str)

    res = generator("test", max_tokens=10)
    assert isinstance(res, str)

    res = generator("test", stop_at=".")
    assert isinstance(res, str)

    res = generator("test", stop_at=[".", "ab"])
    assert isinstance(res, str)

    res = generator("test", stop_at=[".", "ab"])
    assert isinstance(res, str)

    res1 = generator("test", seed=1)
    res2 = generator("test", seed=1)
    assert isinstance(res1, str)
    assert isinstance(res2, str)
    assert res1 == res2

    res = generator(["test", "test1"])
    assert len(res) == 2


def test_vllm_sampling_params(model):
    generator = generate.text(model)

    sampling_params = SamplingParams(n=2)
    res = generator("test", sampling_params=sampling_params)
    assert len(res) == 2
    assert isinstance(res[0], str)
    assert isinstance(res[1], str)

    sampling_params = SamplingParams(seed=2)
    res1 = generator("test", sampling_params=sampling_params)
    res2 = generator("test", sampling_params=sampling_params)
    assert res1 == res2


def test_vllm_greedy_sampling(model):
    sampler = samplers.greedy()
    generator = generate.text(model, sampler)
    res = generator("test")
    assert isinstance(res, str)


def test_vllm_multinomial_sampling(model):
    sampler = samplers.multinomial()
    generator = generate.text(model, sampler)
    res = generator("test")
    assert isinstance(res, str)

    sampler = samplers.multinomial(3)
    generator = generate.text(model, sampler)
    res = generator("test")
    assert len(res) == 3
    assert isinstance(res[0], str)
    assert isinstance(res[1], str)

    sampler = samplers.multinomial(2, top_k=1)
    generator = generate.text(model, sampler)
    res = generator("test")
    assert res[0] == res[1]

    sampler = samplers.multinomial(1, top_p=.5)
    generator = generate.text(model, sampler)
    res = generator("test")
    assert isinstance(res, str)

    sampler = samplers.multinomial(2, temperature=.00001)
    generator = generate.text(model, sampler)
    res = generator("test")
    assert res[0] == res[1]


def test_vllm_beam_search(model):
    sampler = samplers.beam_search(1)
    generator = generate.text(model, sampler)
    res1 = generator("test")
    sampler = samplers.greedy()
    generator = generate.text(model, sampler)
    res2 = generator("test")
    assert res1 == res2

    sampler = samplers.beam_search(2)
    generator = generate.text(model, sampler)
    res = generator("test")
    assert len(res) == 2
    assert res[0] != res[1]


@pytest.mark.xfail(reason="CFG logits processor not available for vLLM")
def test_cfg_simple(model):
    generator = generate.cfg(model)
    res = generator("test")
