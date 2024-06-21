import datetime
import re

import pytest
import torch
from pydantic import BaseModel, constr
from vllm.sampling_params import SamplingParams

import outlines.generate as generate
import outlines.grammars as grammars
import outlines.models as models
import outlines.samplers as samplers

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="vLLM models can only be run on GPU."
)


@pytest.fixture(scope="module")
def model():
    return models.vllm("gpt2", gpu_memory_utilization=0.5)


@pytest.mark.parametrize(
    "generator_type,params", ((generate.text, []), (generate.regex, ("[0-9]",)))
)
def test_vllm_generation_api(model, generator_type, params):
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

    sampler = samplers.multinomial(1, top_p=0.5)
    generator = generate.text(model, sampler)
    res = generator("test")
    assert isinstance(res, str)

    sampler = samplers.multinomial(2, temperature=0.00001)
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


def test_vllm_text_stop(model):
    prompt = "Write a short sentence containing 'You': "
    sequence = generate.text(model)(prompt, max_tokens=100, seed=10)
    assert sequence.find("news") != -1

    sequence = generate.text(model)(prompt, stop_at="news", max_tokens=100, seed=10)
    assert isinstance(sequence, str)
    assert sequence.find("news") == -1


def test_vllm_regex(model):
    prompt = "Write an email address: "
    regex_str = r"([a-z]{10})@([a-z]{5})\.([a-z]{3})"
    generator = generate.regex(model, regex_str)

    # One prompt
    sequence = generator(prompts=prompt)
    assert isinstance(sequence, str)
    assert re.fullmatch(pattern=regex_str, string=sequence) is not None


def test_vllm_integer(model):
    prompt = "Give me an integer: "
    sequence = generate.format(model, int)(prompt, max_tokens=10)
    assert isinstance(sequence, int)
    assert sequence != ""
    int(sequence)


def test_vllm_float(model):
    prompt = "Give me a floating-point number: "
    sequence = generate.format(model, float)(prompt, max_tokens=10)
    assert isinstance(sequence, float)

    assert sequence != ""
    float(sequence)


def test_vllm_bool(model):
    prompt = "Is this True or False? "
    sequence = generate.format(model, bool)(prompt, max_tokens=10)
    assert isinstance(sequence, bool)

    assert sequence != ""
    bool(sequence)


def test_vllm_date(model):
    prompt = "What day is it today? "
    sequence = generate.format(model, datetime.date)(prompt, max_tokens=10)
    assert isinstance(sequence, datetime.date)


def test_vllm_time(model):
    prompt = "What time is it? "
    sequence = generate.format(model, datetime.time)(prompt, max_tokens=10)
    assert isinstance(sequence, datetime.time)


def test_vllm_datetime(model):
    prompt = "What time is it? "
    sequence = generate.format(model, datetime.datetime)(prompt, max_tokens=20)
    assert isinstance(sequence, datetime.datetime)


def test_vllm_choice(model):
    prompt = "Which one between 'test' and 'choice'? "
    sequence = generate.choice(model, ["test", "choice"])(prompt)
    assert sequence == "test" or sequence == "choice"


def test_vllm_json_basic(model):
    prompt = "Output some JSON. "

    class Spam(BaseModel):
        spam: constr(max_length=10)
        fuzz: bool

    sampling_params = SamplingParams(temperature=0)
    result = generate.json(model, Spam, whitespace_pattern="")(
        prompt, max_tokens=100, seed=1, sampling_params=sampling_params
    )
    assert isinstance(result, BaseModel)
    assert isinstance(result.spam, str)
    assert isinstance(result.fuzz, bool)
    assert len(result.spam) <= 10


def test_vllm_json_schema(model):
    prompt = "Output some JSON. "

    schema = """{
      "title": "spam",
      "type": "object",
      "properties": {
           "foo" : {"type": "boolean"},
           "bar": {"type": "string", "maxLength": 4}
        },
      "required": ["foo", "bar"]
      }
    """

    sampling_params = SamplingParams(temperature=0)
    result = generate.json(model, schema, whitespace_pattern="")(
        prompt, max_tokens=100, seed=10, sampling_params=sampling_params
    )
    assert isinstance(result, dict)
    assert isinstance(result["foo"], bool)
    assert isinstance(result["bar"], str)


@pytest.mark.xfail(
    reason="The CFG logits processor for vLLM has not been implemented yet."
)
def test_vllm_cfg(model):
    prompt = "<|im_start|>user\nOutput a short and valid JSON object with two keys.<|im_end|>\n><|im_start|>assistant\n"
    result = generate.cfg(model, grammars.arithmetic)(prompt, seed=11)
    assert isinstance(result, str)
