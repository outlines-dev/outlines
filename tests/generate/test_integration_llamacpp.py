import datetime
import re
from typing import Dict, Optional, Union

import pytest
from pydantic import BaseModel, constr

import outlines.generate as generate
import outlines.grammars as grammars
import outlines.models as models
import outlines.samplers as samplers

TEST_MODEL = "./llama-test-model/TinyMistral-248M-v2-Instruct.Q4_K_M.gguf"


@pytest.fixture(scope="session")
def model(tmp_path_factory):
    return models.llamacpp(
        repo_id="M4-ai/TinyMistral-248M-v2-Instruct-GGUF",
        filename="TinyMistral-248M-v2-Instruct.Q4_K_M.gguf",
    )


@pytest.mark.parametrize(
    "generator_type,params",
    (
        (generate.text, []),
        (generate.regex, ("[0-9]",)),
        (generate.cfg, (grammars.arithmetic,)),
    ),
)
def test_llamacpp_generation_api(model, generator_type, params):
    generator = generator_type(model, *params)

    res = generator("test", max_tokens=10)
    assert isinstance(res, str)

    res = generator("test", max_tokens=10)
    assert isinstance(res, str)

    res = generator("test", stop_at=".")
    assert isinstance(res, str)

    res = generator("test", stop_at=[".", "ab"])
    assert isinstance(res, str)

    res = generator("test", stop_at=[".", "ab"])
    assert isinstance(res, str)

    res1 = generator("test", seed=1, max_tokens=10)
    res2 = generator("test", seed=1, max_tokens=10)
    assert isinstance(res1, str)
    assert isinstance(res2, str)
    assert res1 == res2


def test_llama_cpp_streaming_api(model):
    generator = generate.text(model)
    token_generator = generator.stream("test", max_tokens=10)
    tokens = [token for token in token_generator]
    assert len(tokens) <= 10
    assert isinstance(tokens[0], str)


@pytest.mark.xfail(reason="Batch inference is not available in `llama-cpp-python`.")
def test_llamacpp_batch_inference(model):
    generator = generate.text(model)
    res = generator(["test", "test1"])
    assert len(res) == 2


def test_llamacpp_sampling_params(model):
    generator = generate.text(model)

    params = {
        "frequency_penalty": 1.0,
        "presence_penalty": 1.0,
    }
    res = generator("test", seed=1, max_tokens=10, **params)
    assert isinstance(res, str)


def test_llamacpp_greedy_sampling(model):
    sampler = samplers.greedy()
    generator = generate.text(model, sampler)
    res = generator("test", max_tokens=20)
    assert isinstance(res, str)


def test_llamacpp_multinomial_sampling(model):
    sampler = samplers.multinomial()
    generator = generate.text(model, sampler)
    res = generator("test", max_tokens=10)
    assert isinstance(res, str)

    sampler = samplers.multinomial(1, temperature=1.0)
    generator = generate.text(model, sampler)
    res = generator("test", max_tokens=10)
    assert isinstance(res, str)

    sampler = samplers.multinomial(1, top_k=1)
    generator = generate.text(model, sampler)
    res = generator("test", max_tokens=10)
    assert isinstance(res, str)

    sampler = samplers.multinomial(1, top_p=0.5)
    generator = generate.text(model, sampler)
    res = generator("test", max_tokens=10)
    assert isinstance(res, str)


def test_llamacpp_several_samples(model):
    sampler = samplers.multinomial(3)
    generator = generate.text(model, sampler)
    with pytest.raises(NotImplementedError, match="allow to take several samples"):
        generator("test")


def test_llamacpp_beam_search(model):
    sampler = samplers.beam_search(1)
    generator = generate.text(model, sampler)

    with pytest.raises(NotImplementedError, match="does not support Beam Search"):
        generator("test")


def test_llamacpp_text_stop(model):
    prompt = (
        "<|im_start|>user\nWrite a short sentence<|im_end|>\n<|im_start|>assistant\n"
    )
    sequence = generate.text(model)(prompt, stop_at="a", max_tokens=100)
    assert isinstance(sequence, str)
    assert sequence.find("a") == -1


def test_llamacpp_regex(model):
    prompt = (
        "<|im_start|>user\nWrite an email address<|im_end|>\n<|im_start|>assistant\n"
    )
    regex_str = r"([a-z]{10})@([a-z]{5})\.([a-z]{3})"
    generator = generate.regex(model, regex_str)

    # One prompt
    sequence = generator(prompts=prompt)
    assert isinstance(sequence, str)
    assert re.fullmatch(pattern=regex_str, string=sequence) is not None


def test_llamacpp_integer(model):
    prompt = (
        "<|im_start|>user\nWrite a short sentence<|im_end|>\n<|im_start|>assistant\n"
    )
    sequence = generate.format(model, int)(prompt, max_tokens=10)
    assert isinstance(sequence, int)
    assert sequence != ""
    int(sequence)


def test_llamacpp_float(model):
    prompt = (
        "<|im_start|>user\nWrite a short sentence<|im_end|>\n<|im_start|>assistant\n"
    )
    sequence = generate.format(model, float)(prompt, max_tokens=10)
    assert isinstance(sequence, float)

    assert sequence != ""
    float(sequence)


def test_llamacpp_bool(model):
    prompt = (
        "<|im_start|>user\nIs this True or False?<|im_end|>\n<|im_start|>assistant\n"
    )
    sequence = generate.format(model, bool)(prompt, max_tokens=10)
    assert isinstance(sequence, bool)

    assert sequence != ""
    bool(sequence)


def test_llamacpp_date(model):
    prompt = (
        "<|im_start|>user\nWhat day is it today?<|im_end|>\n<|im_start|>assistant\n"
    )
    sequence = generate.format(model, datetime.date)(prompt, max_tokens=20, seed=10)
    assert isinstance(sequence, datetime.date)


def test_llamacpp_time(model):
    prompt = "<|im_start|>user\nWhat time is it?<|im_end|>\n<|im_start|>assistant\n"
    sequence = generate.format(model, datetime.time)(prompt, max_tokens=10)
    assert isinstance(sequence, datetime.time)


def test_llamacpp_datetime(model):
    prompt = "<|im_start|>user\nWhat time is it?<|im_end|>\n<|im_start|>assistant\n"
    sequence = generate.format(model, datetime.datetime)(prompt, max_tokens=20)
    assert isinstance(sequence, datetime.datetime)


def test_llamacpp_choice(model):
    prompt = (
        "<|im_start|>user\nWrite a short sentence<|im_end|>\n<|im_start|>assistant\n"
    )
    sequence = generate.choice(model, ["test", "choice"])(prompt)
    assert sequence == "test" or sequence == "choice"


def test_llamacpp_json_basic(model):
    prompt = "<|im_start|>user\nOutput some JSON<|im_end|>\n<|im_start|>assistant\n"

    class Spam(BaseModel):
        spam: constr(max_length=10)
        fuzz: bool

    result = generate.json(model, Spam, whitespace_pattern="")(
        prompt, max_tokens=100, temperature=0.0, seed=1
    )
    assert isinstance(result, BaseModel)
    assert isinstance(result.spam, str)
    assert isinstance(result.fuzz, bool)
    assert len(result.spam) <= 10


def test_llamacpp_json_schema(model):
    prompt = "<|im_start|>user\nOutput some JSON<|im_end|>\n<|im_start|>assistant\n"

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

    result = generate.json(model, schema, whitespace_pattern="")(
        prompt, max_tokens=100, temperature=0, seed=10
    )
    assert isinstance(result, dict)
    assert isinstance(result["foo"], bool)
    assert isinstance(result["bar"], str)


def test_llamacpp_json_dict(model):
    prompt = "<|im_start|>user\nOutput some JSON<|im_end|>\n<|im_start|>assistant\n"

    schema_dict = {
        "properties": {
            "user_id": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "title": "User Id",
            },
            "name": {
                "additionalProperties": {"type": "integer"},
                "title": "Name",
                "type": "object",
            },
            "password": {
                "anyOf": [{"type": "string"}, {"type": "integer"}],
                "title": "Password",
            },
        },
        "required": ["user_id", "name", "password"],
        "title": "UserPydantic",
        "type": "object",
    }

    result = generate.json(model, schema_dict, whitespace_pattern="")(
        prompt, max_tokens=100, temperature=0, seed=10
    )
    assert isinstance(result, dict)
    assert isinstance(result["foo"], bool)
    assert isinstance(result["bar"], str)


def test_json_equivalence(model):
    """Test that all methods of generating from json create the same fsm."""

    # Different inputs for json generator
    def user_callable(
        user_id: Optional[int], name: Dict[str, int], password: Union[str, int]
    ):
        pass

    class UserPydantic(BaseModel):
        user_id: Optional[int]
        name: Dict[str, int]
        password: Union[str, int]

    user_dict = {
        "properties": {
            "user_id": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "title": "User Id",
            },
            "name": {
                "additionalProperties": {"type": "integer"},
                "title": "Name",
                "type": "object",
            },
            "password": {
                "anyOf": [{"type": "string"}, {"type": "integer"}],
                "title": "Password",
            },
        },
        "required": ["user_id", "name", "password"],
        "title": "User",
        "type": "object",
    }

    user_str = '{"properties": {"user_id": {"anyOf": [{"type": "integer"}, {"type": "null"}], "title": "User Id"}, "name": {"additionalProperties": {"type": "integer"}, "title": "Name", "type": "object"}, "password": {"anyOf": [{"type": "string"}, {"type": "integer"}], "title": "Password"}}, "required": ["user_id", "name", "password"], "title": "User", "type": "object"}'

    # Initialize the generators
    generator_callable = generate.json(model, user_callable)
    generator_pydantic = generate.json(model, UserPydantic)
    generator_dict = generate.json(model, user_dict)
    generator_str = generate.json(model, user_str)

    # Check finite state machines are the same
    assert (
        generator_callable.logits_processor.fsm.states_to_token_maps
        == generator_pydantic.logits_processor.fsm.states_to_token_maps
        == generator_dict.logits_processor.fsm.states_to_token_maps
        == generator_str.logits_processor.fsm.states_to_token_maps
    )
    assert (
        generator_callable.logits_processor.fsm.empty_token_ids
        == generator_pydantic.logits_processor.fsm.empty_token_ids
        == generator_dict.logits_processor.fsm.empty_token_ids
        == generator_str.logits_processor.fsm.empty_token_ids
    )
    assert (
        generator_callable.logits_processor.fsm.eos_token_id
        == generator_pydantic.logits_processor.fsm.eos_token_id
        == generator_dict.logits_processor.fsm.eos_token_id
        == generator_str.logits_processor.fsm.eos_token_id
    )
    assert (
        generator_callable.logits_processor.fsm.final_states
        == generator_pydantic.logits_processor.fsm.final_states
        == generator_dict.logits_processor.fsm.final_states
        == generator_str.logits_processor.fsm.final_states
    )


def test_llamacpp_cfg(model):
    prompt = "<|im_start|>user\nOutput a short and valid JSON object with two keys.<|im_end|>\n><|im_start|>assistant\n"
    result = generate.cfg(model, grammars.arithmetic)(prompt, seed=11)
    assert isinstance(result, str)


@pytest.mark.parametrize(
    "repo,model_path,hf_tokenizer_uri",
    [
        ("Qwen/Qwen1.5-0.5B-Chat-GGUF", "*q2*.gguf", "Qwen/Qwen1.5-0.5B-Chat"),
        ("TheBloke/phi-2-GGUF", "*Q2*.gguf", "microsoft/phi-2"),
    ],
)
def test_byte_tokenizer_regression(repo, model_path, hf_tokenizer_uri):
    """Reproduce https://github.com/outlines-dev/outlines/issues/820"""
    import llama_cpp

    model = models.llamacpp(
        repo,
        model_path,
        tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(
            hf_tokenizer_uri
        ),
    )
    generator = generate.choice(model, ["skirt", "dress", "pen", "jacket"])
    generator("Pick the odd word out: skirt, dress, pen, jacket")


def test_llama_cpp_pre_tokenizer_remains_broken():
    """If fails, llama.cpp pre-tokenizer is fixed -> revert #892, remove `with pytest.raises`"""
    repo = "Qwen/Qwen1.5-0.5B-Chat-GGUF"
    model_path = "*q2*.gguf"

    model = models.llamacpp(repo, model_path)
    with pytest.raises(RuntimeError):
        generate.choice(model, ["skirt", "dress", "pen", "jacket"])
