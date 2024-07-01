import datetime
import re

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
        # (generate.cfg, (grammars.arithmetic,)),  # Awaiting CFG fix
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


def test_llamacpp_cfg(model):
    prompt = "<|im_start|>user\nOutput a short and valid JSON object with two keys.<|im_end|>\n><|im_start|>assistant\n"

    # remove this statement once cfg is implemented
    with pytest.raises(NotImplementedError):
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


def test_RegexGuide_caching(model, temp_cache_dir):
    import llama_cpp

    import outlines.caching
    from outlines.fsm.guide import create_states_mapping

    assert outlines.caching._caching_enabled

    regex = r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)"
    prompt = "What is the IP address of the Google DNS servers? "

    cache = outlines.caching.get_cache()

    # Returns (hits, misses)
    _ = cache.stats(enable=True)
    assert cache.statistics

    assert create_states_mapping.__memory__ is cache

    generator = generate.regex(model, regex, sampler=samplers.greedy())
    assert cache.stats() == (0, 1)

    model_2 = models.llamacpp(
        "Qwen/Qwen1.5-0.5B-Chat-GGUF",
        "*q2*.gguf",
        tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(
            "Qwen/Qwen1.5-0.5B-Chat"
        ),
    )
    generator_2 = generate.regex(model_2, regex, sampler=samplers.greedy())
    assert cache.stats() == (0, 2)

    # These two different models and tokenizers should not have the same state
    # mapping results
    assert (
        generator.logits_processor.fsm.states_to_token_maps
        != generator_2.logits_processor.fsm.states_to_token_maps
    )

    generator_3 = generate.regex(model_2, regex, sampler=samplers.greedy())
    assert cache.stats() == (1, 2)
    assert (
        generator_2.logits_processor.fsm.states_to_token_maps
        == generator_3.logits_processor.fsm.states_to_token_maps
    )

    # Just for fun...
    structured = generator(prompt, max_tokens=30)
    structured_2 = generator_2(prompt, max_tokens=30)

    assert re.fullmatch(regex, structured)
    assert re.fullmatch(regex, structured_2)
    assert structured != structured_2


def test_tokenizer_vocabulary_decode_sanity():
    """Assert the decoded newline token (198) is the same as the normalized vocab token"""
    import llama_cpp

    model = models.llamacpp(
        "bartowski/Meta-Llama-3-8B-Instruct-GGUF",
        "Meta-Llama-3-8B-Instruct-IQ1_M.gguf",
        tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(
            "NousResearch/Hermes-2-Pro-Llama-3-8B"
        ),
    )
    tokenizer = generate.regex(model, "a").logits_processor.tokenizer

    decoded_nl_token = tokenizer.decode([198])[0]
    vocab_nl_token = tokenizer.convert_token_to_string(
        [token for token, token_id in tokenizer.vocabulary.items() if token_id == 198][
            0
        ]
    )
    assert decoded_nl_token == vocab_nl_token


def test_no_length_constraint_when_unset():
    """Assert that models.llamacpp doesn't have an implicit max_tokens preventing full sequence generation"""
    import llama_cpp

    model = models.llamacpp(
        repo_id="M4-ai/TinyMistral-248M-v2-Instruct-GGUF",
        filename="TinyMistral-248M-v2-Instruct.Q4_K_M.gguf",
        tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(
            "Locutusque/TinyMistral-248M-Instruct"
        ),
    )

    long_pattern = "abcdefg" * 10
    generator = generate.regex(model, long_pattern)

    output = generator("a")
    assert re.match(long_pattern, output)
