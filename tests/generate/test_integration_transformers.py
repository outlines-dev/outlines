import datetime
import re
from enum import Enum
from typing import List, Union

import pytest
import torch
from pydantic import BaseModel, constr

import outlines.generate as generate
import outlines.models as models
from outlines.fsm.regex import reduced_vocabulary
from outlines.models.transformers import Transformers, TransformerTokenizer
from outlines.samplers import beam_search, greedy, multinomial


@pytest.fixture(scope="session")
def model(tmp_path_factory):
    return models.transformers(
        "hf-internal-testing/tiny-random-GPTJForCausalLM", device="cpu"
    )


def test_transformers_integration_text(model):
    sequence = generate.text(model)(
        "Write a short sentence ", seed=10000, max_tokens=10
    )
    assert isinstance(sequence, str)
    assert model.tokenizer.eos_token not in sequence

    sequence = generate.text(model)(
        "Write a short sentence ",
        max_tokens=20,
        stop_at="a",
        seed=10000,
    )
    assert isinstance(sequence, str)

    prompts = ["Write a short sentence ", "And another one "]
    sequence = generate.text(model)(
        prompts, max_tokens=10, stop_at=[".", ","], seed=10000
    )
    assert isinstance(sequence, list)
    assert len(sequence) == 2
    assert isinstance(sequence[0], str)


def test_transformers_integration_text_multiple_samples(model):
    sampler = multinomial(2)

    sequence = generate.text(model, sampler=sampler)(
        "Write a short sentence ", seed=10000
    )
    assert isinstance(sequence, list)
    assert len(sequence) == 2
    assert model.tokenizer.eos_token not in sequence

    prompts = ["Write a short sentence ", "And another one "]
    sequence = generate.text(model, sampler=sampler)(
        prompts, max_tokens=10, stop_at=[".", ","], seed=10000
    )
    assert isinstance(sequence, list)
    assert len(sequence) == 2
    assert isinstance(sequence[0], list)
    assert len(sequence) == 2
    assert isinstance(sequence[0][0], str)


def test_transformers_integration_streaming(model):
    sequence = generate.text(model).stream(
        "Write a short sentence ", max_tokens=10, stop_at=[".", ","], seed=10000
    )

    token = next(sequence)
    assert isinstance(token, str)

    remaining = "".join([token for token in sequence])
    assert isinstance(remaining, str)

    sequence = generate.text(model).stream(
        ["Prompt1", "Prompt2"], max_tokens=10, stop_at=[".", ","], seed=10000
    )
    tokens = next(sequence)
    assert isinstance(tokens, list)
    assert isinstance(tokens[0], str)
    assert isinstance(tokens[1], str)


def test_transformers_integration_streaming_batch_samples(model):
    sampler = multinomial(samples=2)

    sequence = generate.text(model, sampler=sampler).stream(
        ["Prompt1", "Prompt2"], max_tokens=10, stop_at=[".", ","], seed=10000
    )
    tokens = next(sequence)
    assert isinstance(tokens, list)
    assert len(tokens) == 2
    assert isinstance(tokens[0], list)
    assert len(tokens[0]) == 2
    assert isinstance(tokens[0], list)
    assert len(tokens[1]) == 2


def test_transformers_integration_streaming_batch_beam_search(model):
    sampler = beam_search(beams=2)

    sequence = generate.regex(model, r"ab[cd]e", sampler=sampler).stream(
        ["Prompt1", "Prompt2"], max_tokens=10, stop_at=["c", "d"], seed=10000
    )
    tokens = next(sequence)
    assert isinstance(tokens, list)
    assert len(tokens) == 2
    assert isinstance(tokens[0], list)
    assert len(tokens[0]) == 2
    assert isinstance(tokens[0], list)
    assert len(tokens[1]) == 2


def test_transformers_integration_text_stop(model):
    prompt = "Write a short sentence "
    sequence = generate.text(model)(prompt, stop_at="a", seed=10000)
    assert sequence[len(prompt) :].find("a") == -1


def test_transformers_various_regexes(model):
    prompt = "Write an email address"
    regex_str = r"([a-z]{10})@([a-z]{5})\.([a-z]{3})"
    generator = generate.regex(model, regex_str)

    # One prompt
    sequence = generator(prompt, seed=0)
    assert re.fullmatch(regex_str, sequence) is not None


def test_transformers_various_regexes_prompt_list(model):
    prompt = "Write an email address"
    regex_str = r"([a-z]{10})@([a-z]{5})\.([a-z]{3})"
    generator = generate.regex(model, regex_str)

    # Two prompts
    sequence = generator([prompt, prompt], seed=0)
    assert re.fullmatch(regex_str, sequence[0]) is not None
    assert re.fullmatch(regex_str, sequence[1]) is not None


def test_transformers_various_regexes_prompt_list_multiple_samples(model):
    sampler = multinomial(samples=2)
    prompt = "Write an email address"
    regex_str = r"([a-z]{10})@([a-z]{5})\.([a-z]{3})"
    generator = generate.regex(model, regex_str, sampler=sampler)

    # Two prompts
    sequence = generator([prompt, prompt], seed=0, max_tokens=500)
    assert isinstance(sequence, list)
    assert len(sequence) == 2
    assert re.fullmatch(regex_str, sequence[0][0]) is not None
    assert re.fullmatch(regex_str, sequence[0][1]) is not None
    assert re.fullmatch(regex_str, sequence[1][0]) is not None
    assert re.fullmatch(regex_str, sequence[1][1]) is not None


def test_transformers_various_regexes_prompt_list_beam_search(model):
    sampler = beam_search(5)
    prompt_1 = "Write an email address"
    prompt_2 = "Random"
    regex_str = r"([a-z]{10})@([a-z]{5})\.([a-z]{3})"
    generator = generate.regex(model, regex_str, sampler=sampler)

    # Two prompts
    sequence = generator([prompt_1, prompt_2], seed=0)
    assert isinstance(sequence, list)
    assert len(sequence) == 2
    assert len(sequence[0]) == 5
    assert re.fullmatch(regex_str, sequence[0][0]) is not None
    assert re.fullmatch(regex_str, sequence[0][1]) is not None
    assert re.fullmatch(regex_str, sequence[1][0]) is not None
    assert re.fullmatch(regex_str, sequence[1][1]) is not None


def test_transformers_integration_integer(model):
    prompt = "Write a short sentence"
    sequence = generate.format(model, int)(prompt, max_tokens=10, seed=0)

    assert isinstance(sequence, int)


def test_transformers_integration_integer_array(model):
    prompts = ["Give me a number", "And another one"]
    sequence = generate.format(model, int)(prompts, max_tokens=10, seed=0)
    assert isinstance(sequence, list)
    assert len(sequence) == 2
    assert isinstance(sequence[0], int)
    assert isinstance(sequence[1], int)


def test_transformers_integration_float(model):
    prompt = "Write a short sentence"
    sequence = generate.format(model, float)(prompt, max_tokens=10, seed=0)

    assert sequence != ""
    assert isinstance(sequence, float)


def test_transformers_integration_bool(model):
    prompt = "Is this True or False?"
    sequence = generate.format(model, bool)(prompt, max_tokens=10, seed=0)

    assert sequence != ""
    assert isinstance(sequence, bool)


def test_transformers_integration_date(model):
    prompt = "What day is it today?"
    sequence = generate.format(model, datetime.date)(prompt, max_tokens=10, seed=0)

    assert sequence != ""
    assert isinstance(sequence, datetime.date)


def test_transformers_integration_time(model):
    prompt = "What time is it?"
    sequence = generate.format(model, datetime.time)(prompt, max_tokens=10, seed=0)

    assert sequence != ""
    assert isinstance(sequence, datetime.time)


def test_transformers_integration_datetime(model):
    prompt = "What time is it?"
    sequence = generate.format(model, datetime.datetime)(prompt, max_tokens=20, seed=0)

    assert sequence != 0
    assert isinstance(sequence, datetime.datetime)


def test_transformers_integration_choice(model):
    prompt = "Write a short sentence "
    sequence = generate.choice(model, ["test", "choice"])(prompt, seed=0)

    assert sequence == "test" or sequence == "choice"


def test_transformers_integration_with_pad_token():
    model_name = "hf-internal-testing/tiny-random-XLMRobertaXLForCausalLM"
    model = models.transformers(model_name, device="meta")
    assert model.tokenizer.pad_token_id == 1
    assert model.tokenizer.pad_token == "<pad>"


def test_transformers_json_basic(model):
    prompt = "Output some JSON "

    class Spam(BaseModel):
        foo: int
        bar: float
        spam: constr(max_length=10)
        fuzz: bool

    result = generate.json(model, Spam)(prompt, max_tokens=500, seed=0)
    assert isinstance(result, BaseModel)
    assert isinstance(result.foo, int)
    assert isinstance(result.bar, float)
    assert isinstance(result.spam, str)
    assert isinstance(result.fuzz, bool)
    assert len(result.spam) <= 10


def test_transformers_json_schema(model):
    prompt = "Output some JSON "

    schema = """{
      "title": "spam",
      "type": "object",
      "properties": {
           "foo" : {"type": "integer"},
           "bar": {"type": "string", "maxLength": 4}
        },
      "required": ["foo", "bar"]
      }
    """

    result = generate.json(model, schema)(prompt, max_tokens=500, seed=0)
    assert isinstance(result, dict)
    assert isinstance(result["foo"], int)
    assert isinstance(result["bar"], str)


def test_transformers_json_batch(model):
    prompts = ["Output some JSON ", "Output more JSON"]

    class Spam(BaseModel):
        foo: int
        bar: float
        spam: constr(max_length=10)
        fuzz: bool

    result = generate.json(model, Spam)(prompts, max_tokens=500, seed=0)
    assert isinstance(result[0], BaseModel)
    assert isinstance(result[1], BaseModel)


def test_transformers_json_batch_multiple_samples(model):
    sampler = multinomial(samples=2)
    prompts = ["Output some JSON ", "Output more JSON"]

    class Spam(BaseModel):
        foo: int
        bar: float
        spam: constr(max_length=10)
        fuzz: bool

    result = generate.json(model, Spam, sampler=sampler)(
        prompts, max_tokens=500, seed=0
    )
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0][0], BaseModel)
    assert isinstance(result[0][1], BaseModel)
    assert isinstance(result[1][0], BaseModel)
    assert isinstance(result[1][1], BaseModel)


def test_transformers_json_str_enum(model):
    prompt = "Output some JSON "

    class Name(str, Enum):
        john = "John"
        marc = "Marc"
        michel = "Michel"

    class User(BaseModel):
        user_id: int
        name: Name

    result = generate.json(model, User)(prompt, seed=0)
    assert isinstance(result, BaseModel)
    assert isinstance(result.user_id, int)
    assert result.name in ["John", "Marc", "Michel"]


def test_transformers_json_int_enum(model):
    prompt = "Output some JSON "

    class Id(int, Enum):
        one = 1
        two = 2

    class User(BaseModel):
        user_id: Id

    result = generate.json(model, User)(prompt, seed=0)
    assert isinstance(result, BaseModel)
    assert isinstance(result.user_id, int)
    assert result.user_id in [1, 2]


def test_transformers_json_array(model):
    prompt = "Output some JSON "

    class User(BaseModel):
        user_id: int
        value: List[float]

    result = generate.json(model, User)(prompt, seed=0)
    assert isinstance(result, BaseModel)
    assert isinstance(result.user_id, int)
    assert isinstance(result.value, list)
    for value in result.value:
        assert isinstance(value, float) or isinstance(value, int)


@pytest.mark.xfail(reason="The implementation of `anyOf` is incorrect")
def test_transformers_json_union(model):
    prompt = "Output some JSON "

    class Spam(BaseModel):
        foo: int
        bar: Union[constr(max_length=10), float]

    result = generate.json(model, Spam)(prompt, max_tokens=100, seed=4)
    assert isinstance(result, BaseModel)
    assert (
        isinstance(result.bar, int)
        or isinstance(result.bar, float)
        or isinstance(result.bar, str)
    )


def test_transformers_json_function(model):
    prompt = "Output arguments for the function"

    def function(foo: int, bar: List[int]):
        return foo + sum(bar)

    sequence = generate.json(model, function)(prompt, max_tokens=100, seed=4)
    assert isinstance(sequence, dict)
    assert isinstance(function(**sequence), int)


def test_transformers_logits_vocab_size(model):
    # Artificially increase the weights/logits size relative
    # to the vocabulary
    model.model.resize_token_embeddings(pad_to_multiple_of=3)

    assert len(model.tokenizer.vocabulary) == 1024
    assert model.model.base_model.wte.weight.shape[0] == 1026

    generator = generate.choice(model, ["True", "False"])

    sequence = generator("blah", seed=101)
    assert sequence == "False"


def test_transformers_json_custom_ws(model):
    prompt = "Output some JSON with newlines"  # try to force model to use newlines

    schema = """{
      "title": "spam",
      "type": "object",
      "properties": {
           "foo" : {"type": "integer"},
           "bar": {"type": "integer"}
        },
      "required": ["foo", "bar"]
      }
    """

    generator = generate.json(model, schema, whitespace_pattern=r"[ ]?")
    generator.format_sequence = lambda x: x  # patch to return raw text
    assert "\n" not in generator(prompt, max_tokens=500, seed=0)


def test_transformers_reduced_vocabulary_caching():
    from transformers import AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer = TransformerTokenizer(hf_tokenizer)
    tokenizer2 = TransformerTokenizer(hf_tokenizer)

    # TODO: We might actually want only one copy of a given tokenizer.
    assert tokenizer is not tokenizer2

    vocab = reduced_vocabulary(tokenizer)
    vocab2 = reduced_vocabulary(tokenizer2)

    assert vocab2 is vocab


@pytest.mark.skip(reason="Custom Sampler Disabled in Transformers Integration")
def test_custom_sampler(model):
    seen = False
    target_token_ids = model.tokenizer.encode(["c"])[0]

    class biased_sampler:
        def __init__(self, samples: int = 1):
            self.samples = samples

        def __call__(
            logits: torch.DoubleTensor, samples: int, *_
        ) -> torch.DoubleTensor:
            nonlocal seen

            if not seen:
                seen = True
                return target_token_ids, torch.tensor([0]), None
            else:
                return (
                    torch.tensor([[model.tokenizer.eos_token_id]]),
                    torch.tensor([0]),
                    None,
                )

    generator = generate.choice(model, ["a", "b", "c"], sampler=biased_sampler(1))
    sequence = generator(
        """What is 1+1?
    a. 3
    b. 4
    c. 2"""
    )

    assert sequence == "c"


def test_transformers_use_existing_model_and_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    model = Transformers(hf_model, hf_tokenizer)
    sequence = generate.text(model)("Write a short sentence ", seed=10000)
    assert isinstance(sequence, str)


def test_RegexGuide_caching(temp_cache_dir):
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

    model = models.transformers(
        "hf-internal-testing/tiny-random-XLMRobertaXLForCausalLM", device="cpu"
    )
    generator = generate.regex(model, regex, sampler=greedy())
    assert cache.stats() == (0, 1)

    model_2 = models.transformers(
        "hf-internal-testing/tiny-random-GPTJForCausalLM", device="cpu"
    )
    generator_2 = generate.regex(model_2, regex, sampler=greedy())
    assert cache.stats() == (0, 2)

    # These two different models and tokenizers should not have the same state
    # mapping results
    assert (
        generator.logits_processor.fsm.states_to_token_maps
        != generator_2.logits_processor.fsm.states_to_token_maps
    )

    generator_3 = generate.regex(model_2, regex, sampler=greedy())
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
