import contextlib
import re

import pytest

import outlines.generate as generate
import outlines.models as models
import outlines.samplers as samplers


@pytest.fixture(scope="session")
def model_llamacpp(tmp_path_factory):
    return models.llamacpp(
        repo_id="M4-ai/TinyMistral-248M-v2-Instruct-GGUF",
        filename="TinyMistral-248M-v2-Instruct.Q4_K_M.gguf",
    )


@pytest.fixture(scope="session")
def model_mlxlm(tmp_path_factory):
    return models.mlxlm("mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit")


@pytest.fixture(scope="session")
def model_transformers_random(tmp_path_factory):
    return models.transformers("hf-internal-testing/tiny-random-gpt2", device="cpu")


@pytest.fixture(scope="session")
def model_transformers_opt125m(tmp_path_factory):
    return models.transformers("facebook/opt-125m", device="cpu")


ALL_MODEL_FIXTURES = (
    "model_llamacpp",
    "model_mlxlm",
    "model_transformers_random",
    "model_transformers_opt125m",
)


NOT_IMPLEMENTED = {
    "batch": ["model_llamacpp"],
    "stream": ["model_vllm"],
    "beam_search": ["model_llamacpp"],
    "multiple_samples": ["model_llamacpp"],
}


def enforce_not_implemented(model_fixture, *task_names):
    """
    Per `NOT_IMPLEMENTED`, mapping, if a model hasn't implemented a task,
    assert an NotImplementedError is raised. Otherwise, run normally
    """
    for task_name in task_names:
        if model_fixture in NOT_IMPLEMENTED.get(task_name, []):
            return pytest.raises(NotImplementedError)
    else:
        return contextlib.nullcontext()


REGEX_PATTERNS = [
    "(123456789)|(abcdefghijklmnop)",
    "abc*",
    "\\+?[1-9][0-9]{7,14}",
    r"([a-z]{10})@([a-z]{5})\.([a-z]{3})",
]


@pytest.mark.parametrize("sampler_name", ("greedy", "multinomial", "beam_search"))
@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
def test_generate_text(request, model_fixture, sampler_name):
    model = request.getfixturevalue(model_fixture)
    generator = generate.text(model, getattr(samplers, sampler_name)())
    with enforce_not_implemented(model_fixture, sampler_name):
        res = generator("test", max_tokens=10)
        assert isinstance(res, str)


@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
def test_generate_batch_text(request, model_fixture):
    model = request.getfixturevalue(model_fixture)
    generator = generate.text(model)
    with enforce_not_implemented(model_fixture, "batch"):
        res = generator(["test", "test2"], max_tokens=10)
        assert isinstance(res, list)
        assert isinstance(res[0], str)


@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
def test_generate_text_stream(request, model_fixture):
    model = request.getfixturevalue(model_fixture)
    generator = generate.text(model)
    with enforce_not_implemented(model_fixture, "stream"):
        for token in generator.stream("a b c ", max_tokens=10):
            assert isinstance(token, str)


@pytest.mark.parametrize("pattern", REGEX_PATTERNS)
@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
def test_generate_regex(request, model_fixture, pattern):
    model = request.getfixturevalue(model_fixture)
    generator = generate.regex(model, pattern)
    res = generator("foobarbaz", max_tokens=20)
    assert re.fullmatch(pattern, res) is not None, res


@pytest.mark.parametrize("pattern", REGEX_PATTERNS)
@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
def test_generate_regex_stream(request, model_fixture, pattern):
    model = request.getfixturevalue(model_fixture)
    generator = generate.regex(model, pattern)
    with enforce_not_implemented(model_fixture, "stream"):
        output = ""
        for token in generator.stream("output:", max_tokens=20):
            output += token
        assert re.fullmatch(pattern, output) is not None, output


@pytest.mark.parametrize("pattern", REGEX_PATTERNS)
@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
def test_generate_regex_batch_stream(request, model_fixture, pattern):
    model = request.getfixturevalue(model_fixture)
    generator = generate.regex(model, pattern)
    with enforce_not_implemented(model_fixture, "batch", "stream"):
        outputs = ["", ""]
        for tokens in generator.stream(["input 0", "input 1"], max_tokens=20):
            outputs[0] += tokens[0]
            outputs[1] += tokens[1]
        for output in outputs:
            assert re.fullmatch(pattern, output) is not None, output


@pytest.mark.parametrize("pattern", REGEX_PATTERNS)
@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
def test_generate_regex_batch(request, model_fixture, pattern):
    """Ensure batch requests work and fsm order is maintained"""
    model = request.getfixturevalue(model_fixture)
    generator = generate.regex(model, pattern)
    with enforce_not_implemented(model_fixture, "batch"):
        outputs = generator(["abc", "123", "123bce", "33aa"], max_tokens=20)
        for output in outputs:
            assert re.fullmatch(pattern, output) is not None, output


@pytest.mark.parametrize("pattern", REGEX_PATTERNS)
@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
def test_generate_regex_single_multinomial(request, model_fixture, pattern):
    """Ensure batch requests work and fsm order is maintained"""
    model = request.getfixturevalue(model_fixture)
    generator = generate.regex(model, pattern, sampler=samplers.multinomial(4))
    with enforce_not_implemented(model_fixture, "multiple_samples"):
        output_sample_groups = generator("single input", max_tokens=40)
        for output in output_sample_groups:
            assert re.fullmatch(pattern, output) is not None, output


@pytest.mark.parametrize("pattern", REGEX_PATTERNS)
@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
def test_generate_regex_batch_multinomial(request, model_fixture, pattern):
    """Ensure batch requests work and fsm order is maintained"""
    model = request.getfixturevalue(model_fixture)
    generator = generate.regex(model, pattern, sampler=samplers.multinomial(4))
    with enforce_not_implemented(model_fixture, "batch", "multiple_samples"):
        output_batch_groups = generator(["abc", "123", "123bce", "33aa"], max_tokens=40)
        for output_sample_groups in output_batch_groups:
            for output in output_sample_groups:
                assert re.fullmatch(pattern, output) is not None, output


@pytest.mark.parametrize("pattern", REGEX_PATTERNS)
@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
def test_generate_regex_batch_beam_search(request, model_fixture, pattern):
    """Ensure batch requests work and fsm order is maintained"""
    model = request.getfixturevalue(model_fixture)
    generator = generate.regex(model, pattern, sampler=samplers.beam_search(4))
    with enforce_not_implemented(model_fixture, "batch", "multiple_samples"):
        output_batch_groups = generator(["abc", "123", "123bce", "33aa"], max_tokens=40)
        for output_sample_groups in output_batch_groups:
            for output in output_sample_groups:
                assert re.fullmatch(pattern, output) is not None, output
