import contextlib
import re

import pytest

import outlines.generate as generate
import outlines.models as models
import outlines.samplers as samplers

##########################################
# Model Fixtures
##########################################


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
def model_mlxlm_phi3(tmp_path_factory):
    return models.mlxlm("mlx-community/Phi-3-mini-4k-instruct-4bit")


@pytest.fixture(scope="session")
def model_transformers_random(tmp_path_factory):
    return models.transformers("hf-internal-testing/tiny-random-gpt2", device="cpu")


@pytest.fixture(scope="session")
def model_transformers_opt125m(tmp_path_factory):
    return models.transformers("facebook/opt-125m", device="cpu")


@pytest.fixture(scope="session")
def model_mamba(tmp_path_factory):
    return models.mamba(model_name="state-spaces/mamba-130m-hf", device="cpu")


@pytest.fixture(scope="session")
def model_bart(tmp_path_factory):
    from transformers import AutoModelForSeq2SeqLM

    return models.transformers(
        "facebook/bart-base", device="cpu", model_class=AutoModelForSeq2SeqLM
    )


@pytest.fixture(scope="session")
def model_transformers_vision(tmp_path_factory):
    import torch
    from transformers import LlavaNextForConditionalGeneration

    return models.transformers_vision(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        model_class=LlavaNextForConditionalGeneration,
        device="cuda",
        model_kwargs=dict(
            torch_dtype=torch.bfloat16,
            load_in_4bit=True,
            low_cpu_mem_usage=True,
        ),
    )


@pytest.fixture(scope="session")
def model_vllm(tmp_path_factory):
    return models.vllm("facebook/opt-125m", gpu_memory_utilization=0.1)


# TODO: exllamav2 failing in main, address in https://github.com/outlines-dev/outlines/issues/808
# TODO: t5 tokenizer doesn't work with streaming
"""
@pytest.fixture(scope="session")
def model_exllamav2(tmp_path_factory):
    return models.exllamav2(
        model_path="blockblockblock/TinyLlama-1.1B-Chat-v1.0-bpw4-exl2",
        device="cpu"
    )

@pytest.fixture(scope="session")
def model_t5(tmp_path_factory):
    from transformers import AutoModelForSeq2SeqLM

    return models.transformers(
        "EleutherAI/pile-t5-base", device="cpu", model_class=AutoModelForSeq2SeqLM
    )
"""


ALL_MODEL_FIXTURES = (
    "model_llamacpp",
    "model_mlxlm",
    "model_mlxlm_phi3",
    "model_transformers_random",
    "model_transformers_opt125m",
    "model_mamba",
    "model_bart",
    "model_transformers_vision",
    "model_vllm",
)


##########################################
# Stuctured Generation Inputs
##########################################


@pytest.fixture()
def sample_schema():
    from pydantic import BaseModel, conint, conlist, constr

    class SampleSchema(BaseModel):
        title: constr(max_length=10)
        numbers: conlist(conint(strict=True), min_length=3, max_length=3)
        labels: conlist(constr(min_length=1, max_length=5), min_length=3, max_length=3)

    return SampleSchema


@pytest.fixture()
def sample_choices():
    return ["foo", "bar", "baz"]


REGEX_PATTERNS = [
    "a b c d e",  # ensure proper tokenizer whitespace prefix handling
    "(123456789)|(abcdefghijklmnop)",  # ensure consistent correct sequence handling during batch
    r"([a-z]{10})@([a-z]{5})\.([a-z]{3})",  # email example
]


###########################################
# Model/Generator Pair Behavior Definitions
###########################################


def enforce_not_implemented(model_fixture, *task_names):
    """
    Per `NOT_IMPLEMENTED`, mapping, if a model hasn't implemented a task,
    assert an NotImplementedError is raised. Otherwise, run normally
    """
    NOT_IMPLEMENTED = {
        "stream": ["model_transformers_vision", "model_vllm"],
        "batch": ["model_llamacpp", "model_mlxlm", "model_mlxlm_phi3"],
        "beam_search": ["model_llamacpp", "model_mlxlm", "model_mlxlm_phi3"],
        "multiple_samples": ["model_llamacpp", "model_mlxlm", "model_mlxlm_phi3"],
    }
    for task_name in task_names:
        if model_fixture in NOT_IMPLEMENTED.get(task_name, []):
            return pytest.raises(NotImplementedError)
    else:
        return contextlib.nullcontext()


def get_inputs(fixture_name, batch_size=None):
    """Get generator kwargs, just the prompt by default, but include images for transformers_visian"""
    from io import BytesIO
    from urllib.request import urlopen

    from PIL import Image  # type: ignore

    prompts = ["abcd", "efgh", "1234", "5678", "foo", "bar", "baz", "bif"]
    prompts = prompts[0] if batch_size is None else prompts[:batch_size]

    if fixture_name.endswith("_vision"):
        img_url = "https://python-pillow.org/pillow-perf/static/space_pil_lanczos.png"
        img = Image.open(BytesIO(urlopen(img_url).read())).convert("RGB")

        if batch_size is None:
            return {"prompts": f"<image> {prompts}", "media": [img]}
        else:
            return {
                "prompts": [f"<image> {p}" for p in prompts],
                "media": [[img] for _ in range(batch_size)],
            }

    else:
        return {"prompts": prompts}


###########################################
# Tests
###########################################


@pytest.mark.parametrize("sampler_name", ("greedy", "multinomial", "beam_search"))
@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
def test_generate_text(request, model_fixture, sampler_name):
    model = request.getfixturevalue(model_fixture)
    generator = generate.text(model, getattr(samplers, sampler_name)())
    with enforce_not_implemented(model_fixture, sampler_name):
        res = generator(**get_inputs(model_fixture), max_tokens=10)
        assert isinstance(res, str)


@pytest.mark.parametrize("pattern", REGEX_PATTERNS)
@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
def test_generate_regex(request, model_fixture, pattern):
    model = request.getfixturevalue(model_fixture)
    generator = generate.regex(model, pattern)
    res = generator(**get_inputs(model_fixture), max_tokens=20)
    assert re.fullmatch(pattern, res) is not None, res


@pytest.mark.parametrize("pattern", REGEX_PATTERNS)
@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
def test_generate_fsm(request, model_fixture, pattern):
    import interegular

    model = request.getfixturevalue(model_fixture)
    generator = generate.fsm(model, interegular.parse_pattern(pattern).to_fsm())
    res = generator(**get_inputs(model_fixture))
    assert re.fullmatch(pattern, res) is not None, res


@pytest.mark.skip(
    "Fix issues with JSON, some models fail this test https://github.com/outlines-dev/outlines/issues/985"
)
@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
def test_generate_json(request, model_fixture, sample_schema):
    model = request.getfixturevalue(model_fixture)
    generator = generate.json(model, sample_schema)
    # asserts valid within call
    generator(**get_inputs(model_fixture), max_tokens=100)


@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
def test_generate_choice(request, model_fixture, sample_choices):
    model = request.getfixturevalue(model_fixture)
    generator = generate.choice(model, sample_choices)
    res = generator(**get_inputs(model_fixture))
    assert res in sample_choices


@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
def test_generate_format_bool(request, model_fixture):
    model = request.getfixturevalue(model_fixture)
    generator = generate.format(model, bool)
    res = generator(**get_inputs(model_fixture))
    assert isinstance(res, bool)


@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
def test_generate_text_stream(request, model_fixture):
    model = request.getfixturevalue(model_fixture)
    generator = generate.text(model)
    with enforce_not_implemented(model_fixture, "stream"):
        for token in generator.stream(**get_inputs(model_fixture), max_tokens=10):
            assert isinstance(token, str)


@pytest.mark.parametrize("pattern", REGEX_PATTERNS)
@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
def test_generate_regex_stream(request, model_fixture, pattern):
    model = request.getfixturevalue(model_fixture)
    generator = generate.regex(model, pattern)
    with enforce_not_implemented(model_fixture, "stream"):
        output = ""
        for token in generator.stream(**get_inputs(model_fixture), max_tokens=20):
            output += token
        assert re.fullmatch(pattern, output) is not None, output


@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
def test_generate_batch_text(request, model_fixture):
    model = request.getfixturevalue(model_fixture)
    generator = generate.text(model)
    with enforce_not_implemented(model_fixture, "batch"):
        res = generator(**get_inputs(model_fixture, 2), max_tokens=10)
        assert isinstance(res, list)
        assert isinstance(res[0], str)


@pytest.mark.parametrize("pattern", REGEX_PATTERNS)
@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
def test_generate_regex_batch(request, model_fixture, pattern):
    """Ensure batch requests work and fsm order is maintained"""
    model = request.getfixturevalue(model_fixture)
    generator = generate.regex(model, pattern)
    with enforce_not_implemented(model_fixture, "batch"):
        outputs = generator(**get_inputs(model_fixture, 4), max_tokens=20)
        for output in outputs:
            assert re.fullmatch(pattern, output) is not None, output


@pytest.mark.parametrize("pattern", REGEX_PATTERNS)
@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
def test_generate_regex_batch_stream(request, model_fixture, pattern):
    model = request.getfixturevalue(model_fixture)
    generator = generate.regex(model, pattern)
    with enforce_not_implemented(model_fixture, "batch", "stream"):
        outputs = ["", ""]
        for tokens in generator.stream(**get_inputs(model_fixture, 2), max_tokens=20):
            outputs[0] += tokens[0]
            outputs[1] += tokens[1]
        for output in outputs:
            assert re.fullmatch(pattern, output) is not None, output


@pytest.mark.parametrize("pattern", REGEX_PATTERNS)
@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
def test_generate_regex_single_multinomial(request, model_fixture, pattern):
    """Ensure batch requests work and fsm order is maintained"""
    model = request.getfixturevalue(model_fixture)
    generator = generate.regex(model, pattern, sampler=samplers.multinomial(4))
    with enforce_not_implemented(model_fixture, "multiple_samples"):
        output_sample_groups = generator(**get_inputs(model_fixture), max_tokens=40)
        for output in output_sample_groups:
            assert re.fullmatch(pattern, output) is not None, output


@pytest.mark.parametrize("pattern", REGEX_PATTERNS)
@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
@pytest.mark.parametrize("sampler_name", ("multinomial", "beam_search"))
def test_generate_regex_batch_multi_sample(
    request, model_fixture, pattern, sampler_name
):
    """Ensure batch requests work and fsm order is maintained"""
    model = request.getfixturevalue(model_fixture)
    generator = generate.regex(
        model, pattern, sampler=getattr(samplers, sampler_name)(4)
    )
    with enforce_not_implemented(model_fixture, "batch", "multiple_samples"):
        output_batch_groups = generator(**get_inputs(model_fixture, 4), max_tokens=40)
        for output_sample_groups in output_batch_groups:
            for output in output_sample_groups:
                assert re.fullmatch(pattern, output) is not None, output
