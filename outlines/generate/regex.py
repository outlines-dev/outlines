from functools import singledispatch

from outlines.fsm.guide import RegexGuide
from outlines.generate.api import SequenceGenerator, SequenceGeneratorAdapter
from outlines.models import OpenAI
from outlines.models.llamacpp import LlamaCpp
from outlines.models.mlxlm import MLXLM
from outlines.models.vllm import VLLM
from outlines.samplers import Sampler, multinomial


@singledispatch
def regex(
    model,
    regex_str: str,
    sampler: Sampler = multinomial(),
    apply_chat_template: bool = True,
):
    """Generate structured text in the language of a regular expression.

    Parameters
    ----------
    model:
        An instance of `Transformer` that represents a model from the
        `transformers` library.
    regex_str:
        The regular expression that the output must follow.
    sampler:
        The sampling algorithm to use to generate token ids from the logits
        distribution.

    Returns
    -------
    A `SequenceGenerator` instance that generates text constrained by the
    regular expression.

    """
    fsm = RegexGuide(regex_str, model.tokenizer)

    device = model.device
    generator = SequenceGenerator(fsm, model, sampler, device, apply_chat_template)

    return generator


@regex.register(MLXLM)
def regex_mlxlm(
    model: MLXLM,
    regex_str: str,
    sampler: Sampler = multinomial(),
    apply_chat_template: bool = True,
):
    from outlines.processors import RegexLogitsProcessor

    logits_processor = RegexLogitsProcessor(regex_str, tokenizer=model.tokenizer)
    return SequenceGeneratorAdapter(
        model, logits_processor, sampler, apply_chat_template
    )


@regex.register(LlamaCpp)
def regex_llamacpp(
    model: LlamaCpp,
    regex_str: str,
    sampler: Sampler = multinomial(),
):
    from outlines.integrations.llamacpp import RegexLogitsProcessor

    logits_processor = RegexLogitsProcessor(regex_str, llm=model.model)
    return SequenceGeneratorAdapter(
        model, logits_processor, sampler, apply_chat_template=False
    )


@regex.register(VLLM)
def regex_vllm(
    model: VLLM,
    regex_str: str,
    sampler: Sampler = multinomial(),
    apply_chat_template: bool = True,
):
    from outlines.integrations.vllm import RegexLogitsProcessor

    logits_processor = RegexLogitsProcessor(regex_str, model.model)
    return SequenceGeneratorAdapter(
        model, logits_processor, sampler, apply_chat_template
    )


@regex.register(OpenAI)
def regex_openai(
    model: OpenAI,
    regex_str: str,
    sampler: Sampler = multinomial(),
):
    raise NotImplementedError(
        "Cannot use regex-structured generation with an OpenAI model"
        + "due to the limitations of the OpenAI API."
    )
