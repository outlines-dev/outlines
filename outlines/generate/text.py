from functools import singledispatch

from outlines.fsm.guide import StopAtEOSGuide
from outlines.generate.api import SequenceGenerator, SequenceGeneratorAdapter
from outlines.models import MLXLM, VLLM, LlamaCpp, OpenAI
from outlines.samplers import Sampler, multinomial


@singledispatch
def text(
    model, sampler: Sampler = multinomial(), apply_chat_template: bool = True
) -> SequenceGenerator:
    """Generate text with a `Transformer` model.

    Note
    ----
    Python 3.11 allows dispatching on Union types and
    this should greatly simplify the code.

    Arguments
    ---------
    model:
        An instance of `Transformer` that represents a model from the
        `transformers` library.
    sampler:
        The sampling algorithm to use to generate token ids from the logits
        distribution.

    Returns
    -------
    A `SequenceGenerator` instance that generates text.

    """
    fsm = StopAtEOSGuide(model.tokenizer)
    device = model.device
    generator = SequenceGenerator(fsm, model, sampler, device, apply_chat_template)

    return generator


@text.register(MLXLM)
def text_mlxlm(
    model: MLXLM, sampler: Sampler = multinomial(), apply_chat_template: bool = True
):
    return SequenceGeneratorAdapter(model, None, sampler, apply_chat_template)


@text.register(VLLM)
def text_vllm(
    model: VLLM, sampler: Sampler = multinomial(), apply_chat_template: bool = True
):
    return SequenceGeneratorAdapter(model, None, sampler, apply_chat_template)


@text.register(LlamaCpp)
def text_llamacpp(model: LlamaCpp, sampler: Sampler = multinomial()):
    return SequenceGeneratorAdapter(model, None, sampler, apply_chat_template=False)


@text.register(OpenAI)
def text_openai(model: OpenAI, sampler: Sampler = multinomial()) -> OpenAI:
    if not isinstance(sampler, multinomial):
        raise NotImplementedError(
            r"The OpenAI API does not support any other sampling algorithm "
            + "than the multinomial sampler."
        )

    return model
