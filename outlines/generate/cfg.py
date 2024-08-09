from functools import singledispatch

from outlines.generate.api import (
    SequenceGeneratorAdapter,
    VisionSequenceGeneratorAdapter,
)
from outlines.models import ExLlamaV2Model, LlamaCpp, OpenAI, TransformersVision
from outlines.samplers import Sampler, multinomial


@singledispatch
def cfg(
    model, cfg_str: str, sampler: Sampler = multinomial()
) -> SequenceGeneratorAdapter:
    """Generate text in the language of a Context-Free Grammar

    Arguments
    ---------
    model:
        An `outlines.model` instance.
    sampler:
        The sampling algorithm to use to generate token ids from the logits
        distribution.

    Returns
    -------
    A `SequenceGeneratorAdapter` instance that generates text.

    """
    from outlines.processors import CFGLogitsProcessor

    logits_processor = CFGLogitsProcessor(cfg_str, tokenizer=model.tokenizer)
    return SequenceGeneratorAdapter(model, logits_processor, sampler)


@cfg.register(TransformersVision)
def cfg_vision(model, cfg_str: str, sampler: Sampler = multinomial()):
    from outlines.processors import CFGLogitsProcessor

    logits_processor = CFGLogitsProcessor(cfg_str, tokenizer=model.tokenizer)
    return VisionSequenceGeneratorAdapter(model, logits_processor, sampler)


@cfg.register(ExLlamaV2Model)
def cfg_exllamav2(model, cfg_str: str, sampler: Sampler = multinomial()):
    raise NotImplementedError(
        "Not yet available, track progress in https://github.com/outlines-dev/outlines/pull/1010"
    )


@cfg.register(LlamaCpp)
def cfg_llamacpp(model, cfg_str: str, sampler: Sampler = multinomial()):
    raise NotImplementedError("Not yet available due to bug in llama_cpp tokenizer")


@cfg.register(OpenAI)
def cfg_openai(model, cfg_str: str, sampler: Sampler = multinomial()):
    raise NotImplementedError(
        "Cannot use grammar-structured generation with an OpenAI model"
        + "due to the limitations of the OpenAI API."
    )
