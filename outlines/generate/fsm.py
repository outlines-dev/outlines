from functools import singledispatch

import interegular

from outlines.fsm.guide import RegexGuide
from outlines.generate.api import (
    SequenceGenerator,
    SequenceGeneratorAdapter,
    VisionSequenceGeneratorAdapter,
)
from outlines.models import ExLlamaV2Model, TransformersVision
from outlines.samplers import Sampler, multinomial


@singledispatch
def fsm(
    model, fsm: interegular.fsm.FSM, sampler: Sampler = multinomial()
) -> SequenceGeneratorAdapter:
    from outlines.processors import FSMLogitsProcessor

    fsm = RegexGuide.from_interegular_fsm(fsm, model.tokenizer)
    logits_processor = FSMLogitsProcessor(tokenizer=model.tokenizer, fsm=fsm)
    return SequenceGeneratorAdapter(model, logits_processor, sampler)


@fsm.register(TransformersVision)
def fsm_vision(model, fsm: interegular.fsm.FSM, sampler: Sampler = multinomial()):
    from outlines.processors import FSMLogitsProcessor

    fsm = RegexGuide.from_interegular_fsm(fsm, model.tokenizer)
    logits_processor = FSMLogitsProcessor(tokenizer=model.tokenizer, fsm=fsm)
    return VisionSequenceGeneratorAdapter(model, logits_processor, sampler)
