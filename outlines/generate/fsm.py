from functools import singledispatch

import interegular

from outlines.fsm.guide import RegexGuide
from outlines.generate.api import SequenceGenerator, SequenceGeneratorAdapter
from outlines.models import MLXLM, LlamaCpp, Transformers
from outlines.samplers import Sampler, multinomial


@singledispatch
def fsm(
    model, fsm: interegular.fsm.FSM, sampler: Sampler = multinomial()
) -> SequenceGenerator:
    fsm = RegexGuide.from_interegular_fsm(fsm, model.tokenizer)
    device = model.device
    generator = SequenceGenerator(fsm, model, sampler, device)
    return generator


@fsm.register(MLXLM)
@fsm.register(Transformers)
@fsm.register(LlamaCpp)
def fsm_unified(
    model, fsm: interegular.fsm.FSM, sampler: Sampler = multinomial()
) -> SequenceGeneratorAdapter:
    from outlines.processors import FSMLogitsProcessor

    fsm = RegexGuide.from_interegular_fsm(fsm, model.tokenizer)
    logits_processor = FSMLogitsProcessor(tokenizer=model.tokenizer, fsm=fsm)
    return SequenceGeneratorAdapter(model, logits_processor, sampler)
