import random

from transformers import AutoTokenizer

import outlines.grammars
from outlines.caching import cache_disabled
from outlines.fsm.guide import CFGGuide
from outlines.models.transformers import TransformerTokenizer

from .common import ensure_numba_compiled

random.seed(42)


def get_tiny_tokenizer():
    """1000 tokens in vocabulary"""
    return TransformerTokenizer(
        AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    )


benched_grammars = {
    "json": outlines.grammars.json,
    "arithmetic": outlines.grammars.arithmetic,
}


class CFGGuideBenchmark:
    params = benched_grammars.keys()

    def setup(self, grammar_name):
        self.tokenizer = get_tiny_tokenizer()
        ensure_numba_compiled(
            self.tokenizer
        )  # numba not currently used, but will be in the future
        self.prebuilt_cfg_guide = CFGGuide(
            benched_grammars[grammar_name], self.tokenizer
        )

    @staticmethod
    def _run_random_cfg(guide):
        state = guide.initial_state
        token_ids = list(guide.tokenizer.vocabulary.values())
        for i in range(40):
            # simulate ordering of logits top prob to lowest prob
            random.shuffle(token_ids)
            # simulate sampling and state update
            next_token_id = next(guide.iter_valid_token_ids(state, token_ids))
            state = guide.get_next_state(state, next_token_id)

    @cache_disabled()
    def time_cfg_guide_setup(self, grammar_name):
        CFGGuide(benched_grammars[grammar_name], self.tokenizer)

    @cache_disabled()
    def time_cfg_guide_run(self, grammar):
        self._run_random_cfg(self.prebuilt_cfg_guide)

    @cache_disabled()
    def peakmem_cfg_guide_run(self, grammar):
        self._run_random_cfg(self.prebuilt_cfg_guide)
