import pytest
from transformers import AutoTokenizer

import outlines
from outlines.fsm.fsm import RegexFSM
from outlines.fsm.guide import RegexGuide  # noqa: E402
from outlines.models.transformers import TransformerTokenizer

outlines.disable_cache()


regex_samples = {
    "email": r"[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?",
    "complex_phone": "\\+?\\d{1,4}?[-.\\s]?\\(?\\d{1,3}?\\)?[-.\\s]?\\d{1,4}[-.\\s]?\\d{1,4}[-.\\s]?\\d{1,9}",
    "simple_phone": "\\+?[1-9][0-9]{7,14}",
    "date": r"([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])(\.|-|/)([1-9]|0[1-9]|1[0-2])(\.|-|/)([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])|([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])(\.|-|/)([1-9]|0[1-9]|1[0-2])(\.|-|/)([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])",
    "time": r"(0?[1-9]|1[0-2]):[0-5]\d\s?(am|pm)?",
    "ip": r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)",
    "url": r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?",
    "ssn": r"\d{3}-\d{2}-\d{4}",
    "complex_span_constrained_relation_extraction": "(['\"\\ ,]?((?:of|resulting|case|which|cultures|a|core|extreme|selflessness|spiritual|various|However|both|vary|in|other|secular|the|religious|among|moral|and|It|object|worldviews|altruism|traditional|material|aspect|or|life|beings|virtue|is|however|opposite|concern|an|practice|it|for|s|quality|religions|In|Altruism|animals|happiness|many|become|principle|human|selfishness|may|synonym)['\"\\ ,]?)+['\"\\ ,]?\\s\\|\\s([^|\\(\\)\n]{1,})\\s\\|\\s['\"\\ ,]?((?:of|resulting|case|which|cultures|a|core|extreme|selflessness|spiritual|various|However|both|vary|in|other|secular|the|religious|among|moral|and|It|object|worldviews|altruism|traditional|material|aspect|or|life|beings|virtue|is|however|opposite|concern|an|practice|it|for|s|quality|religions|In|Altruism|animals|happiness|many|become|principle|human|selfishness|may|synonym)['\"\\ ,]?)+['\"\\ ,]?(\\s\\|\\s\\(([^|\\(\\)\n]{1,})\\s\\|\\s([^|\\(\\)\n]{1,})\\))*\\n)*",
    "high_ram_consumption": r"A: [\w \.\*\-=\+,\?/]{50,85}\. The answer is [1-9][0-9]{0,9}\.\n",
}


@pytest.mark.parametrize("regex_name", regex_samples.keys())
def test_benchmark_regex_to_fsm(
    benchmark, tokenizer, ensure_numba_compiled, regex_name
):
    """Benchmark converting regex to FSM"""
    regex_str = regex_samples[regex_name]
    benchmark.pedantic(
        RegexGuide,
        args=(regex_str, tokenizer),
        rounds=8,
    )


def setup_large_fsm():
    regex_str = r"A: [\w \.\*\-=\+,\?/]{50,65}\. The answer is [1-9][0-9]{0,9}\.\n"
    tokenizer_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer = TransformerTokenizer(tokenizer)
    fsm = RegexFSM(regex_str, tokenizer)
    return (fsm,), {}


def test_benchmark_regex_fsm_token_ids(benchmark, tokenizer, ensure_numba_compiled):
    def dict_run_through(fsm):
        for start_state in fsm.states_to_token_maps.keys():
            _ = fsm.allowed_token_ids(start_state)

    benchmark.pedantic(dict_run_through, rounds=8, setup=setup_large_fsm)


def test_benchmark_regex_fsm_states(benchmark, tokenizer, ensure_numba_compiled):
    def dict_run_through(fsm):
        for start_state in fsm.states_to_token_maps.keys():
            for token in fsm.allowed_token_ids(start_state):
                # Access every end state individually, to get an amortized
                # cost of dictionary accesses.
                next_state = fsm.next_state(start_state, token)
                assert next_state is not None

    benchmark.pedantic(dict_run_through, rounds=8, setup=setup_large_fsm)
