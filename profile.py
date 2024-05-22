import interegular
import line_profiler
from transformers import SPIECE_UNDERLINE

from outlines import generate, models
from outlines.fsm.regex import (
    _walk_fsm,
    create_fsm_index_end_to_end,
    create_fsm_index_tokenizer,
    make_byte_level_fsm,
    make_deterministic_fsm,
    reduced_vocabulary,
    state_scan_tokens,
)


def run_model():
    model = models.vllm(
        "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ", quantization="gptq", dtype="half"
    )
    tokenizer = model.model.get_tokenizer()
    tokenizer.vocabulary = tokenizer.get_vocab()
    tokenizer.special_tokens = set(tokenizer.all_special_tokens)

    def convert_token_to_string(token):
        string = tokenizer.convert_tokens_to_string([token])

        # A hack to handle missing spaces to HF's Llama tokenizers
        if (
            type(token) is str
            and token.startswith(SPIECE_UNDERLINE)
            or token == "<0x20>"
        ):
            return " " + string

        return string

    tokenizer.convert_token_to_string = convert_token_to_string

    regex_string = '\\{[\n ]*"name"[\n ]*:[\n ]*"(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.){,10}"[\n ]*,[\n ]*"age"[\n ]*:[\n ]*(0|[1-9][0-9]*)[\n ]*,[\n ]*"armor"[\n ]*:[\n ]*("leather"|"chainmail"|"plate")[\n ]*,[\n ]*"strength"[\n ]*:[\n ]*(0|[1-9][0-9]*)[\n ]*\\}'
    regex_pattern = interegular.parse_pattern(regex_string)
    byte_fsm = make_byte_level_fsm(regex_pattern.to_fsm().reduce(), keep_utf8=True)
    regex_fsm, _ = make_deterministic_fsm(byte_fsm)
    states_to_token_maps, empty_token_ids = create_fsm_index_tokenizer(
        regex_fsm, tokenizer
    )


profile = line_profiler.LineProfiler()
profile.add_function(create_fsm_index_tokenizer)
profile.add_function(create_fsm_index_end_to_end)
profile.add_function(reduced_vocabulary)
profile(run_model)()
profile.print_stats()
