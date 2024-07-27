from collections import namedtuple
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from outlines import grammars, models
from outlines.fsm.guide import CFGGuide


@pytest.fixture
def cleanup_lark_import():
    import importlib

    import lark.lark

    yield
    # Clean up lark.lark.LarkOptions._defaults
    importlib.reload(lark.lark)


TestInputs = namedtuple(
    "TestInputs",
    [
        "grammar",  # the lark grammar to validate against
        "vocabulary",  # the token strings which can be concatenated for a generation
        "generated",  # the tokens which have been generated so far
        "legal_next_tokens",  # the subset of the vocabulary which can legally be next in `generated`
    ],
)


cfg_test_inputs = {
    "Next Token Doesn't Complete Terminal": TestInputs(
        grammar=r'?start: "a" "bc"',
        vocabulary=["a", "ab", "b", "c"],
        generated=["a"],
        legal_next_tokens=["b"],
    ),
    "Ambiguous Terminal Completion": TestInputs(
        grammar=r'?start: "ab" | "abc"',
        vocabulary=["a", "ab", "abc", "abcd", "b", "c"],
        generated=["a"],
        legal_next_tokens=["b"],
    ),
    "Token is Substring of Another Token": TestInputs(
        grammar=r'?start: "abc" | "abcd"',
        vocabulary=["a", "b", "bc", "bcd", "bcde"],
        generated=["a"],
        legal_next_tokens=["b", "bc", "bcd"],
    ),
    "Multiple Valid Continuations": TestInputs(
        grammar=r'?start: ("a" "b") | ("a" "c")',
        vocabulary=["a", "b", "bc", "c"],
        generated=["a"],
        legal_next_tokens=["b", "c"],
    ),
    "Prefix Matches Multiple Terminals": TestInputs(
        grammar=r'?start: "abcd" | "abef"',
        vocabulary=["a", "b", "be", "bcd", "bef", "bed"],
        generated=["a"],
        legal_next_tokens=["b", "be", "bcd", "bef"],
    ),
    "Token Matches Multiple Paths in Grammar": TestInputs(
        grammar=r'?start: ("a" "b" "c") | ("a" "b" "d")',
        vocabulary=["a", "b", "c", "d"],
        generated=["a", "b"],
        legal_next_tokens=["c", "d"],
    ),
    "Incomplete Terminal at End of Prefix": TestInputs(
        grammar=r'?start: "abc"',
        vocabulary=["a", "ab", "c", "abc", "abcd"],
        generated=["ab"],
        legal_next_tokens=["c"],
    ),
    "Complex Grammar Rules": TestInputs(
        grammar=r'?start: "a" "b" ["c"]',
        vocabulary=["a", "b", "c"],
        generated=["a", "b"],
        legal_next_tokens=["c", None],  # Allowing the document to end after "a" "b"
    ),
    "Empty Prefix String": TestInputs(
        grammar=r'?start: "a" | "b"',
        vocabulary=["a", "b", "c", "d"],
        generated=[],
        legal_next_tokens=["a", "b"],
    ),
    "Ambiguous Pattern Completion": TestInputs(
        grammar=r'?start: /a+/ "b" /c?d/',
        vocabulary=["a", "aa", "b", "cd", "d"],
        generated=["a", "a", "b"],
        legal_next_tokens=["cd", "d"],
    ),
    "Optional Patterns with Overlapping Tokens": TestInputs(
        grammar=r'?start: "a" "b"? "c"',
        vocabulary=["a", "b", "bc", "c"],
        generated=["a"],
        legal_next_tokens=["b", "bc", "c"],
    ),
    "Greedy vs. Non-Greedy Matching": TestInputs(
        grammar=r'?start: /a+?/ "b" /c/',
        vocabulary=["a", "aa", "aaa", "b", "c"],
        generated=["a", "a", "b"],
        legal_next_tokens=["c"],
    ),
    "Nested Optional Elements": TestInputs(
        grammar=r'?start: "a" ["b" ["c"]]',
        vocabulary=["a", "b", "bc", "c"],
        generated=["a"],
        legal_next_tokens=[
            "b",
            "bc",
            None,
        ],  # Allowing the document to end after "a" "b"
    ),
    "Recursive Patterns": TestInputs(
        grammar=r'?start: /a(bc)*/ "d"',
        vocabulary=["a", "bc", "bcbcbc", "d"],
        generated=["a", "bc", "d"],
        legal_next_tokens=[None],  # Allowing the document to end after "a" "bc" "d"
    ),
    "Overlapping Character Classes": TestInputs(
        grammar=r'?start: /[ab]+/ "d"',
        vocabulary=["a", "b", "c", "aa", "bb", "cc", "d"],
        generated=["a", "b"],
        legal_next_tokens=["d", "a", "b", "aa", "bb"],
    ),
    "Conditional Patterns": TestInputs(
        grammar=r'?start: "a" /b/ "c" (/d/)?',
        vocabulary=["a", "b", "c", "d"],
        generated=["a", "b", "c"],
        legal_next_tokens=["d", None],  # Allowing the document to end after "a" "b" "c"
    ),
    "Unicode and Special Characters": TestInputs(
        grammar=r'?start: /[a-zA-Z]/ "é" /[0-9]+/',
        vocabulary=["a", "b", "é", "1", "2", "12"],
        generated=["a", "é"],
        legal_next_tokens=["1", "2", "12"],
    ),
    "Unicode and Special Characters Are Choices": TestInputs(
        grammar=r'?start: /[a-zA-Z]/ "é" /[0-9]+/',
        vocabulary=["a", "b", "é", "é9", "2", "12"],
        generated=["a"],
        legal_next_tokens=["é", "é9"],
    ),
    "Whitespace and Ignored Characters": TestInputs(
        grammar=r'?start: "a" / *\s*b/ "c"',
        vocabulary=["a", " b", " c", "c"],
        generated=["a", " b"],
        legal_next_tokens=["c"],
    ),
    "Token Overlaps Multiple Terminals": TestInputs(
        grammar=r'?start: "a" "b" "c" "ab"',
        vocabulary=["a", "b", "bc", "cab", "abc"],
        generated=["a"],
        legal_next_tokens=["b", "bc"],
    ),
    "Interleaved Sequences": TestInputs(
        grammar=r'?start: ("a" "b") | ("a" "c")',
        vocabulary=["a", "b", "c", "ab", "ac"],
        generated=["a"],
        legal_next_tokens=["b", "c"],
    ),
    "Repeated and Nested Patterns": TestInputs(
        grammar=r'?start: "a" ("b" "c")* "d"',
        vocabulary=["a", "b", "c", "bc", "bcc", "cbc", "bcbc", "cbccccd", "d", "bcbcd"],
        generated=["a", "b", "c"],
        legal_next_tokens=["b", "bc", "bcbc", "d", "bcbcd"],
    ),
    "Ambiguous Ending Patterns": TestInputs(
        grammar=r'?start: "a" (/b/)? (/c/)*',
        vocabulary=["a", "b", "c"],
        generated=["a", "b"],
        legal_next_tokens=["c", None],  # Allowing the document to end after "a" "b"
    ),
    "Whitespace Handling in Patterns": TestInputs(
        grammar=r'?start: "a" / *\s*b/ /c /',
        vocabulary=["a", " b", "c"],
        generated=["a", " b"],
        legal_next_tokens=["c"],
    ),
    "Token with Escape Characters": TestInputs(
        grammar=r'?start: "a\n" ("\t")? "b"',
        vocabulary=["a\nb", "a", "b", "\n", "\tb", "\t"],
        generated=["a", "\n"],
        legal_next_tokens=["b", "\t", "\tb"],
    ),
    "Complex Nesting": TestInputs(
        grammar=r'?start: "a" ("b" ("c" "d"))',
        vocabulary=["a", "b", "c", "d"],
        generated=["a", "b", "c"],
        legal_next_tokens=["d"],
    ),
    "Repeated Optional Patterns": TestInputs(
        grammar=r'?start: ("a" ["b"])*',
        vocabulary=["a", "b"],
        generated=["a", "b", "a"],
        legal_next_tokens=["a", "b", None],
    ),
    "Multiple Non-Terminal Symbols": TestInputs(
        grammar=r"""
        ?start: A B
        A: "a"
        B: "b"
        """,
        vocabulary=["a", "b"],
        generated=["a"],
        legal_next_tokens=["b"],
    ),
    "Recursive Definitions": TestInputs(
        grammar=r"""
        ?start: term_a
        term_a: "a" term_a | "b"
        """,
        vocabulary=["a", "b"],
        generated=["a", "a"],
        legal_next_tokens=["a", "b"],
    ),
    "Ignored Patterns": TestInputs(
        grammar=r"""
        ?start: "a" "b" "c"
        %ignore /\s+/
        """,
        vocabulary=["a", "b", "c", " "],
        generated=["a", " ", "b"],
        legal_next_tokens=["c", " "],
    ),
    "Cross-References": TestInputs(
        grammar=r"""
        ?start: term_a
        term_a : /a/ term_b
        term_b : /b/ term_a | /c/
        """,
        vocabulary=["a", "b", "c", "bac"],
        generated=["a"],
        legal_next_tokens=["b", "bac", "c"],
    ),
    "Multiple Complex Non-Terminal Rules": TestInputs(
        grammar=r"""
            ?start: S1 S2 S3
            S1: "a" | "b"
            S2: "c" | "d"
            S3: "e" "f" | "g"
        """,
        vocabulary=["a", "b", "c", "d", "e", "f", "g"],
        generated=["a", "c"],
        legal_next_tokens=["e", "g"],
    ),
    #
    #
    # TODO: fix
    #       parser incorrectly requires consumption of a or b in this case for unknown reasons
    # "Nested Patterns with Repetition": TestInputs(
    #     grammar=r'?start: "a" ("b" | "c")* "d"',
    #     vocabulary=["a", "b", "bc", "bcc", "cbc", "bcbc", "cbccccd", "c", "d" "bcdcb"],
    #     generated=["a"],
    #     legal_next_tokens=["b", "bc", "bcc", "cbc", "bcbc", "cbccccd", "c", "d"],
    # ),
    #
    #
    # TODO: fix
    #       adjacent terminals with ambiguous starts and ends not handled properly
    #       ensure parser isn't greedy incorrectly
    # "Ambiguous Overlapping Patterns": TestInputs(
    #     grammar=r"?start: /ab*/ /bc?/",
    #     vocabulary=["a", "ab", "abc", "b", "bc", "c"],
    #     generated=["a", "b", "b"],
    #     legal_next_tokens=["b", "c", "bc"],
    # ),
    # "Ambiguous Overlapping Patterns In Generation": TestInputs(
    #     grammar=r"?start: /ab*/ /bc?/",
    #     vocabulary=["a", "ab", "abc", "b", "bc", "c", "abbbc"],
    #     generated=["a", "b", "b"],
    #     legal_next_tokens=["b", "c", "bc"],
    # ),
    #
    #
    # SKIP:
    # Awaiting negative lookarounds in interegular
    # "Lookahead and Lookbehind with Nested Conditions": TestInputs(
    #    grammar=r'?start: /(?<=a)b(?=c)/ "d"',
    #    vocabulary=["a", "b", "c", "d"],
    #    generated=["a", "b", "c"],
    #    legal_next_tokens=["d"]
    # ),
    # "Lookbehind Patterns": TestInputs(
    #    grammar=r'?start: /(?<=a)b/ "c"',
    #    vocabulary=["a", "b", "c"],
    #    generated=["a", "b"],
    #    legal_next_tokens=["c"]
    # ),
}


@pytest.mark.parametrize("name", cfg_test_inputs.keys())
def test_cfg_next_token(name, cleanup_lark_import):
    inputs = cfg_test_inputs[name]

    class MockTokenizer:
        vocabulary = {token: i + 1 for i, token in enumerate(inputs.vocabulary)}
        vocabulary["<eos>"] = 0
        reverse_vocab = {i: tok for tok, i in vocabulary.items()}
        special_tokens = {"<eos>"}
        eos_token_id = 0

        def convert_token_to_string(self, token):
            return token

        def decode(self, token_ids):
            if isinstance(token_ids[0], list):
                return [
                    "".join(map(self.reverse_vocab.get, token_ids_sublist))
                    for token_ids_sublist in token_ids
                ]
            return [self.reverse_vocab[token_id] for token_id in token_ids]

    # create a guide and the appropriate state advanced
    # per the inputs generated tokens
    tokenizer = MockTokenizer()
    guide = CFGGuide(inputs.grammar, tokenizer)
    state = guide.initial_state
    for token in inputs.generated:
        state = guide.get_next_state(state, tokenizer.vocabulary[token])
    instruction = guide.get_next_instruction(state)

    # normalize expectations and returned tokens for simple comparison
    returned_next_tokens = sorted(
        {tokenizer.reverse_vocab[int(t)] for t in instruction.tokens}
    )
    expected_next_tokens = sorted(
        {
            t
            if t is not None
            else tokenizer.reverse_vocab[tokenizer.eos_token_id]  # None -> "<eos>"
            for t in inputs.legal_next_tokens
        }
    )

    assert returned_next_tokens == expected_next_tokens


@pytest.fixture(scope="session")
def tokenizer_sentencepiece_gpt2():
    return models.TransformerTokenizer(AutoTokenizer.from_pretrained("gpt2"))


@pytest.fixture(scope="session")
def tokenizer_sentencepiece_llama1():
    return models.TransformerTokenizer(
        AutoTokenizer.from_pretrained(
            "trl-internal-testing/tiny-random-LlamaForCausalLM"
        )
    )


@pytest.fixture(scope="session")
def tokenizer_tiktoken_llama3():
    return models.TransformerTokenizer(
        AutoTokenizer.from_pretrained("yujiepan/llama-3-tiny-random")
    )


@pytest.fixture(scope="session")
def tokenizer_character_level_byt5():
    return models.TransformerTokenizer(
        AutoTokenizer.from_pretrained("google/byt5-small")
    )


# Collects all samples within cfg_samples/ and makes adding
# a test case as easy as adding a valid sample to cfg_samples/
all_samples = {}
examples_path = Path(__file__).parent.parent / "cfg_samples"
for sample_collection_path in examples_path.iterdir():
    grammar_name = sample_collection_path.name
    grammar = getattr(grammars, grammar_name)
    for sample_path in sample_collection_path.iterdir():
        test_name = f"{grammar_name}_{sample_path.name}"
        with open(sample_path) as f:
            all_samples[test_name] = (grammar_name, grammar, f.read().rstrip("\n"))


@pytest.mark.parametrize("sample_name", all_samples.keys())
def test_cfg_test_sample_valid_with_lark(sample_name):
    """assert the provided sample is valid (testing the test itself)"""
    from lark import Lark, UnexpectedToken

    grammar_name, grammar_str, sample = all_samples[sample_name]
    try:
        parser = Lark(grammar_str, parser="lalr", import_paths=[grammars.GRAMMAR_PATH])
        parser = parser.parse_interactive(sample)
        token = parser.exhaust_lexer()[-1]
        parser.feed_eof(token)
    except UnexpectedToken as e:
        raise Exception(
            f"Invalid test, sample '{sample_name}' isn't a legal generation of '{grammar_name}':\n{e}"
        )


@pytest.mark.parametrize("sample_name", all_samples.keys())
@pytest.mark.parametrize(
    "tokenizer_name",
    [
        "tokenizer_sentencepiece_gpt2",
        "tokenizer_sentencepiece_llama1",
        "tokenizer_tiktoken_llama3",
        "tokenizer_character_level_byt5",
    ],
)
def test_cfg_grammar_sample(request, sample_name, tokenizer_name, cleanup_lark_import):
    """Test whether CFG can generate the exact token sequence as tokenizer.encode(sample) produces"""

    # TODO: enable these tests once improvements are made
    if (
        tokenizer_name != "tokenizer_character_level_byt5"
        or sample_name == "json_outlines.generate.samplers.mypy.json.test"
    ):
        pytest.skip("CFG is too slow, skipping tests for this tokenizer")
    elif sample_name == "arithmetic_lots_of_ops.arithmetic.test":
        pytest.skip("CFG incorrectly handles this valid sample, skipping until bugfix")

    tokenizer = request.getfixturevalue(tokenizer_name)

    grammar_name, grammar_str, sample = all_samples[sample_name]
    cfg_guide = CFGGuide(grammar_str, tokenizer)

    sample_token_ids = tokenizer.tokenizer.encode(
        sample, add_special_tokens=False, return_tensors="pt"
    )[0]
    assert (
        len(sample_token_ids.shape) == 1
    )  # ensure we're encoding in the desired shape for this test

    state = cfg_guide.initial_state
    for i, token_id in enumerate(sample_token_ids):
        if tokenizer.decode([token_id])[0] == "":
            continue
        next_instruction = cfg_guide.get_next_instruction(state)
        if token_id not in next_instruction.tokens:
            processed_str = tokenizer.decode([sample_token_ids[:i]])[0]
            remaining_str = tokenizer.decode([sample_token_ids[i:]])[0]
            if next_instruction.tokens == [tokenizer.eos_token_id]:
                error_label = "CFGGuide required EOS early"
            else:
                expected = tokenizer.decode(next_instruction.tokens)
                error_label = (
                    f"Mismatched expectations, Guide expected {sorted(expected)}"
                )
            raise Exception(
                f"{error_label}\n"
                f"processed:\n```{processed_str}```\n"
                f"remaining:\n```{remaining_str}```"
            )
            next_instruction.tokens
        state = cfg_guide.get_next_state(state, token_id)
    final_instruction = cfg_guide.get_next_instruction(state)
    assert tokenizer.eos_token_id in final_instruction.tokens
