import torch
from transformers import AutoTokenizer

from outlines.fsm.guide import AlignmentGuide, RegexGuide
from outlines.models.transformers import TransformerTokenizer


class MockTokenizer:
    def __init__(self, vocabulary):
        self.vocabulary = {tok: i for i, tok in enumerate(vocabulary)}
        self.vocabulary["<eos>"] = len(self.vocabulary)
        self.special_tokens = {"<eos>"}
        self.eos_token_id = self.vocabulary["<eos>"]
        self.pad_token_id = -1

        self.inverse_vocabulary = {i: tok for tok, i in self.vocabulary.items()}

    def convert_token_to_string(self, token):
        return token

    def decode(self, token_ids):
        if token_ids == []:
            return ""
        if isinstance(list(token_ids)[0], list):
            return [
                "".join(map(self.inverse_vocabulary.get, token_ids_sublist))
                for token_ids_sublist in token_ids
            ]
        return [self.inverse_vocabulary[int(token_id)] for token_id in token_ids]

    def encode(self, texts):
        """
        Encodes the input texts by finding the longest matching tokens in the vocabulary.
        """
        seqs = []
        for text in texts:
            tokens = []
            while text:
                token = next(
                    (
                        tok
                        for tok in sorted(self.vocabulary, key=len, reverse=True)
                        if text.startswith(tok)
                    ),
                    None,
                )
                if token is None:
                    tokens = [self.pad_token_id]
                    break
                tokens.append(self.vocabulary[token])
                text = text[len(token) :]
            seqs.append(tokens)

        max_len = max(len(seq) for seq in seqs)
        padded_seqs = torch.tensor(
            [seq + [self.pad_token_id] * (max_len - len(seq)) for seq in seqs]
        )
        return padded_seqs, None


def test_alignment_with_pseudo_token_and_regex_guide():
    # Mock tokenizer with the vocabulary for "hello", "world", "wo", "rld", and "!"
    tokenizer = MockTokenizer(["hello", " world", " wo", "rld", "!"])
    prompt = "hello wo"

    # Create a RegexGuide that expects the sequence "rld!"
    child_guide = RegexGuide(regex_string="rld!", tokenizer=tokenizer)

    # Create the AlignmentGuide with the child guide
    guide = AlignmentGuide(prompt, tokenizer, child_guide=child_guide)

    assert guide.alignment_prompt == "hello"

    # assert " world!" is legal and final
    seq = [tokenizer.vocabulary[" world"], tokenizer.vocabulary["!"]]
    assert guide.accepts(seq)
    assert guide.is_final_state(guide.derive(seq, guide.initial_state)) is True


def test_alignment_guide_gpt2_url():
    # Based on notebook
    # https://github.com/guidance-ai/guidance/blob/af63e6/notebooks/tutorials/token_healing.ipynb#L4
    tokenizer = TransformerTokenizer(AutoTokenizer.from_pretrained("gpt2"))
    prompt = "The url of Google is http:"
    guide = AlignmentGuide(prompt, tokenizer)
    assert guide.alignment_prompt == "The url of Google is http"
    assert guide.accepts(list(tokenizer.encode("://google.com")[0][0]))
