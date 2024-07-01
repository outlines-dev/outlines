import pytest
import torch
from transformers import AutoTokenizer
from transformers.models.gpt2 import GPT2TokenizerFast

from outlines.models.transformers import TransformerTokenizer, transformers

TEST_MODEL = "hf-internal-testing/tiny-random-GPTJForCausalLM"


def test_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL, padding_side="left")
    tokenizer = TransformerTokenizer(tokenizer)
    assert tokenizer.eos_token_id == 0
    assert tokenizer.pad_token_id == 0
    assert isinstance(tokenizer.tokenizer, GPT2TokenizerFast)

    token_ids, attention_mask = tokenizer.encode("Test")
    assert token_ids.ndim == 2
    assert token_ids.shape[0] == 1
    assert isinstance(token_ids, torch.LongTensor)
    assert token_ids.shape == attention_mask.shape

    token_ids, attention_mask = tokenizer.encode(["Test", "Test"])
    assert token_ids.ndim == 2
    assert token_ids.shape[0] == 2
    assert isinstance(token_ids, torch.LongTensor)
    assert token_ids.shape == attention_mask.shape

    token_ids, attention_mask = tokenizer.encode(["Test", "A long sentence"])
    assert token_ids.shape == attention_mask.shape
    assert attention_mask[0][0] == tokenizer.pad_token_id

    text = tokenizer.decode(torch.tensor([[0, 1, 2]]))
    isinstance(text, str)

    text = tokenizer.decode(torch.tensor([[0, 1, 2], [3, 4, 5]]))
    isinstance(text, list)
    isinstance(text[0], str)
    isinstance(text[1], str)

    tokenizer = AutoTokenizer.from_pretrained(
        TEST_MODEL, additional_special_tokens=["<t1>", "<t2>"]
    )
    tokenizer = TransformerTokenizer(tokenizer)
    assert "<t1>" in tokenizer.special_tokens
    assert "<t2>" in tokenizer.special_tokens


def test_llama_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    tokenizer = TransformerTokenizer(tokenizer)

    # Broken
    assert tokenizer.tokenizer.convert_tokens_to_string(["▁baz"]) == "baz"
    assert tokenizer.tokenizer.convert_tokens_to_string(["<0x20>"]) == ""
    assert tokenizer.tokenizer.convert_tokens_to_string(["▁▁▁"]) == "  "

    # Not broken
    assert tokenizer.convert_token_to_string("▁baz") == " baz"
    assert tokenizer.convert_token_to_string("<0x20>") == " "
    assert tokenizer.convert_token_to_string("▁▁▁") == "   "


def test_model():
    model = transformers(TEST_MODEL, device="cpu")
    assert isinstance(model.tokenizer, TransformerTokenizer)
    assert model.model.device.type == "cpu"

    model = transformers(TEST_MODEL, model_kwargs={"device_map": "cpu"})
    assert isinstance(model.tokenizer, TransformerTokenizer)
    assert model.model.device.type == "cpu"

    model = transformers(TEST_MODEL, device="cpu", model_kwargs={"device_map": "cuda"})
    assert isinstance(model.tokenizer, TransformerTokenizer)
    assert model.model.device.type == "cpu"

    input_ids = torch.tensor([[0, 1, 2]])
    logits, kv_cache = model(input_ids, torch.ones_like(input_ids))
    assert logits.type() == "torch.FloatTensor"
    assert logits.ndim == 2
    assert logits.shape[0] == 1
    assert len(kv_cache) == model.model.config.n_layer
    assert len(kv_cache[0]) == 2
    assert kv_cache[0][0].shape[1] == model.model.config.n_head
    assert kv_cache[0][0].shape[2] == 3  # number of tokens

    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    logits, kv_cache = model(input_ids, torch.ones_like(input_ids))
    assert logits.type() == "torch.FloatTensor"
    assert logits.ndim == 2
    assert logits.shape[0] == 3
    assert len(kv_cache) == model.model.config.n_layer
    assert len(kv_cache[0]) == 2
    assert kv_cache[0][0].shape[1] == model.model.config.n_head
    assert kv_cache[0][0].shape[2] == 3  # number of tokens

    with pytest.raises(AssertionError):
        input_ids = torch.tensor([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [0, 1, 2]]])
        logits = model(input_ids, torch.ones_like(input_ids))


def test_tokenizer_eq_hash():
    tokenizer_hf = AutoTokenizer.from_pretrained("gpt2")

    tokenizer = TransformerTokenizer(tokenizer_hf)
    tokenizer_2 = TransformerTokenizer(tokenizer_hf)

    assert tokenizer == tokenizer_2
    assert hash(tokenizer) == hash(tokenizer_2)

    tokenizer_hf_2 = AutoTokenizer.from_pretrained("gpt2")
    tokenizer_hf_2.add_tokens(["test_token"])

    tokenizer_3 = TransformerTokenizer(tokenizer_hf_2)
    assert tokenizer != tokenizer_3
    assert hash(tokenizer) != hash(tokenizer_3)
