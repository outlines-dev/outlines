from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from datasets.fingerprint import Hasher

from outlines.models.tokenizer import Tokenizer

if TYPE_CHECKING:
    import torch
    from transformers import (
        PreTrainedModel,
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
    )

__all__ = ["transformers"]


KVCacheType = Tuple[Tuple["torch.DoubleTensor", "torch.DoubleTensor"], ...]


def get_llama_tokenizer_types():
    """Get all the Llama tokenizer types/classes that need work-arounds.

    When they can't be imported, a dummy class is created.

    """
    try:
        from transformers.models.llama import LlamaTokenizer
    except ImportError:

        class LlamaTokenizer:  # type: ignore
            pass

    try:
        from transformers.models.llama import LlamaTokenizerFast
    except ImportError:

        class LlamaTokenizerFast:  # type: ignore
            pass

    try:
        from transformers.models.code_llama import CodeLlamaTokenizer
    except ImportError:

        class CodeLlamaTokenizer:  # type: ignore
            pass

    try:
        from transformers.models.code_llama import CodeLlamaTokenizerFast
    except ImportError:

        class CodeLlamaTokenizerFast:  # type: ignore
            pass

    return (
        LlamaTokenizer,
        LlamaTokenizerFast,
        CodeLlamaTokenizer,
        CodeLlamaTokenizerFast,
    )


class TransformerTokenizer(Tokenizer):
    """Represents a tokenizer for models in the `transformers` library."""

    def __init__(
        self,
        tokenizer: Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"],
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.eos_token = self.tokenizer.eos_token

        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.pad_token_id = self.eos_token_id
        else:
            self.pad_token_id = self.tokenizer.pad_token_id
            self.pad_token = self.tokenizer.pad_token

        self.special_tokens = set(self.tokenizer.all_special_tokens)

        self.vocabulary = self.tokenizer.get_vocab()
        self.is_llama = isinstance(self.tokenizer, get_llama_tokenizer_types())

        # set to None if no system message
        self.SYSTEM_MESSAGE = {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
        self.USER_MESSAGE_TEMPLATE = lambda x: {"role": "user", "content": x}
        self.ASSISTANT_MESSAGE_TEMPLATE = lambda x: {"role": "assistant", "content": x}

    def encode(
        self, prompt: Union[str, List[str]], **kwargs
    ) -> Tuple["torch.LongTensor", "torch.LongTensor"]:
        kwargs["padding"] = True
        kwargs["return_tensors"] = "pt"
        output = self.tokenizer(prompt, **kwargs)
        return output["input_ids"], output["attention_mask"]

    def decode(self, token_ids: "torch.LongTensor") -> List[str]:
        text = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        return text

    def convert_token_to_string(self, token: str) -> str:
        from transformers.file_utils import SPIECE_UNDERLINE

        string = self.tokenizer.convert_tokens_to_string([token])

        if self.is_llama:
            # A hack to handle missing spaces to HF's Llama tokenizers
            if token.startswith(SPIECE_UNDERLINE) or token == "<0x20>":
                return " " + string

        return string

    def get_messages(self, prompt: str) -> List[Dict[str, str]]:
        """
        Template for one-turn chat.
        """
        return (
            [self.SYSTEM_MESSAGE.copy(), self.USER_MESSAGE_TEMPLATE(prompt)]
            if self.SYSTEM_MESSAGE is not None
            else [self.USER_MESSAGE_TEMPLATE(prompt)]
        )

    def apply_chat_template(self, prompt: str) -> str:
        messages = self.get_messages(prompt)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def __eq__(self, other):
        if isinstance(other, type(self)):
            if hasattr(self, "model_name") and hasattr(self, "kwargs"):
                return (
                    other.model_name == self.model_name and other.kwargs == self.kwargs
                )
            else:
                return other.tokenizer == self.tokenizer
        return NotImplemented

    def __hash__(self):
        return hash(Hasher.hash(self.tokenizer))

    def __getstate__(self):
        state = {"tokenizer": self.tokenizer}
        return state

    def __setstate__(self, state):
        self.__init__(state["tokenizer"])


class Transformers:
    """Represents a `transformers` model."""

    def __init__(
        self,
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
    ):
        self.device = model.device
        self.model = model
        self.tokenizer = TransformerTokenizer(tokenizer)

    def forward(
        self,
        input_ids: "torch.LongTensor",
        attention_mask: "torch.LongTensor",
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple["torch.FloatTensor", Optional[KVCacheType]]:
        """Compute a forward pass through the transformer model.

        Parameters
        ----------
        input_ids
            The input token ids.  Must be one or two dimensional.
        attention_mask
            The attention mask.  Must be one or two dimensional.
        past_key_values
            A tuple of tuples containing the cached key and value tensors for each
            attention head.

        Returns
        -------
        The computed logits and the new cached key and value tensors.

        """
        try:
            import torch
        except ImportError:
            ImportError(
                "The `torch` library needs to be installed to use `transformers` models."
            )
        assert 0 < input_ids.ndim < 3

        if past_key_values:
            input_ids = input_ids[..., -1].unsqueeze(-1)

        with torch.inference_mode():
            output = self.model(
                input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
                past_key_values=past_key_values,
            )

        return output.logits, output.past_key_values

    def __call__(
        self,
        input_ids: "torch.LongTensor",
        attention_mask: "torch.LongTensor",
        past_key_values: Optional[Tuple] = None,
    ) -> "torch.FloatTensor":
        logits, kv_cache = self.forward(input_ids, attention_mask, past_key_values)
        next_token_logits = logits[..., -1, :]

        return next_token_logits, kv_cache


def transformers(
    model_name: str,
    device: Optional[str] = None,
    model_kwargs: dict = {},
    tokenizer_kwargs: dict = {},
):
    """Instantiate a model from the `transformers` library and its tokenizer.

    Parameters
    ----------
    model_name
        The name of the model as listed on Hugging Face's model page.
    device
        The device(s) on which the model should be loaded. This overrides
        the `device_map` entry in `model_kwargs` when provided.
    model_kwargs
        A dictionary that contains the keyword arguments to pass to the
        `from_pretrained` method when loading the model.
    tokenizer_kwargs
        A dictionary that contains the keyword arguments to pass to the
        `from_pretrained` method when loading the tokenizer.

    Returns
    -------
    A `TransformersModel` model instance.

    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "The `transformers` library needs to be installed in order to use `transformers` models."
        )

    if device is not None:
        model_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    tokenizer_kwargs.setdefault("padding_side", "left")
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

    return Transformers(model, tokenizer)
