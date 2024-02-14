from typing import TYPE_CHECKING, Optional

import torch

from outlines.models.model import OutlinesModel

from .transformers import TransformerTokenizer

if TYPE_CHECKING:
    from exllamav2 import ExLlamaV2, ExLlamaV2Cache
    from transformers import PreTrainedTokenizer


class ExLlamaV2Model(OutlinesModel):
    """Represents a `exl2` model."""

    def __init__(
        self,
        model: "ExLlamaV2",
        tokenizer: "PreTrainedTokenizer",
        device,
        cache: "ExLlamaV2Cache",
    ):
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache
        self.past_seq = None

    def forward(self, input_ids: torch.LongTensor, *_):
        """Compute a forward pass through the exl2 model."""

        # Caching with past_seq
        reset = True
        seq_tensor = input_ids[0]

        if self.past_seq is not None:
            min_length = min(self.past_seq.shape[0], seq_tensor.shape[0])
            indices = torch.nonzero(
                ~torch.eq(self.past_seq[:min_length], seq_tensor[:min_length])
            )
            if len(indices) > 0:
                longest_prefix = indices[0].item()
            else:
                longest_prefix = min_length

            if longest_prefix > 0:
                reset = False
                self.cache.current_seq_len = longest_prefix
                if seq_tensor.shape[0] - longest_prefix > 1:
                    self.model.forward(
                        seq_tensor[longest_prefix:-1].view(1, -1),
                        self.cache,
                        preprocess_only=True,
                    )
                elif seq_tensor.shape[0] == longest_prefix:
                    self.cache.current_seq_len -= 1

        if reset:
            self.cache.current_seq_len = 0
            if seq_tensor.shape[0] > 1:
                self.model.forward(
                    seq_tensor[:-1].view(1, -1),
                    self.cache,
                    preprocess_only=True,
                )

        self.past_seq = seq_tensor

        return self.model.forward(seq_tensor[-1:].view(1, -1), self.cache)

    def __call__(self, input_ids: torch.LongTensor, *_) -> torch.FloatTensor:
        logits = self.forward(input_ids)
        next_token_logits = logits[..., -1, :]

        return next_token_logits, None


def exl2(
    model_name: str,
    device: Optional[str] = None,
    model_kwargs: dict = {},
    tokenizer_kwargs: dict = {},
):
    try:
        from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config
    except ImportError:
        raise ImportError(
            "The `exllamav2` library needs to be installed in order to use `exllamav2` models."
        )

    config = ExLlamaV2Config()
    config.model_dir = model_name
    config.prepare()

    config.max_seq_len = model_kwargs.pop("max_seq_len", config.max_seq_len)
    config.scale_pos_emb = model_kwargs.pop("scale_pos_emb", config.scale_pos_emb)
    config.scale_alpha_value = model_kwargs.pop(
        "scale_alpha_value", config.scale_alpha_value
    )
    config.no_flash_attn = model_kwargs.pop("no_flash_attn", config.no_flash_attn)
    config.num_experts_per_token = int(
        model_kwargs.pop("num_experts_per_token", config.num_experts_per_token)
    )

    model = ExLlamaV2(config)

    split = None
    if "gpu_split" in model_kwargs.keys():
        split = [float(alloc) for alloc in model_kwargs["gpu_split"].split(",")]

    model.load(split)
    tokenizer = TransformerTokenizer(model_name, **tokenizer_kwargs)
    cache = ExLlamaV2Cache(model)

    return ExLlamaV2Model(model, tokenizer, device, cache)
