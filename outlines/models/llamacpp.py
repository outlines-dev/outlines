import ctypes
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray

from outlines.models.model import OutlinesModel
from outlines.models.tokenizer import Tokenizer


class LlamaCpp(OutlinesModel):
    """Represents a `llama_cpp` model."""

    def __init__(
        self, llama_instance, model, tokenizer, device, context_params, **kwargs
    ):
        self.device = device
        self.llama_instance = llama_instance
        self.tokenizer = tokenizer

        # Note: the concept of padding does not exist in llama.cpp as a batched sequence is just
        # a flat array of tokens that can be assigned to one or more sequences.
        # To make it compatible with the transformers inspired tokenizer interface
        # we need a padding token to homogenize to token_ids tensor.
        self.pad_token_id = -1

        self.n_past = 0
        self.n_vocab = kwargs.pop("n_vocab")

        self.ctx = llama_instance.llama_new_context_with_model(model, context_params)

    def forward(self, input_ids: torch.LongTensor, *_):
        """Compute a forward pass through the llama_cpp model."""
        if input_ids.ndim == 2:
            seq_tensor = input_ids[:, self.n_past :]
        elif input_ids.ndim == 1:
            seq_tensor = input_ids.view(1, -1)[:, self.n_past :]
        else:
            raise Exception("Only one and two dimensional inputs allowed.")

        tokens_total = torch.numel(seq_tensor[seq_tensor != self.pad_token_id])
        batch = self.llama_instance.llama_batch_init(tokens_total, 0, 1)

        seq_token_ids = []
        for seq_idx, seq in enumerate(seq_tensor):
            for token_pos, token_id in enumerate(seq):
                if token_id == self.pad_token_id:
                    break
                batch.token[batch.n_tokens] = token_id.item()
                batch.pos[batch.n_tokens] = token_pos
                batch.seq_id[batch.n_tokens][0] = seq_idx
                batch.n_seq_id[batch.n_tokens] = 1
                batch.logits[batch.n_tokens] = False

                batch.n_tokens += 1
                self.n_past += 1

            batch.logits[batch.n_tokens - 1] = True
            seq_token_ids.append(batch.n_tokens - 1)

        if self.llama_instance.llama_decode(self.ctx, batch) != 0:
            print("Error decoding")

        all_logits = []
        for seq_token in seq_token_ids:
            logits = self.llama_instance.llama_get_logits_ith(self.ctx, seq_token)
            logits_list = (ctypes.c_float * self.n_vocab)(
                *[logits[token_id] for token_id in range(self.n_vocab)]
            )
            logits_tensor = torch.tensor(logits_list)
            all_logits.append(logits_tensor)

        self.llama_instance.llama_batch_free(batch)

        stacked_logits = torch.stack(all_logits)
        return stacked_logits, None

    def __call__(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        past_key_values: Optional[Tuple] = None,
    ) -> torch.FloatTensor:
        logits, kv_cache = self.forward(input_ids, attention_mask, past_key_values)
        next_token_logits = logits

        return next_token_logits, kv_cache


class LlamaCppTokenizer(Tokenizer):
    def __init__(self, llama_instance, model, model_name: str, **kwargs):
        self.model_name = model_name
        self.llama_instance = llama_instance
        self.is_llama = False

        self.model = model
        self.n_vocab = kwargs.pop("n_vocab")

        self.eos_token_id = llama_instance.llama_token_eos(model)
        self.eos_token = self._get_eos_token()
        self.pad_token_id = -1
        self.bos_token_id = llama_instance.llama_token_eos(model)
        self.nl_token_id = llama_instance.llama_token_nl(model)
        self.vocabulary = {}
        self._create_vocabulary()

        self.n_past = 0

        self.special_tokens = {
            self.eos_token_id,
            self.pad_token_id,
            self.bos_token_id,
            self.nl_token_id,
        }

    def _create_vocabulary(self):
        for t in range(self.n_vocab):
            size = 32
            buffer = (ctypes.c_char * size)()
            n = self.llama_instance.llama_token_to_piece(
                self.model, self.llama_instance.llama_token(t), buffer, size
            )

            try:
                token_piece = buffer[:n].decode("utf-8")
                self.vocabulary[token_piece] = t
            except Exception as e:
                print(f"Failed to convert token ({buffer[:n]}): {e}")
                continue

    def encode(
        self, prompt: Union[str, List[str]]
    ) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        if isinstance(prompt, list):
            prompts = prompt
        else:
            prompts = [prompt]

        max_len = 0
        token_ids = []
        for p in prompts:
            embd_inp = (self.llama_instance.llama_token * (len(p) + 1))()

            n_of_tok = self.llama_instance.llama_tokenize(
                model=self.model,
                text=bytes(str(p), "utf-8"),
                text_len=len(embd_inp),
                tokens=embd_inp,
                n_max_tokens=len(embd_inp),
                add_bos=self.n_past == 0,
                special=False,
            )

            self.n_past += n_of_tok

            if n_of_tok > max_len:
                max_len = n_of_tok

            embd_inp = embd_inp[:n_of_tok]
            token_ids.append(np.array(embd_inp))

        max_len = np.max([len(a) for a in token_ids])
        padded = np.asarray(
            [
                np.pad(
                    a,
                    (0, max_len - len(a)),
                    "constant",
                    constant_values=self.pad_token_id,
                )
                for a in token_ids
            ]
        )

        token_ids = torch.LongTensor(padded)
        return token_ids, torch.ones_like(token_ids)

    def decode(self, token_ids: NDArray[np.int64]) -> List[str]:
        if isinstance(token_ids, list):
            token_ids = np.array(token_ids)
        if token_ids.ndim == 1:
            token_ids = [token_ids]

        pieces = []
        for tokens in token_ids:
            seq = []
            for id in tokens:
                size = 32
                buffer = (ctypes.c_char * size)()
                n = self.llama_instance.llama_token_to_piece(
                    self.model, self.llama_instance.llama_token(id), buffer, size
                )

                token_piece = buffer[:n].decode("utf-8")  # type: ignore

                seq.append(token_piece)

            pieces.append("".join(seq))

        return pieces

    def _get_eos_token(self):
        size = 32
        buffer = (ctypes.c_char * size)()
        n = self.llama_instance.llama_token_to_piece(
            self.model, self.llama_instance.llama_token(self.eos_token_id), buffer, size
        )

        token_piece = buffer[:n].decode("utf-8")

        return token_piece

    def convert_token_to_string(self, token: str) -> str:
        return token

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return other.model_name == self.model_name and other.kwargs == self.kwargs
        return NotImplemented

    def __hash__(self):
        return hash(self.model_name)


def llamacpp(
    model_name: str,
    device: Optional[str] = None,
    model_kwargs: dict = {},
    tokenizer_kwargs: dict = {},
):
    try:
        import llama_cpp
    except ImportError:
        raise ImportError(
            "The `llama-cpp-python` library needs to be installed in order to use LlamaCpp."
        )

    if device is None:
        device = "cpu"

    llama_cpp.llama_backend_init(numa=False)

    model_params = llama_cpp.llama_model_default_params()

    if "cuda" in device:
        model_params.n_gpu_layers = 999
    else:
        model_params.n_gpu_layers = model_kwargs.pop(
            "n_gpu_layers", model_params.n_gpu_layers
        )

    if "tensor_split" in model_kwargs.keys():
        tensor_split = model_kwargs.get("tensor_split")
        if isinstance(tensor_split, list):
            tensor_split_arr = (ctypes.c_float * llama_cpp.LLAMA_MAX_DEVICES)(
                *[t for t in tensor_split]
            )
            model_params.tensor_split = tensor_split_arr

    context_params = llama_cpp.llama_context_default_params()
    context_params.n_batch = model_kwargs.pop("n_batch", context_params.n_batch)
    context_params.n_ctx = model_kwargs.pop("n_ctx", context_params.n_ctx)
    context_params.n_threads = model_kwargs.pop("n_threads", context_params.n_threads)
    context_params.n_threads_batch = model_kwargs.pop(
        "n_threads_batch", context_params.n_threads_batch
    )
    context_params.rope_scaling_type = model_kwargs.pop(
        "rope_scaling_type", context_params.rope_scaling_type
    )
    context_params.rope_freq_base = model_kwargs.pop(
        "rope_freq_base", context_params.rope_freq_base
    )
    context_params.rope_freq_scale = model_kwargs.pop(
        "rope_freq_scale", context_params.rope_freq_scale
    )
    context_params.yarn_ext_factor = model_kwargs.pop(
        "yarn_ext_factor", context_params.yarn_ext_factor
    )
    context_params.yarn_attn_factor = model_kwargs.pop(
        "yarn_attn_factor", context_params.yarn_attn_factor
    )
    context_params.yarn_beta_fast = model_kwargs.pop(
        "yarn_beta_fast", context_params.yarn_beta_fast
    )
    context_params.yarn_beta_slow = model_kwargs.pop(
        "yarn_beta_slow", context_params.yarn_beta_slow
    )
    context_params.yarn_orig_ctx = model_kwargs.pop(
        "yarn_orig_ctx", context_params.yarn_orig_ctx
    )
    context_params.offload_kqv = model_kwargs.pop(
        "offload_kqv", context_params.offload_kqv
    )

    model = llama_cpp.llama_load_model_from_file(
        model_name.encode("utf-8"), model_params
    )

    model_kwargs["n_vocab"] = llama_cpp.llama_n_vocab(model)
    tokenizer_kwargs["n_vocab"] = model_kwargs.get("n_vocab")

    tokenizer = LlamaCppTokenizer(llama_cpp, model, model_name, **tokenizer_kwargs)

    return LlamaCpp(llama_cpp, model, tokenizer, "cpu", context_params, **model_kwargs)
