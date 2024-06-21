import dataclasses
from typing import TYPE_CHECKING, Generator, Iterator, List, Optional, Tuple, Union

from .transformers import TransformerTokenizer

if TYPE_CHECKING:
    import mlx.core as mx
    import mlx.nn as nn
    from transformers import PreTrainedTokenizer

    from outlines.generate.api import GenerationParameters, SamplingParameters
    from outlines.processors import OutlinesLogitsProcessor


class MLXLM:
    """
    Represents an `mlx_lm` model
    """

    def __init__(
        self,
        model: "nn.Module",
        tokenizer: "PreTrainedTokenizer",
    ):
        self.model = model
        self.mlx_tokenizer = tokenizer  # returns mlx tensors, used for encode()
        self.tokenizer = TransformerTokenizer(
            tokenizer._tokenizer
        )  # _tokenizer is HF Tokenizer

    def generate(
        self,
        prompts: Union[str, List[str]],
        generation_parameters: "GenerationParameters",
        logits_processor,
        sampling_parameters: "SamplingParameters",
    ) -> str:
        streamer = self.stream(
            prompts, generation_parameters, logits_processor, sampling_parameters
        )
        return "".join(list(streamer))

    def stream(
        self,
        prompts: Union[str, List[str]],
        generation_parameters: "GenerationParameters",
        logits_processor,
        sampling_parameters: "SamplingParameters",
    ) -> Iterator[str]:
        """Generate text using `mlx_lm`.

        Arguments
        ---------
        prompts
            A prompt or list of prompts.
        generation_parameters
            An instance of `GenerationParameters` that contains the prompt,
            the maximum number of tokens, stop sequences and seed. All the
            arguments to `SequenceGeneratorAdapter`'s `__cal__` method.
        logits_processor
            The logits processor to use when generating text.
        sampling_parameters
            An instance of `SamplingParameters`, a dataclass that contains
            the name of the sampler to use and related parameters as available
            in Outlines.
        Returns
        -------
        The generated text.
        """
        import mlx.core as mx

        max_tokens, stop_at, seed = dataclasses.astuple(generation_parameters)
        sampler, num_samples, top_p, top_k, temperature = dataclasses.astuple(
            sampling_parameters
        )
        if max_tokens is None:
            max_tokens = int(1e9)

        if not isinstance(prompts, str):
            raise NotImplementedError(
                "The `mlx-lm` library does not support batch inference."
            )
        if sampler == "beam_search":
            raise NotImplementedError(
                "The `mlx-lm` library does not support Beam Search."
            )
        if num_samples != 1:
            raise NotImplementedError(
                "The `mlx-lm` library does not allow to take several samples."
            )
        if top_k is not None:
            raise NotImplementedError("The `mlx-lm` library does not support top_k.")
        if seed is not None:
            raise NotImplementedError("The `mlx-lm` library does not support seed.")
        if stop_at is not None:
            raise NotImplementedError("The `mlx-lm` library does not support stop_at.")

        generate_kwargs = {
            "temp": temperature,
            "top_p": top_p,
            "sampler": sampler,
            "logits_processor": logits_processor,
        }

        # Adapted from
        # https://github.com/ml-explore/mlx-examples/blob/4872727/llms/mlx_lm/utils.py#L267
        prompt_tokens = mx.array(self.mlx_tokenizer.encode(prompts))

        for (token, prob), n in zip(
            self.generate_step(prompt_tokens, **generate_kwargs),
            range(max_tokens),
        ):
            if token == self.tokenizer.eos_token_id:
                break
            yield self.tokenizer.decode([token])[0]

    def generate_step(
        self,
        prompt: "mx.array",
        temp: Optional[float],
        top_p: Optional[float],
        sampler: str,
        logits_processor: "OutlinesLogitsProcessor",
    ) -> Generator[Tuple[int, float], None, None]:
        """
        Adapted from
        https://github.com/ml-explore/mlx-examples/blob/4872727/llms/mlx_lm/utils.py#L129

        A generator producing token ids based on the given prompt from the model.

            Args:
                prompt (mx.array): The input prompt.
                temp (float): The temperature for sampling, if 0 the argmax is used.
                  Default: ``0``.
                top_p (float, optional): Nulceus sampling, higher means model considers
                  more less likely words.
                sampler (str): The sampler string defined by SequenceGeneratorAdapter
                logits_processor (OutlinesLogitsProcessor): Augment logits before sampling.
        """
        import mlx.core as mx
        import mlx_lm

        temperature: float = temp or 1.0

        def sample(logits: "mx.array") -> Tuple["mx.array", float]:
            softmax_logits = mx.softmax(logits)

            if temperature == 0.0 or sampler == "greedy":
                token = mx.argmax(logits, axis=-1)
            elif sampler == "multinomial":
                if top_p is not None and top_p > 0 and top_p < 1.0:
                    token = mlx_lm.sample_utils.top_p_sampling(
                        logits, top_p, temperature
                    )
                else:
                    token = mx.random.categorical(logits * (1 / temperature))
            else:
                raise ValueError(f"Invalid mlx-lm sampler: `{sampler}`")

            prob = softmax_logits[0, token]
            return token, prob

        kv_heads = (
            [self.model.n_kv_heads] * len(self.model.layers)
            if isinstance(self.model.n_kv_heads, int)
            else self.model.n_kv_heads
        )
        cache = [mlx_lm.models.base.KVCache(self.model.head_dim, n) for n in kv_heads]

        # kv cache contains processed input IDs, we pass the unprocessed inputs and cache to model()
        unprocessed_input_ids = prompt
        generated_ids: List[int] = []

        while True:
            logits = self.model(unprocessed_input_ids[None], cache=cache)
            logits = logits[:, -1, :]

            if logits_processor is not None:
                # convert to logits_processor 1d expectation, apply, then convert back
                logits_1d = logits.reshape(-1)
                logits_1d = logits_processor(generated_ids, logits_1d)
                logits = logits_1d.reshape(1, -1)

            new_token_single, prob = sample(logits)
            new_token = new_token_single.item()
            yield new_token, prob

            generated_ids.append(new_token)
            unprocessed_input_ids = new_token_single


def mlxlm(
    model_name: str,
    tokenizer_config: dict = {},
    model_config: dict = {},
    adapter_path: Optional[str] = None,
    lazy: bool = False,
):
    """Instantiate a model from the `mlx_lm` library and its tokenizer.

    Signature adapted from
    https://github.com/ml-explore/mlx-examples/blob/4872727/llms/mlx_lm/utils.py#L422

    Parameters
    ----------
    Args:
        path_or_hf_repo (Path): The path or the huggingface repository to load the model from.
        tokenizer_config (dict, optional): Configuration parameters specifically for the tokenizer.
            Defaults to an empty dictionary.
        model_config(dict, optional): Configuration parameters specifically for the model.
            Defaults to an empty dictionary.
        adapter_path (str, optional): Path to the LoRA adapters. If provided, applies LoRA layers
            to the model. Default: ``None``.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``

    Returns
    -------
    A `MLXLM` model instance.

    """
    try:
        import mlx.core as mx
        import mlx_lm
    except ImportError:
        raise ImportError(
            "The `mlx_lm` library needs to be installed in order to use `mlx_lm` models."
        )
    if not mx.metal.is_available():
        raise RuntimeError("You cannot use `mlx_lm` without Apple Silicon (Metal)")

    model, tokenizer = mlx_lm.load(
        model_name,
        tokenizer_config=tokenizer_config,
        model_config=model_config,
        adapter_path=adapter_path,
        lazy=lazy,
    )
    return MLXLM(model, tokenizer)
