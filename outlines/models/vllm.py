import concurrent.futures
import dataclasses
from typing import TYPE_CHECKING, List, Optional, Union

from outlines.generate.api import GenerationParameters, SamplingParameters
from outlines.integrations.utils import adapt_tokenizer

if TYPE_CHECKING:
    from vllm import LLM
    from vllm.sampling_params import SamplingParams


class VLLM:
    """Represents a vLLM model.

    We wrap models from model providing libraries in order to give all of
    them the same interface in Outlines and allow users to easily switch
    between providers. This class wraps the `vllm.LLM` class from the
    `vllm` library.

    """

    def __init__(self, model: "LLM", tokenizer):
        self.model = model
        self.lora_request = None

    def generate(
        self,
        prompts: Union[str, List[str]],
        generation_parameters: GenerationParameters,
        logits_processor,
        sampling_parameters: SamplingParameters,
        *,
        sampling_params: Optional["SamplingParams"] = None,
    ):
        """Generate text using vLLM.

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
        samplng_params
            An instance of `vllm.sampling_params.SamplingParams`. The values
            passed via this dataclass supersede the values of the parameters
            in `generation_parameters` and `sampling_parameters`. See the
            vLLM documentation for more details: https://docs.vllm.ai/en/latest/dev/sampling_params.html.

        Returns
        -------
        The generated text, of shape `(n_batch, n_samples)`. If there are only
        one batch and several samples, the list is of shape `(n_samples)`. If
        this is a batch with several sequences but only one sample the list is
        of shape `(n_batch)`. If there is only one sequence and one sample, a
        string is returned.

        """
        from vllm.sampling_params import SamplingParams

        if sampling_params is None:
            sampling_params = SamplingParams()

        max_tokens, stop_at, seed = dataclasses.astuple(generation_parameters)

        # We only update the values in `sampling_params` if they
        # are specified by the user when calling the generator.
        if max_tokens is not None:
            sampling_params.max_tokens = max_tokens
        else:
            sampling_params.max_tokens = None

        if stop_at is not None:
            if isinstance(stop_at, str):
                stop_at = [stop_at]
            sampling_params.stop = stop_at
        if seed is not None:
            sampling_params.seed = seed

        sampling_params.logits_processors = (
            [logits_processor] if logits_processor is not None else []
        )

        sampler, num_samples, top_p, top_k, temperature = dataclasses.astuple(
            sampling_parameters
        )

        # We only update the values in `sampling_params` that
        # were not specified by the user.
        if sampling_params.n == 1:
            sampling_params.n = num_samples
            sampling_params.best_of = num_samples
        if top_p is not None and sampling_params.top_p == 1.0:
            sampling_params.top_p = top_p
        if top_k is not None and sampling_params.top_k == -1:
            sampling_params.top_k = top_k
        if temperature is not None and sampling_params.temperature == 1.0:
            sampling_params.temperature = temperature
        if sampler == "beam_search":
            sampling_params.use_beam_search = True

        results = self.model.generate(
            prompts, sampling_params=sampling_params, lora_request=self.lora_request
        )
        results = [[sample.text for sample in batch.outputs] for batch in results]

        batch_size = len(results)
        sample_size = len(results[0])

        if batch_size == 1 and sample_size == 1:
            return results[0][0]
        elif batch_size == 1:
            return results[0]
        elif sample_size == 1:
            return [batch[0] for batch in results]

        return results

    def stream(self, *args, **kwargs):
        """Return a text generator.

        Streaming is not yet available for `vllm.LLM`.

        TODO: Implement the streaming functionality ourselves.

        """
        raise NotImplementedError(
            "Streaming is not available for the vLLM integration."
        )

    def load_lora(self, adapter_path: Optional[str]):
        from vllm.lora.request import LoRARequest

        if adapter_path is None:
            self.lora_request = None
        else:
            self.lora_request = LoRARequest(adapter_path, 1, adapter_path)


def load_model(model_name, **vllm_model_params):
    """Load the model in GPU memory."""
    from vllm import LLM

    model = LLM(model_name, **vllm_model_params)
    return model


def load_and_convert_tokenizer(model_name):
    """Convert vocabulary types and JIT-compile function."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = convert_tokenizer(tokenizer)
    jit_compile_numba(tokenizer.numba_vocabulary)

    return tokenizer


def convert_tokenizer(tokenizer):
    from outlines.fsm.regex import reduced_vocabulary

    tokenizer = adapt_tokenizer(tokenizer)
    vocabulary, _ = reduced_vocabulary(tokenizer)
    tokenizer.vocabulary_numba = vocabulary

    return tokenizer


def jit_compile_numba(vocabulary):
    pass


def vllm(model_or_model_name: Union[str, "LLM"], **vllm_model_params):
    """Load a vLLM model.

    Arguments
    ---------
    model_name
        The name of the model to load from the HuggingFace hub.
    vllm_model_params
        vLLM-specific model parameters. See the vLLM code for the full list:
        https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py

    """
    from vllm import LLM

    if isinstance(model_or_model_name, LLM):
        model = model_or_model_name
        tokenizer = convert_tokenizer(model.tokenizer)
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            model_name = model_or_model_name
            future_model = executor.submit(load_model, model_name, **vllm_model_params)
            future_tokenizer = executor.submit(load_and_convert_tokenizer, model_name)

            tokenizer = future_tokenizer.result()
            model = future_model.result()

    return VLLM(model, tokenizer)
