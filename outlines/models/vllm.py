import dataclasses
from typing import TYPE_CHECKING, Optional

from outlines.generate.api import GenerationParameters, SamplingParameters

if TYPE_CHECKING:
    from vllm import LLM
    from vllm.sampling_params import SamplingParams


class VLLM:
    def __init__(self, model: "LLM"):
        self.model = model
        self.lora_request = None

    def generate(
        self,
        generation_parameters: GenerationParameters,
        logits_procesor,
        sampling_parameters: SamplingParameters,
        sampling_params: Optional["SamplingParams"] = None,
    ):
        from vllm.sampling_params import SamplingParams

        if sampling_params is None:
            sampling_params = SamplingParams()

        prompts, max_tokens, stop_at, seed = dataclasses.astuple(generation_parameters)

        # We only update the values in `sampling_params` if they
        # are specified by the user when calling the generator.
        if max_tokens is not None:
            sampling_params.max_tokens = max_tokens
        if stop_at is not None:
            sampling_params.stop = stop_at
        if seed is not None:
            sampling_params.seed = seed

        sampling_params.logits_processors = (
            [logits_procesor] if logits_procesor is not None else []
        )

        sampler, num_samples, top_p, top_k, temperature = dataclasses.astuple(sampling_parameters)

        # We only update the values in `sampling_params` that
        # were not specified by the user.
        if sampling_params.n == 1:
            sampling_params.n = num_samples
            sampling_params.best_of = num_samples
        if top_p is not None and sampling_params.top_p == 1.:
            sampling_params.top_p = top_p
        if top_k is not None and sampling_params.top_k==-1:
            sampling_params.top_k = top_k
        if temperature is not None and sampling_params.temperature == 1.:
            sampling_params.temperature = temperature
        if sampler == "beam_search":
            sampling_params.use_beam_search = True

        results = self.model.generate(prompts, sampling_params=sampling_params, lora_request=self.lora_request)
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
        raise NotImplementedError(
            "Streaming is not available for the vLLM integration."
        )

    def load_lora(self, adapter_path: Optional[str]):
        from vllm.lora.request import LoRARequest

        if adapter_path is None:
            self.lora_request = None
        else:
            self.lora_request = LoRARequest(adapter_path, 1, adapter_path)


def vllm(model_name: str, **vllm_model_params):
    from vllm import LLM

    model = LLM(model_name, **vllm_model_params)

    return VLLM(model)
