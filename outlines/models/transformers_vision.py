from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Union

from outlines.generate.api import GenerationParameters, SamplingParameters
from outlines.models import Transformers

if TYPE_CHECKING:
    from outlines.processors import OutlinesLogitsProcessor


class TransformersVision(Transformers):
    def __init__(self, model, tokenizer, processor):
        super().__init__(model, tokenizer)
        self.processor = processor

    def generate(  # type: ignore
        self,
        prompts: Union[str, List[str]],
        media: Union[List[Any], List[List[Any]]],
        generation_parameters: GenerationParameters,
        logits_processor: Optional["OutlinesLogitsProcessor"],
        sampling_parameters: SamplingParameters,
    ) -> Union[str, List[str], List[List[str]]]:
        """Generate text using `transformers`.

        Arguments
        ---------
        prompts
            A prompt or list of prompts.
        media
            A List[PIL.Image] or List[List[PIL.Image]]
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
        The generated text
        """
        inputs = self.processor(prompts, media, padding=True, return_tensors="pt").to(
            self.model.device
        )

        generation_kwargs = self._get_generation_kwargs(
            prompts,
            generation_parameters,
            logits_processor,
            sampling_parameters,
        )
        generated_ids = self._generate_output_seq(prompts, inputs, **generation_kwargs)

        # if single str input and single sample per input, convert to a 1D output
        if isinstance(prompts, str):
            # Should always be true until NotImplementedError above is fixed
            generated_ids = generated_ids.squeeze(0)

        return self._decode_generation(generated_ids)

    def stream(  # type: ignore
        self,
        prompts: Union[str, List[str]],
        media: Union[Any, List[Any]],  # TODO: docstring
        generation_parameters: GenerationParameters,
        logits_processor: Optional["OutlinesLogitsProcessor"],
        sampling_parameters: SamplingParameters,
    ) -> Iterator[Union[str, List[str]]]:
        raise NotImplementedError


def transformers_vision(
    model_name: str,
    model_class,
    device: Optional[str] = None,
    model_kwargs: dict = {},
    processor_kwargs: dict = {},
    tokenizer_class=None,
    processor_class=None,
):
    """Instantiate a model from the `transformers` library and its tokenizer.

    Parameters
    ----------
    model_name
        The name of the model as listed on Hugging Face's model page.
    model_class
        The `PreTrainedModel` class from transformers to use in initializing the vision model from `model_name`.
        https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel
    device
        The device(s) on which the model should be loaded. This overrides
        the `device_map` entry in `model_kwargs` when provided.
    model_kwargs
        A dictionary that contains the keyword arguments to pass to the
        `from_pretrained` method when loading the model.
    processor_kwargs
        A dictionary that contains the keyword arguments to pass to the
        `from_pretrained` method when loading the processor.

    Returns
    -------
    A `TransformersModel` model instance.

    """
    if processor_class is None or tokenizer_class is None:
        try:
            from transformers import AutoProcessor, AutoTokenizer
        except ImportError:
            raise ImportError(
                "The `transformers` library needs to be installed in order to use `transformers` models."
            )
    if processor_class is None:
        processor_class = AutoProcessor
    if tokenizer_class is None:
        tokenizer_class = AutoTokenizer

    if device is not None:
        model_kwargs["device_map"] = device

    model = model_class.from_pretrained(model_name, **model_kwargs)

    processor_kwargs.setdefault("padding_side", "left")
    processor_kwargs.setdefault("pad_token", "[PAD]")
    processor = processor_class.from_pretrained(model_name, **processor_kwargs)

    if tokenizer_class is None:
        if getattr(processor, "tokenizer", None):
            tokenizer = processor.tokenizer
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, **processor_kwargs)
    else:
        tokenizer = tokenizer_class.from_pretrained(model_name, **processor_kwargs)

    return TransformersVision(model, tokenizer, processor)
