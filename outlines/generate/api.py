import datetime
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Union

from outlines.generate.generator import sequence_generator
from outlines.samplers import BeamSearchSampler, GreedySampler, MultinomialSampler

if TYPE_CHECKING:
    import torch

FormattedOutput = Union[
    str, int, float, bool, datetime.date, datetime.time, datetime.datetime
]


class SequenceGenerator:
    def __init__(
        self,
        fsm,
        model,
        sampler,
        device,
    ):
        self.fsm = fsm
        self.model = model
        self.sampler = sampler
        self.tokenizer = model.tokenizer
        self.device = device
        self.num_samples = sampler.samples

    def get_generated_token_ids(
        self,
        prompt_token_ids: "torch.Tensor",
        token_ids: "torch.Tensor",
    ) -> List["torch.Tensor"]:
        """Get the tokens generated so far.

        Parameters
        ----------
        prompt_token_ids
            Tensor that contains the token ids of the sequences' prompts.
        token_ids
            The generated token ids.

        Returns
        -------
        A tensor that contains the token ids that have been generated so far.

        """
        prompt_lengths = [len(prompt) for prompt in prompt_token_ids]
        token_ids = [
            cur_token_ids[length:]
            for cur_token_ids, length in zip(token_ids, prompt_lengths)
        ]

        return token_ids

    def is_stop_sequence_found(
        self, generated_sequences: List[str], stop_sequences: List[str]
    ) -> bool:
        """Determine whether one of the stop sequences has been generated.

        Parameters
        ----------
        generated_sequences
            The list of sequences generated so far.
        stop_sequences
            The list that contains the sequence which stop the generation when
            found.

        Returns
        -------
        True if at least one of the stop sequences has been found in each generated
        sequence.

        """
        return all(
            [
                any([seq in generated for seq in stop_sequences])
                for generated in generated_sequences
            ]
        )

    def strip_stop_sequences(
        self, sequence: str, stop_sequences: Optional[List[str]]
    ) -> str:
        """Remove the stop sequences from the generated sequences.

        Parameters
        ----------
        sequence
            One of the generated sequences.
        stop_sequences
            The list that contains the sequence which stop the generation when
            found.

        """
        if stop_sequences:
            match_indexes = [sequence.find(seq) for seq in stop_sequences]
            if any([index != -1 for index in match_indexes]):
                # select the stop_sequence that is found first in the sequence
                min_match_index_value = min([i for i in match_indexes if i != -1])
                min_match_index_pos = match_indexes.index(min_match_index_value)
                sequence = sequence[
                    : match_indexes[min_match_index_pos]
                    + len(stop_sequences[min_match_index_pos])
                ]

        return sequence

    def format_sequence(self, sequence: str) -> FormattedOutput:
        """Translate the generated sequence to another type.

        This method is for instance overridden when generating JSON to either
        return a dictionnary or a Pydantic model.

        Parameters
        ----------
        sequence
            A generated sequences.

        Returns
        -------
        The formatted sequence.

        """
        return sequence
    
    def __call__(
            self,
            prompts: Optional[Union[str, List[str]]],
            max_tokens: Optional[int] = None,
            stop_at: Optional[Union[str, List[str]]] = None,
            rng: Optional["torch.Generator"] = None,
            include_prompt: bool = False,
        ) -> Union[FormattedOutput, List[FormattedOutput], List[List[FormattedOutput]]]:
        """Generate the full text sequence.

        Since `SequenceGenerator.stream` calls the tokenizer at every step this
        method loops over the generator returned by `sequence_generator` itself
        so the tokenizer is called only once after all token ids have been
        generated.

        Parameters
        ----------
        prompts
            A string or list of strings that are passed to the model before
            generating the first token.
        max_tokens
            An integer representing maximum number of tokens that will be generated
            (per prompt)
        stop_at
            A string or list of strings at which the text generated will stop
        rng
            The random number generator. Defaults to a non-seeded `torch.Generator`
            instance.
        include prompt
            Whether the prompt itself should be included in the generation pattern. Also
            has the effect of the prompt being in the output. Useful if the input is
            the start of a json dict, for example, and you want to generate the rest.

        Returns
        -------
        The generation(s), potentially cast to another type.
        """
        import torch

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(stop_at, str):
            stop_at = [stop_at]

        stop_sequences = stop_at
        num_samples = self.num_samples

        if rng is None:
            rng = torch.Generator(device=self.device)
            rng.seed()

        prompt_token_ids, attention_masks = self.tokenizer.encode(prompts)

        prompt_token_ids = prompt_token_ids.to(self.device)
        attention_masks = attention_masks.to(self.device)

        # To draw multiple samples we repeat the prompt as many times
        # as there are samples. We copy the FSMs and initialize the
        # FSM states.
        num_samples = self.num_samples
        batch_size = len(prompts)

        prompt_token_ids = torch.repeat_interleave(prompt_token_ids, num_samples, dim=0)
        attention_masks = torch.repeat_interleave(attention_masks, num_samples, dim=0)
        fsm_states = [0 for _ in range(batch_size * num_samples)]
        fsms = [self.fsm.copy() for _ in range(batch_size * num_samples)]
        weights = torch.zeros(
            (batch_size * num_samples), dtype=torch.float, device=self.device
        )

        if include_prompt:
            final_fsm_states = []
            for fsm, state, prompt_token_id_seq, attention_mask_seq in mit.zip_equal(
                fsms, fsm_states, prompt_token_ids, attention_masks):
                assert state == 0, f"Expected state to be 0, got {state}"
                assert isinstance(state, int), f"Expected state to be an int, got {state}"
                for token_id, attention_mask in zip(prompt_token_id_seq, attention_mask_seq):
                    state = fsm.get_next_state(state=state, token_id=token_id.item())
                    if state < 0:
                        raise ValueError(f"Invalid state {state}")
                final_fsm_states.append(state)

            start_len = len(fsm_states)
            fsm_states = final_fsm_states
            assert len(fsm_states) == start_len, f"{len(fsm_states) = }, {start_len = }"

        states = sequence_generator(
            self.model,
            self.sampler,
            fsms,
            prompt_token_ids,
            weights,
            attention_masks,
            fsm_states,
            rng=rng,
        )

        while True:
            try:
                last_state = next(states)
                if max_tokens or stop_sequences:
                    token_ids = last_state.token_ids
                    generated_token_ids = self.get_generated_token_ids(
                        prompt_token_ids, token_ids
                    )

                    if max_tokens and len(generated_token_ids[0]) >= max_tokens:
                        break
                    
                    if stop_sequences and self.is_stop_sequence_found(
                        self.tokenizer.decode(generated_token_ids), stop_sequences
                    ):
                        break
            except StopIteration:
                break

        token_ids = last_state.token_ids
        

        # The generated token ids are the same as the token ids if we include the prompt.
        # To retain the ability to parse to json, we need to keep the prompt token ids.        
        if include_prompt:
            generated_token_ids = token_ids
        else:
            generated_token_ids = self.get_generated_token_ids(prompt_token_ids, token_ids)

        generated = self.tokenizer.decode(generated_token_ids)
        stripped = [
            self.strip_stop_sequences(sequence, stop_sequences)
            for sequence in generated
        ]
        formatted = [self.format_sequence(sequence) for sequence in stripped]

        # We reshape the output to (batch_size, sample_size)
        output: List[List[FormattedOutput]] = list()
        for i in range(0, batch_size * num_samples, num_samples):
            output.append(formatted[i : i + num_samples])

        # We remove leading dimensions for the output
        if batch_size == 1 and num_samples == 1:
            return output[0][0]
        elif batch_size == 1:
            return output[0]
        elif num_samples == 1:
            return [samples[0] for samples in output]
        else:
            return output


    def stream(
        self,
        prompts: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
        rng: Optional["torch.Generator"] = None,
    ) -> Iterator[Union[List[str], str, List[List[str]]]]:
        """Generate the text sequence one token at a time.

        Since `Tokenizer.decode` strips the whitespaces from the tokens we have no
        choice but to decode the generated token ids at each step and compare the
        current decoded strings to the previously decoded strings.

        Parameters
        ----------
        prompts
            A string or list of strings that are passed to the model before
            generating the first token.
        max_tokens
            An integer representing maximum number of tokens that will be generated
            (per prompt)
        stop_at
            A string or list of strings at which the text generated will stop
        rng
            The random number generator. Defaults to a non-seeded `torch.Generator`
            instance.

        Returns
        -------
        A string or list of strings that contain the generated text.

        """
        import torch

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(stop_at, str):
            stop_at = [stop_at]

        stop_sequences = stop_at
        num_samples = self.num_samples

        prompt_token_ids, attention_masks = self.tokenizer.encode(prompts)
        prompt_token_ids = prompt_token_ids.to(self.device)
        attention_masks = attention_masks.to(prompt_token_ids.device)

        # To draw multiple samples we repeat the prompt as many times
        # as there are samples. We copy the FSMs and initialize the
        # FSM states.
        num_samples = self.num_samples
        batch_size = len(prompts)

        prompt_token_ids = torch.repeat_interleave(prompt_token_ids, num_samples, dim=0)
        attention_masks = torch.repeat_interleave(attention_masks, num_samples, dim=0)
        fsm_states = [0 for _ in range(batch_size * num_samples)]
        fsms = [self.fsm.copy() for _ in range(batch_size * num_samples)]
        weights = torch.zeros(
            (batch_size * num_samples),
            dtype=torch.float,
            device=prompt_token_ids.device,
        )

        if rng is None:
            rng = torch.Generator(device=prompt_token_ids.device)
            rng.seed()

        states = sequence_generator(
            self.model,
            self.sampler,
            fsms,
            prompt_token_ids,
            weights,
            attention_masks,
            fsm_states,
            rng=rng,
        )

        def token_generator() -> Iterator[Union[List[str], str, List[List[str]]]]:
            previously_generated_sequences = [
                "" for _ in range(batch_size)
            ] * num_samples
            num_generated = 0
            is_stop_at_reached = [False for _ in range(batch_size)] * num_samples
            while True:
                if (max_tokens and num_generated >= max_tokens) or all(
                    is_stop_at_reached
                ):
                    return
                try:
                    sequence = next(states)
                    num_generated += 1
                except StopIteration:
                    return
                generated_token_ids = sequence.token_ids[:, -num_generated:]
                generated_sequences = self.tokenizer.decode(generated_token_ids)
                if stop_sequences:
                    is_stop_at_reached = [
                        stop
                        or self.is_stop_sequence_found(
                            [generated_sequence], stop_sequences
                        )
                        for generated_sequence, stop in zip(
                            generated_sequences, is_stop_at_reached
                        )
                    ]

                    generated_sequences = [
                        self.format_sequence(
                            self.strip_stop_sequences(sequence, stop_sequences)
                        )
                        if stop
                        else sequence
                        for sequence, stop in zip(
                            generated_sequences, is_stop_at_reached
                        )
                    ]
                next_tokens = [
                    token[len(sequence) :]
                    for token, sequence, stop in zip(
                        generated_sequences,
                        previously_generated_sequences,
                        is_stop_at_reached,
                    )
                ]
                previously_generated_sequences = generated_sequences
                # We reshape the output to (batch_size, sample_size)
                output: List[List[str]] = list()
                for i in range(0, batch_size * num_samples, num_samples):
                    output.append(next_tokens[i : i + num_samples])

                # We remove leading dimensions for the output
                if batch_size == 1 and num_samples == 1:
                    yield output[0][0]
                elif batch_size == 1:
                    yield output[0]
                elif num_samples == 1:
                    yield [samples[0] for samples in output]
                else:
                    yield output

        return token_generator()


@dataclass(frozen=True)
class GenerationParameters:
    """Generation parameters used in Outlines' public API."""

    max_tokens: Optional[int]
    stop_at: Optional[Union[str, List[str]]]
    seed: Optional[int]


@dataclass(frozen=True)
class SamplingParameters:
    """Sampling parameters available in Outlines."""

    sampler: str
    num_samples: int = 1
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    temperature: Optional[float] = None


class SequenceGeneratorAdapter:
    """Class used to unify the interface to the model providers'
    generation functions.

    Attributes
    ----------
    model
        The wrapped model.
    logits_processor
        The logits processor to use to generate text.
    sampler
        The sampler to use to generate text.

    """

    def __init__(self, model, logits_processor, sampler):
        self.model = model
        self.logits_processor = logits_processor

        if isinstance(sampler, MultinomialSampler):
            self.sampling_params = SamplingParameters(
                "multinomial",
                sampler.samples,
                sampler.top_p,
                sampler.top_k,
                sampler.temperature,
            )
        elif isinstance(sampler, GreedySampler):
            self.sampling_params = SamplingParameters(
                "greedy", sampler.samples, None, None, 0.0
            )
        elif isinstance(sampler, BeamSearchSampler):
            self.sampling_params = SamplingParameters(
                "beam_search", sampler.samples, None, None, 1.0
            )

    def prepare_generation_parameters(
        self,
        max_tokens: Optional[int],
        stop_at: Optional[Union[str, List[str]]],
        seed: Optional[int],
    ):
        if isinstance(stop_at, str):
            stop_at = [stop_at]

        generation_params = GenerationParameters(
            max_tokens,
            stop_at,
            seed,
        )

        return generation_params

    def format_sequence(self, sequence: str) -> FormattedOutput:
        """Translate the generated sequence to another type.

        This method is for instance overridden when generating JSON to either
        return a dictionnary or a Pydantic model.

        Parameters
        ----------
        sequence
            A generated sequences.

        Returns
        -------
        The formatted sequence.

        """
        return sequence

    def _format(self, sequences):
        """Apply formatting to every string in a completion."""
        if isinstance(sequences, list):
            return [self._format(sequence) for sequence in sequences]
        else:
            return self.format_sequence(sequences)

    def __call__(
        self,
        prompts: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        **model_specific_params,
    ):
        """Generate text from a prompt of list of prompts."""

        generation_params = self.prepare_generation_parameters(
            max_tokens, stop_at, seed
        )

        completions = self.model.generate(
            prompts,
            generation_params,
            copy(self.logits_processor),
            self.sampling_params,
            **model_specific_params,
        )

        return self._format(completions)

    def stream(
        self,
        prompts: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        **model_specific_params,
    ):
        """Return a text generator from a prompt or a list of prompts."""
        generation_params = self.prepare_generation_parameters(
            max_tokens, stop_at, seed
        )
        return self.model.stream(
            prompts,
            generation_params,
            copy(self.logits_processor),
            self.sampling_params,
            **model_specific_params,
        )


class VisionSequenceGeneratorAdapter(SequenceGeneratorAdapter):
    def __call__(  # type: ignore
        self,
        prompts: Union[str, List[str]],
        media: Union[str, Any],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        **model_specific_params,
    ):
        """
        Generate text from a prompt of list of prompts.

        Media: A URI to construct media or media object itself. Used as AutoProcessor argument.
        """
        prompts, media = self._validate_prompt_media_types(prompts, media)

        generation_params = self.prepare_generation_parameters(
            max_tokens, stop_at, seed
        )

        completions = self.model.generate(
            prompts,
            media,
            generation_params,
            copy(self.logits_processor),
            self.sampling_params,
            **model_specific_params,
        )

        return self._format(completions)

    def stream(  # type: ignore
        self,
        prompts: Union[str, List[str]],
        media: List[Union[str, Any, List[Union[str, Any]]]],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        **model_specific_params,
    ):
        """Return a text generator from a prompt or a list of prompts."""
        prompts, media = self._validate_prompt_media_types(prompts, media)
        generation_params = self.prepare_generation_parameters(
            max_tokens, stop_at, seed
        )
        return self.model.stream(
            prompts,
            media,
            generation_params,
            copy(self.logits_processor),
            self.sampling_params,
            **model_specific_params,
        )

    @classmethod
    def _validate_prompt_media_types(
        cls,
        prompts: Union[str, List[str]],
        media: Union[str, Any, List[Union[str, Any]]],
    ) -> Union[Any, List[Any]]:
        """
        Prepare media as PIL.Image and ensure for every prompt str there is one List[PIL.Image]
        """

        def valid_types(prompts, media):
            from PIL import Image  # type: ignore

            if isinstance(prompts, list):
                if not isinstance(media, list) or len(prompts) != len(media):
                    return False
                for subprompt, submedia in zip(prompts, media):
                    if not isinstance(subprompt, str) or not all(
                        isinstance(m, Image.Image) for m in submedia
                    ):
                        return False
            elif isinstance(prompts, str):
                if not all(isinstance(m, Image.Image) for m in media):
                    return False
            return True

        if not valid_types(prompts, media):
            raise TypeError(
                "Expected (prompts, media) to be of type "
                "(str, List[Image])), or (List[str], List[List[Image]]) "
                f"instead got prompts={prompts}, media={media}"
            )

        return prompts, media
