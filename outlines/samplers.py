import math
from typing import TYPE_CHECKING, Callable, Optional, Protocol, Tuple

if TYPE_CHECKING:
    import torch


class Sampler(Protocol):
    samples: int

    def __call__(
        self,
        next_token_logits: "torch.DoubleTensor",
        sequence_weights: "torch.DoubleTensor",
        rng: "torch.Generator",
    ) -> "torch.DoubleTensor":
        ...


class GreedySampler:
    """Greedy Sampling algorithm.

    Greedy sampling consists in choosing the token with the largest
    likelihood at every step.

    We don't allow more than one sample. We could attribute this a meaning, for
    instance the k-th sample represents the k-th most likely token. In which
    case it would be equivalent to beam search without the sequence weights.

    Attributes
    ----------
    samples
        The number of samples taken for each input sequence.

    """

    def __init__(self):
        self.samples = 1

    def __call__(
        self,
        next_token_logits: "torch.DoubleTensor",
        sequence_weights: "torch.DoubleTensor",
        _,
    ) -> "torch.DoubleTensor":
        """Call the greedy sampler.

        Parameters
        ----------
        next_token_logits
            A tensor of shape ``(n_seqs, vocab_size,)`` that represents the
            probability distribution of the next token over the vocabulary.
        sequence_weights
            A tensor of shape ``(n_seqs,)`` that represents the cumulative
            weight of each sequence.
        rng
            A random number generator.

        Returns
        -------
        A tuple with an array that contains the ids of the sampled tokens of
        shape ``(n_seqs, 1)``, an array that contains the ancestors of each
        sampled id of shape ``(n_seqs,)`` and an array that contains the updated
        cumulative weights of each sequence of shape ``(n_seqs,)``.

        """
        import torch

        logprobs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
        next_token_ids = torch.argmax(logprobs, dim=-1, keepdim=True)

        ancestors = torch.arange(
            next_token_logits.shape[0], device=next_token_logits.device
        )
        weights = sequence_weights + torch.gather(logprobs, 1, next_token_ids).squeeze()

        return next_token_ids, ancestors, weights


greedy = GreedySampler


class MultinomialSampler:
    """Multinomial sampling algorithm.

    Multinomial sampling consists in randomly sampling the next token assuming
    its distribution is a Categorical distribution parametrized by the
    next-token logits.


    Attributes
    ----------
    samples
        The number of samples taken for each input sequence.

    """

    def __init__(
        self,
        samples: int = 1,
        *,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
    ):
        self.samples = samples
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

        self.logits_processors = []
        if top_k is not None:
            self.logits_processors.append(keep_top_k_logits(top_k))
        elif top_p is not None:
            self.logits_processors.append(keep_top_p_logits(top_p))

        if temperature is not None:
            self.logits_processors.append(rescale_logits(temperature))

    def __call__(
        self,
        next_token_logits: "torch.DoubleTensor",
        sequence_weights: "torch.DoubleTensor",
        rng: "torch.Generator",
    ) -> Tuple["torch.DoubleTensor", "torch.DoubleTensor", "torch.DoubleTensor"]:
        """Call the multinomial sampler.

        Parameters
        ----------
        next_token_logits
            A tensor of shape ``(n_seqs, vocab_size,)`` that represents the
            probability distribution of the next token over the vocabulary.
        sequence_weights
            A tensor of shape ``(n_seqs,)`` that represents the cumulative
            weight of each sequence.
        rng
            A random number generator.

        Returns
        -------
        A tuple with an array that contains the ids of the sampled tokens of
        shape ``(n_seqs, 1)``, an array that contains the ancestors of each
        sampled id of shape ``(n_seqs,)`` and an array that contains the updated
        cumulative weights of each sequence of shape ``(n_seqs,)``.

        """
        import torch

        altered_next_token_logits = next_token_logits
        for logit_processor in self.logits_processors:
            altered_next_token_logits = logit_processor(next_token_logits)

        probs = torch.nn.functional.softmax(altered_next_token_logits, dim=-1)
        next_token_ids = torch.multinomial(probs, num_samples=1, generator=rng)

        logprobs = torch.nn.functional.log_softmax(altered_next_token_logits, dim=-1)
        ancestors = torch.arange(
            altered_next_token_logits.shape[0], device=next_token_logits.device
        )
        weights = sequence_weights + torch.gather(logprobs, 1, next_token_ids).squeeze()

        return next_token_ids, ancestors, weights


multinomial = MultinomialSampler


def keep_top_k_logits(k: int) -> Callable[["torch.Tensor"], "torch.Tensor"]:
    """Build a function that masks logits values smaller than the top `k` ones.

    Parameters
    ----------
    k
        The ranking below which logit values are replaced by `-math.inf`.

    """
    import torch

    if not isinstance(k, int) or k < 1:
        raise ValueError(f"`k` must be a strictly positive integers, got {k} instead.")

    def logits_processor(logits: torch.Tensor) -> torch.Tensor:
        num_to_keep = min(k, logits.size(-1))
        mask_idx = logits < torch.topk(logits, num_to_keep)[0][..., -1, None]
        return logits.masked_fill(mask_idx, -math.inf)

    return logits_processor


def keep_top_p_logits(p: float) -> Callable[["torch.Tensor"], "torch.Tensor"]:
    """Build a function that masks the lowest probability tokens whose
    cumulative probability is below a certain threshold.

    Parameters
    ----------
    p
        The value of the threshold. We keep the highest probability tokens whose
        cumulative distribution is greater than or equal to `p` and mask the
        others. Its value must be between 0 (excluded) and 1 (included).

    """
    import torch

    if p <= 0.0 or p > 1.0:
        raise ValueError(
            f"`p` must be a floating point number between 0 (excluded) and 1 (included), got {p} instead."
        )

    def logits_processor(logits: torch.Tensor) -> torch.Tensor:
        sorted_logits, sorted_idx = torch.sort(logits, descending=False)
        cumulative_probabilties = torch.nn.functional.softmax(
            sorted_logits, dim=-1
        ).cumsum(dim=-1)

        sorted_masked_idx = cumulative_probabilties <= (1 - p)
        mask_idx = torch.scatter(sorted_masked_idx, 1, sorted_idx, sorted_masked_idx)
        return logits.masked_fill(mask_idx, -math.inf)

    return logits_processor


def rescale_logits(temperature: float) -> Callable[["torch.Tensor"], "torch.Tensor"]:
    """Build a function that rescales the token probabilities exponentially.

    Parameters
    ----------
    temperature
        The value by which we rescale the logits.

    """

    if not isinstance(temperature, float) or temperature < 0.0:
        raise ValueError(
            f"`temperature` must be a strictly positive floating point number, got {temperature} instead."
        )
    elif temperature == 0.0:
        raise ValueError(
            "Please use the greedy sampler instead of setting the temperature to 0."
        )

    def logits_processor(logits: "torch.Tensor") -> "torch.Tensor":
        return logits / temperature

    return logits_processor


class BeamSearchSampler:
    """Beam Search sampling algorithm.

    Attributes
    ----------
    samples
        The number of samples taken for each input sequence.

    """

    def __init__(self, beams: int = 1):
        self.samples = beams

    def __call__(
        self,
        next_token_logits: "torch.DoubleTensor",
        sequence_weights: "torch.DoubleTensor",
        rng: "torch.Generator",
    ) -> Tuple["torch.DoubleTensor", "torch.DoubleTensor", "torch.DoubleTensor"]:
        """Call the beam search sampler.

        Parameters
        ----------
        next_token_logits
            A tensor of shape ``(n_seqs, vocab_size,)`` that represents the
            probability distribution of the next token over the vocabulary.
        sequence_weights
            A tensor of shape ``(n_seqs,)`` that represents the cumulative
            weight of each sequence.
        rng
            A random number generator.

        Returns
        -------
        A tuple with an array that contains the ids of the sampled tokens of
        shape ``(n_seqs, 1)``, an array that contains the ancestors of each
        sampled id of shape ``(n_seqs,)`` and an array that contains the updated
        cumulative weights of each sequence of shape ``(n_seqs,)``.

        """
        import torch

        logprobs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
        weights = logprobs + sequence_weights.unsqueeze(1).expand_as(next_token_logits)

        # Flatten scores to (n_batch, n_samples * vocab_size)
        # and find the top-k weights for each batch.
        batch_size = next_token_logits.shape[0] // self.samples
        vocab_size = next_token_logits.shape[-1]
        weights = weights.view(batch_size, self.samples * vocab_size)

        # If the weights are all equal to 0 we are at the beginning of the search
        # and thus only need to sample from one set of token logits for each
        # batch.
        if torch.all(sequence_weights == 0):
            weights = weights[:, :vocab_size]

        weights, indices = torch.topk(
            weights, self.samples, dim=1, largest=True, sorted=True
        )

        ancestors = torch.div(indices, vocab_size, rounding_mode="floor")
        next_token_ids = indices % vocab_size

        # Re-shape the weights, next_token_ids and ancestors to (n_batch * n_samples, 1)
        first_batch_idx = torch.arange(
            0, batch_size * self.samples, self.samples, device=next_token_logits.device
        ).unsqueeze(1)
        ancestors = ancestors + first_batch_idx

        ancestors = ancestors.view(self.samples * batch_size)
        weights = weights.view(self.samples * batch_size)
        next_token_ids = next_token_ids.view(self.samples * batch_size, 1)

        return next_token_ids, ancestors, weights


beam_search = BeamSearchSampler
