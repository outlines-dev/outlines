import logging
import sys
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from outlines.models.tokenizer import Tokenizer


LOG_LEVEL_OFF = logging.CRITICAL + 100
logits_logger = logging.getLogger("logits_logger")
logits_logger.setLevel(LOG_LEVEL_OFF)
logits_logger.addHandler(logging.StreamHandler(sys.stderr))


def enable_logits_logging():
    logits_logger.setLevel(logging.DEBUG)


def log_logits(
    tokenizer: "Tokenizer",
    token_ids_group: torch.Tensor,
    unbiased_logits_group: torch.Tensor,
    biased_logits_group: torch.Tensor,
    next_token_ids: torch.Tensor,
    top_n: int = 8,
):
    """
    Intended for use with sequence generators to help debug low quality generations.

    Disabled unless you call enable_logits_logging()

    Simple utility function which logs
    - selected next token string
    - probabilities for the EOS token, and the top n tokens for biased and unbiased logits
    """
    # this function is expensive, skip it if logging isn't enabled
    if logits_logger.getEffectiveLevel() >= LOG_LEVEL_OFF:
        return

    def only(l):
        # Assert only one item in iterable, return sole item
        assert len(list(l)) == 1
        return l[0]

    def get_top_token_props_log_str(logits):
        """Get string representation of top_n token probs given logits"""
        # all token probs for the current batch
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # top candidate tokens probs
        top_probs, top_indices = torch.topk(probs, top_n)

        # ensure EOS is included
        if tokenizer.eos_token_id not in top_indices:
            eos_token_id_tensor = torch.tensor(
                [tokenizer.eos_token_id], device=top_indices.device
            )
            eos_prob_tensor = torch.tensor(
                [min(top_probs) - 1e-12], device=top_probs.device
            )
            top_indices = torch.cat((top_indices, eos_token_id_tensor))
            top_probs = torch.cat((top_probs, eos_prob_tensor))

        top_token_reprs = [
            "EOS"
            if idx == tokenizer.eos_token_id
            else repr(only(tokenizer.decode([idx])))
            for idx in top_indices
        ]
        return ", ".join(
            [
                f"{token_repr}: {prob.item():.3f}"
                for prob, token_repr in zip(top_probs, top_token_reprs)
            ]
        )

    # convert logits tensor of dimensions [n] to dimensions [1, n]
    if unbiased_logits_group.dim() == 1:
        token_ids_group = token_ids_group.unsqueeze(0)
        unbiased_logits_group = unbiased_logits_group.unsqueeze(0)
        biased_logits_group = biased_logits_group.unsqueeze(0)
        next_token_ids = next_token_ids.unsqueeze(0)

    logits_logger.info("--------Batch Logit Details--------")

    batch_size = unbiased_logits_group.size(0)
    for b in range(batch_size):
        unbiased_logits = unbiased_logits_group[b]
        biased_logits = biased_logits_group[b]
        next_token_id = next_token_ids[b]
        token_ids = token_ids_group[b]

        generation = tokenizer.decode(token_ids)
        next_token = only(tokenizer.decode(next_token_id))

        # Log the information for the current batch
        logits_logger.info(f"Generation: {generation}")
        logits_logger.info(f"\tSelected: {repr(next_token)} for batch_item={b}")
        logits_logger.info(
            f"\tTop Raw Tokens: {get_top_token_props_log_str(unbiased_logits)}"
        )
        logits_logger.info(
            f"\tTop Guided Tokens: {get_top_token_props_log_str(biased_logits)}"
        )
