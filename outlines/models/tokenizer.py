from typing import Dict, Hashable, List, Protocol, Set, Tuple, Union

import numpy as np
from numpy.typing import NDArray


class Tokenizer(Hashable, Protocol):
    eos_token: str
    eos_token_id: int
    pad_token_id: int
    vocabulary: Dict[str, int]
    special_tokens: Set[str]

    def encode(
        self, prompt: Union[str, List[str]]
    ) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Translate the input prompts into arrays of token ids and attention mask."""
        ...

    def decode(self, token_ids: NDArray[np.int64]) -> List[str]:
        """Translate an array of token ids to a string or list of strings."""
        ...

    def convert_token_to_string(self, token: str) -> str:
        """Convert a token to its equivalent string.

        This is for instance useful for BPE tokenizers where whitespaces are
        represented by the special characted `Ġ`. This prevents matching a raw
        token that includes `Ġ` with a string.
        """
        ...

    def apply_chat_template(
        self,
        prompt_or_conversation: Union[str, List[Dict[str, str]]],
    ) -> str:
        """Apply a chat template to a prompt or conversation."""
        ...
