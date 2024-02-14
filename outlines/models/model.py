from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, Union

from numpy.typing import NDArray
from torch import FloatTensor

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


class OutlinesModel(Protocol):
    tokenizer: "PreTrainedTokenizer"

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Union[FloatTensor, NDArray]:
        ...
