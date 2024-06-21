from abc import abstractmethod
from typing import Any, List, Protocol, Union

import numpy as np
import torch
from numpy.typing import NDArray


def is_mlx_array(logits):
    try:
        import mlx.core as mx
    except ImportError:
        return False
    return isinstance(logits, mx.array)


class OutlinesLogitsProcessor(Protocol):
    """
    Base class for logits processors which normalizes types of logits:
    - ndarray (used by llama-cpp-python), converted to torch.Tensor
    - mlx.core.array (used by mlx-lm), converted to torch.Tensor
    - torch.Tensor (used by everything else)

    Normalization of types and conversion to torch.Tensor
    doesn't move memory, it just casts the type.

    Normalizing the types allows all logits processors inheriting from this class
    to implement a single method for all the business logit: `process_logits()`
    """

    @abstractmethod
    def process_logits(
        self, input_ids: List[List[int]], logits: torch.Tensor
    ) -> torch.Tensor:
        """
        input_ids and logits are always 2D tensors for handling a batch of sequences.

        - input_ids.shape[1] -> contains the sequence of int tokens
        - logits.shape[0] -> Dimension 1 contains one

        Important to keep in mind when designing universal logits processors
        - logits processors are only used once and never re-applied for a new sequence generator
        - Some models only pass output_ids, some models such as llamacpp and transformers prefix with input_ids
        - Some sampling methods, such as beam search, result in unstable sequence ordering in models like vLLM
        """
        pass

    @torch.no_grad()
    def __call__(
        self,
        input_ids: Union[NDArray[np.int64], List[int], torch.Tensor],
        logits: Union[NDArray[np.float32], torch.Tensor],
    ) -> Union[NDArray[np.int64], torch.Tensor]:
        """
        Apply logits processor

        Unify type
        - convert input_ids: either ndarray, List[int], or Tensor -> 2D tensor
        - convert logits: either ndarray, mlx array, Tensor -> 2D Tensor

        Call process_logits() to perform business logic
        """

        # ensure logits are torch Tensors
        torch_logits = self._to_torch(logits)

        assert torch_logits.shape[:-1] == self._to_torch(input_ids).shape[:-1]

        # ensure input_ids are List
        if not isinstance(input_ids, list):
            input_ids = input_ids.tolist()  # compatible with numpy, torch, and mlx

        # Guarantee passed as 2D Tensors, then covert back to original (1D or 2D) shape
        if len(torch_logits.shape) == 2:
            processed_logits = self.process_logits(input_ids, logits)
        elif len(torch_logits.shape) == 1:
            processed_logits = self.process_logits(
                [input_ids], torch_logits.unsqueeze(0)
            ).squeeze(0)

        # return logits as passed array type
        return self._from_torch(processed_logits, type(logits))

    @staticmethod
    def _to_torch(tensor_like: Any) -> torch.Tensor:
        """Convert various types to torch.Tensor."""
        if isinstance(tensor_like, torch.Tensor):
            return tensor_like

        elif isinstance(tensor_like, np.ndarray):
            return torch.from_numpy(tensor_like)

        elif isinstance(tensor_like, list):
            return torch.tensor(tensor_like)

        elif is_mlx_array(tensor_like):
            # mlx -> torch -> mlx conversion docs:
            # https://ml-explore.github.io/mlx/build/html/usage/numpy.html
            return torch.from_dlpack(tensor_like)

        else:
            raise TypeError(
                "LogitsProcessor must be called with either np.NDArray"
                ", torch.Tensor, list, or mlx.core.array typed logits"
            )

    @staticmethod
    def _from_torch(tensor: torch.Tensor, target_type: Any) -> Any:
        """Convert torch.Tensor to the specified target type."""
        if target_type == torch.Tensor:
            return tensor

        elif target_type == np.ndarray:
            return tensor.detach().numpy()

        elif target_type == list:
            return tensor.detach().tolist()

        elif target_type == "mlx_array":
            import mlx.core as mx

            # numpy doesn't support bfloat16, mlx doesn't support direct conversion from torch
            return mx.array(tensor.float().numpy())

        else:
            raise RuntimeError(
                "Failed to convert torch tensors back to original dtype. {tensor}"
                f"tensor={tensor}, target_type={target_type}"
            )
