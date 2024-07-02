import numpy as np
import torch

import outlines.models as models
from outlines.processors import OutlinesLogitsProcessor, RegexLogitsProcessor

try:
    import mlx.core as mx
except ImportError:
    pass


def is_mlx_lm_allowed():
    try:
        import mlx.core as mx
    except ImportError:
        return False
    return mx.metal.is_available()


def get_mock_processor_inputs(array_library, num_tokens=30000):
    """
    logits: (4, 30,000 ) dtype=float
    input_ids shape: (4, 2048) dtype=int
    """
    if array_library == "torch":
        logits = torch.rand((4, num_tokens), dtype=torch.float)
        input_ids = torch.randint(
            low=0, high=num_tokens, size=(4, 2048), dtype=torch.int
        )
    elif array_library == "torch_cuda":
        logits = torch.rand((4, num_tokens), dtype=torch.float, device="cuda")
        input_ids = torch.randint(
            low=0, high=num_tokens, size=(4, 2048), dtype=torch.int, device="cuda"
        )
    elif array_library == "numpy":
        logits = np.random.rand(4, num_tokens).astype(np.float32)
        input_ids = np.random.randint(low=0, high=num_tokens, size=(4, 2048))
    elif array_library == "mlx":
        logits = mx.random.uniform(
            low=-1e9, high=1e9, shape=(4, num_tokens), dtype=mx.float32
        )
        input_ids = mx.random.randint(
            low=0, high=num_tokens, shape=(4, 2048), dtype=mx.int32
        )
    else:
        raise ValueError

    return logits, input_ids


class HalvingLogitsProcessor(OutlinesLogitsProcessor):
    """Simply halve the passed logits"""

    def process_logits(self, input_ids, logits):
        return logits / 2


class LogitsProcessorPassthroughBenchmark:
    """
    Benchmark the time it takes to convert between array frameworks
    This should be on the order of microseconds
    """

    params = ["torch", "numpy"]
    if is_mlx_lm_allowed():
        params += ["mlx"]
    if torch.cuda.is_available():
        params += ["torch_cuda"]

    def setup(self, array_library):
        self.logits_processor = HalvingLogitsProcessor()

        self.logits, self.input_ids = get_mock_processor_inputs(array_library)

    def time_passthrough(self, *params):
        self.logits_processor(self.input_ids, self.logits)


class LogitsProcessorStructuredBenchmark:
    """
    Benchmark structured generation mask application for single decoder pass
    """

    array_libraries = ["torch", "numpy"]
    if is_mlx_lm_allowed():
        array_libraries += ["mlx"]
    # PR TODO
    if torch.cuda.is_available():
        array_libraries += ["torch_cuda"]

    # accept very many or very few tokens, respectively
    patterns = [r"[^Z]*", "Z*"]

    params = [array_libraries, patterns]
    param_names = ["array_library, pattern"]

    def setup(self, array_library, pattern):
        tokenizer = models.transformers("facebook/opt-125m", device="cpu").tokenizer

        self.logits_processor = RegexLogitsProcessor(pattern, tokenizer)

        self.logits, self.input_ids = get_mock_processor_inputs(
            array_library, len(tokenizer.vocabulary)
        )

    def time_structured_generation(self, array_library, pattern):
        self.logits_processor(self.input_ids, self.logits)
