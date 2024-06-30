"""Module that contains all the models integrated in outlines.

We group the models in submodules by provider instead of theme (completion, chat
completion, diffusers, etc.) and use routing functions everywhere else in the
codebase.

"""
from typing import Union

from .exllamav2 import ExLlamaV2Model, exl2
from .llamacpp import LlamaCpp, llamacpp
from .mamba import Mamba, mamba
from .mlxlm import MLXLM, mlxlm
from .openai import OpenAI, azure_openai, openai
from .transformers import Transformers, TransformerTokenizer, transformers
from .vllm import VLLM, vllm

LogitsGenerator = Union[Transformers, LlamaCpp, ExLlamaV2Model, Mamba, MLXLM, VLLM]
