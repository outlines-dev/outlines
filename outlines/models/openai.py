"""Integration with OpenAI's API."""
import functools
import warnings
from dataclasses import asdict, dataclass, field, replace
from itertools import zip_longest
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from outlines.base import vectorize
from outlines.caching import cache

__all__ = ["OpenAI", "openai", "azure_openai"]


@dataclass(frozen=True)
class OpenAIConfig:
    """Represents the parameters of the OpenAI API.

    The information was last fetched on 2023/11/20. We document below the
    properties that are specific to the OpenAI API. Not all these properties are
    supported by Outlines.

    Properties
    ----------
    model
        The name of the model. Available models can be found on OpenAI's website.
    frequence_penalty
        Number between 2.0 and -2.0. Positive values penalize new tokens based on
        their existing frequency in the text,
    logit_bias
        Modifies the likelihood of specified tokens to appear in the completion.
        Number between -100 (forbid) and +100 (only allows).
    n
        The number of completions to return for each prompt.
    presence_penalty
        Similar to frequency penalty.
    response_format
        Specifies the format the model must output. `{"type": "json_object"}`
        enables JSON mode.
    seed
        Two completions with the same `seed` value should return the same
        completion. This is however not guaranteed.
    stop
        Up to 4 words where the API will stop the completion.
    temperature
        Number between 0 and 2. Higher values make the output more random, while
        lower values make it more deterministic.
    top_p
        Number between 0 and 1. Parameter for nucleus sampling.
    tools
        A list of tools the model may call.
    tool_choice
        Specifies which tool the model should use.
    user
        A unique identifier for the end-user.

    """

    model: str = ""
    frequency_penalty: float = 0
    logit_bias: Dict[int, int] = field(default_factory=dict)
    max_tokens: Optional[int] = None
    n: int = 1
    presence_penalty: float = 0
    response_format: Optional[Dict[str, str]] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    temperature: float = 1.0
    top_p: int = 1
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[Union[str, Dict]] = None
    user: str = field(default_factory=str)


class OpenAI:
    """An object that represents the OpenAI API."""

    def __init__(
        self,
        client,
        config,
        tokenizer=None,
        system_prompt: Optional[str] = None,
    ):
        """Create an `OpenAI` instance.

        This class supports the standard OpenAI API, the Azure OpeanAI API as
        well as compatible APIs that rely on the OpenAI client.

        Parameters
        ----------
        client
            An instance of the API's async client.
        config
            An instance of `OpenAIConfig`. Can be useful to specify some
            parameters that cannot be set by calling this class' methods.
        tokenizer
            The tokenizer associated with the model the client connects to.

        """

        self.client = client
        self.tokenizer = tokenizer
        self.config = config

        # We count the total number of prompt and generated tokens as returned
        # by the OpenAI API, summed over all the requests performed with this
        # model instance.
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def __call__(
        self,
        prompt: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[List[str], str]] = None,
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        samples: Optional[int] = None,
    ) -> np.ndarray:
        """Call the OpenAI API to generate text.

        Parameters
        ----------
        prompt
            A string or list of strings that will be used to prompt the model
        max_tokens
            The maximum number of tokens to generate
        stop_at
            A string or array of strings which, such that the generation stops
            when they are generated.
        system_prompt
            The content of the system message that precedes the user's prompt.
        temperature
            The value of the temperature used to sample tokens
        samples
            The number of completions to generate for each prompt
        stop_at
            Up to 4 words where the API will stop the completion.

        """
        if max_tokens is None:
            max_tokens = self.config.max_tokens
        if stop_at is None:
            stop_at = self.config.stop
        if temperature is None:
            temperature = self.config.temperature
        if samples is None:
            samples = self.config.n

        config = replace(self.config, max_tokens=max_tokens, temperature=temperature, n=samples, stop=stop_at)  # type: ignore

        response, prompt_tokens, completion_tokens = generate_chat(
            prompt, system_prompt, self.client, config
        )
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

        return response

    def stream(self, *args, **kwargs):
        raise NotImplementedError(
            "Streaming is currently not supported for the OpenAI API"
        )

    def generate_choice(
        self,
        prompt: str,
        choices: List[str],
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Call the OpenAI API to generate one of several choices.

        Parameters
        ----------
        prompt
            A string or list of strings that will be used to prompt the model
        choices
            The list of strings between which we ask the model to choose
        max_tokens
            The maximum number of tokens to generate
        system_prompt
            The content of the system message that precedes the user's prompt.

        """
        if self.tokenizer is None:
            raise ValueError(
                "You must initialize the `OpenAI` class with a tokenizer to use `outlines.generate.choice`"
            )

        config = replace(self.config, max_tokens=max_tokens)

        greedy = False
        decoded: List[str] = []
        encoded_choices_left: List[List[int]] = [
            self.tokenizer.encode(word) for word in choices
        ]

        while len(encoded_choices_left) > 0:
            max_tokens_left = max([len(tokens) for tokens in encoded_choices_left])
            transposed_choices_left: List[Set] = [
                {item for item in subset if item is not None}
                for subset in zip_longest(*encoded_choices_left)
            ]

            if not greedy:
                mask = build_optimistic_mask(transposed_choices_left)
            else:
                mask = {}
                for token in transposed_choices_left[0]:  # build greedy mask
                    mask[token] = 100

            if len(mask) == 0:
                break

            config = replace(config, logit_bias=mask, max_tokens=max_tokens_left)

            response, prompt_tokens, completion_tokens = generate_chat(
                prompt, system_prompt, self.client, config
            )
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens

            encoded_response = self.tokenizer.encode(response)

            if encoded_response in encoded_choices_left:
                decoded.append(response)
                break
            else:
                (
                    encoded_response,
                    encoded_choices_left,
                ) = find_response_choices_intersection(
                    encoded_response, encoded_choices_left
                )

                if len(encoded_response) == 0:
                    greedy = True  # next iteration will be "greedy"
                    continue
                else:
                    decoded.append("".join(self.tokenizer.decode(encoded_response)))

                    if len(encoded_choices_left) == 1:  # only one choice left
                        choice_left = self.tokenizer.decode(encoded_choices_left[0])
                        decoded.append(choice_left)
                        break

                    greedy = False  # after each success, stay with (or switch to) "optimistic" approach

                prompt = prompt + "".join(decoded)

        choice = "".join(decoded)

        return choice

    def generate_json(
        self,
        prompt: str,
        schema: str,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Call the OpenAI API to generate a JSON object.

        Parameters
        ----------
        prompt
            A string or list of strings that will be used to prompt the model
        schema
            The schema we ask the model to response
        max_tokens
            The maximum number of tokens to generate
        system_prompt
            The content of the system message that precedes the user's prompt.

        """
        # construct tools to be used in the API call
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "model_validate_json",
                    "description": "Validate the given JSON data against the Pydantic model.",
                    "parameters": schema,
                },
            }
        ]
        config = replace(
            self.config, max_tokens=max_tokens, tools=tools, tool_choice="auto"
        )

        response, prompt_tokens, completion_tokens = generate_chat(
            prompt, system_prompt, self.client, config
        )
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        return response

    def __str__(self):
        return self.__class__.__name__ + " API"

    def __repr__(self):
        return str(self.config)


@functools.partial(vectorize, signature="(),(),(),()->(s),(),()")
async def generate_chat(
    prompt: str,
    system_prompt: Union[str, None],
    client,
    config: OpenAIConfig,
) -> Tuple[np.ndarray, int, int]:
    """Call OpenAI's Chat Completion API.

    Parameters
    ----------
    prompt
        The prompt we use to start the generation. Passed to the model
        with the "user" role.
    system_prompt
        The system prompt, passed to the model with the "system" role
        before the prompt.
    client
        The API client
    config
        An `OpenAIConfig` instance.

    Returns
    -------
    A tuple that contains the model's response(s) and usage statistics.

    """

    @error_handler
    @cache()
    async def call_api(prompt, system_prompt, config):
        responses = await client.chat.completions.create(
            messages=system_message + user_message,
            **asdict(config),  # type: ignore
        )
        return responses.model_dump()

    system_message = (
        [{"role": "system", "content": system_prompt}] if system_prompt else []
    )
    user_message = [{"role": "user", "content": prompt}]

    responses = await call_api(prompt, system_prompt, config)

    results = np.array(
        # if tool_calls in message use tool_calls, else use content
        [
            responses["choices"][i]["message"]["tool_calls"][0]["function"]["arguments"]
            if "tool_calls" in responses["choices"][i]["message"]
            and responses["choices"][i]["message"]["tool_calls"]
            else responses["choices"][i]["message"]["content"]
            for i in range(config.n)
        ]
    )
    usage = responses["usage"]
    return results, usage["prompt_tokens"], usage["completion_tokens"]


def find_longest_intersection(response: List[int], choice: List[int]) -> List[int]:
    """Find the longest intersection between the response and the choice."""
    for i, (token_r, token_c) in enumerate(zip_longest(response, choice)):
        if token_r != token_c:
            return response[:i]

    return response


def find_response_choices_intersection(
    response: List[int], choices: List[List[int]]
) -> Tuple[List[int], List[List[int]]]:
    """Find the longest intersection between the response and the different
    choices.

    Say the response is of the form `[1, 2, 3, 4, 5]` and we have the choices
    `[[1, 2], [1, 2, 3], [6, 7, 8]` then the function will return `[1, 2, 3]` as the
    intersection, and `[[]]` as the list of choices left.

    Parameters
    ----------
    response
        The model's response
    choices
        The remaining possible choices

    Returns
    -------
    A tuple that contains the longest intersection between the response and the
    different choices, and the choices which start with this intersection, with the
    intersection removed.

    """
    max_len_prefix = 0
    choices_left = []
    longest_prefix = []
    for i, choice in enumerate(choices):
        # Find the longest intersection between the response and the choice.
        prefix = find_longest_intersection(response, choice)

        if len(prefix) > max_len_prefix:
            max_len_prefix = len(prefix)
            choices_left = [choice[len(prefix) :]]
            longest_prefix = prefix

        elif len(prefix) == max_len_prefix:
            choices_left.append(choice[len(prefix) :])

    return longest_prefix, choices_left


def build_optimistic_mask(
    transposed: List[Set[int]], max_mask_size: int = 300
) -> Dict[int, int]:
    """We build the largest mask possible.

    Tokens are added from left to right, so if the encoded choices are e.g.
    `[[1,2], [3,4]]`, `1` and `3` will be added before `2` and `4`.

    Parameters
    ----------
    transposed
        A list of lists that contain the nth token of each choice.

    """
    mask: Dict[int, int] = {}
    for tokens in transposed:
        for token in tokens:
            if len(mask) == max_mask_size:
                return mask
            mask[token] = 100

    return mask


def error_handler(api_call_fn: Callable) -> Callable:
    """Handle OpenAI API errors and missing API key."""

    def call(*args, **kwargs):
        import openai

        try:
            return api_call_fn(*args, **kwargs)
        except (
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.RateLimitError,
        ) as e:
            raise OSError(f"Could not connect to the OpenAI API: {e}")
        except (
            openai.AuthenticationError,
            openai.BadRequestError,
            openai.ConflictError,
            openai.PermissionDeniedError,
            openai.NotFoundError,
            openai.UnprocessableEntityError,
        ) as e:
            raise e

    return call


@functools.singledispatch
def openai(model_or_client, *args, **kwargs):
    return OpenAI(model_or_client, *args, **kwargs)


@openai.register(str)
def openai_model(
    model_name: str,
    config: Optional[OpenAIConfig] = None,
    **openai_client_params,
):
    try:
        import tiktoken
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError(
            "The `openai` and `tiktoken` libraries needs to be installed in order to use Outlines' OpenAI integration."
        )

    if config is not None:
        config = replace(config, model=model_name)  # type: ignore
    else:
        config = OpenAIConfig(model=model_name)

    client = AsyncOpenAI(**openai_client_params)

    try:
        tokenizer = tiktoken.encoding_for_model(model_name)
    except KeyError:
        warnings.warn(
            f"Could not find a tokenizer for model {model_name}. Using default cl100k_base."
        )
        tokenizer = tiktoken.get_encoding("cl100k_base")

    return OpenAI(client, config, tokenizer)


def azure_openai(
    deployment_name: str,
    model_name: Optional[str] = None,
    config: Optional[OpenAIConfig] = None,
    **azure_openai_client_params,
):
    try:
        import tiktoken
        from openai import AsyncAzureOpenAI
    except ImportError:
        raise ImportError(
            "The `openai` and `tiktoken` libraries needs to be installed in order to use Outlines' Azure OpenAI integration."
        )

    if config is not None:
        config = replace(config, model=deployment_name)  # type: ignore
    if config is None:
        config = OpenAIConfig(model=deployment_name)

    client = AsyncAzureOpenAI(**azure_openai_client_params)

    try:
        tokenizer = tiktoken.encoding_for_model(model_name or deployment_name)
    except KeyError:
        warnings.warn(
            f"Could not find a tokenizer for model {model_name or deployment_name}. Using default cl100k_base."
        )
        tokenizer = tiktoken.get_encoding("cl100k_base")

    return OpenAI(client, config, tokenizer)
