import json as pyjson
from functools import singledispatch
from typing import Callable, Optional, Union

from pydantic import BaseModel

from outlines.fsm.json_schema import build_regex_from_schema, get_schema_from_signature
from outlines.generate.api import SequenceGeneratorAdapter
from outlines.models import OpenAI
from outlines.samplers import Sampler, multinomial

from .regex import regex


@singledispatch
def json(
    model,
    schema_object: Union[str, object, Callable],
    sampler: Sampler = multinomial(),
    whitespace_pattern: Optional[str] = None,
    strict=True,
) -> SequenceGeneratorAdapter:
    """
    Generate structured JSON data with a `Transformer` model based on a specified JSON Schema.

    Parameters
    ----------
    model:
        An instance of `Transformer` that represents a model from the
        `transformers` library.
    schema_object:
        The JSON Schema to generate data for. Can be a JSON string, a Pydantic model, or a callable
        that returns a JSON schema.
    sampler:
        The sampling algorithm to use to generate token ids from the logits
        distribution.
    whitespace_pattern
        Pattern to use for JSON syntactic whitespace (doesn't impact string literals)
        Example: allow only a single space or newline with `whitespace_pattern=r"[\n ]?"`
    strict
        If strict mode is enabled, generations which don't conform to the schema or aren't
        valid JSON will result in an error. Outlines guarantees generation complies with a schema,
        but schemas often allow for infinite repetition and exhaust the model_max_length.

    Returns
    -------
    A `SequenceGenerator` instance that generates text constrained by the schema_object and
    transforms the result if BaseModel is used.

    """

    def maybe_strict_formatter(formatter):
        """If strict, use normal formatter. Otherwise, return error dict on failure"""
        if strict:
            return formatter

        def allow_fail_formatter(generated_output):
            try:
                return formatter(generated_output)
            except Exception as e:
                return {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "output": generated_output,
                }

        return allow_fail_formatter

    if isinstance(schema_object, type(BaseModel)):
        schema = pyjson.dumps(schema_object.model_json_schema())
        regex_str = build_regex_from_schema(schema, whitespace_pattern)
        generator = regex(model, regex_str, sampler)
        generator.format_sequence = maybe_strict_formatter(schema_object.parse_raw)
    elif callable(schema_object):
        schema = pyjson.dumps(get_schema_from_signature(schema_object))
        regex_str = build_regex_from_schema(schema, whitespace_pattern)
        generator = regex(model, regex_str, sampler)
        generator.format_sequence = maybe_strict_formatter(pyjson.loads)
    elif isinstance(schema_object, str):
        schema = schema_object
        regex_str = build_regex_from_schema(schema, whitespace_pattern)
        generator = regex(model, regex_str, sampler)
        generator.format_sequence = maybe_strict_formatter(pyjson.loads)
    else:
        raise ValueError(
            f"Cannot parse schema {schema_object}. The schema must be either "
            + "a Pydantic object, a function or a string that contains the JSON "
            + "Schema specification"
        )

    return generator


@json.register(OpenAI)
def json_openai(
    model, schema_object: Union[str, object], sampler: Sampler = multinomial()
):
    if not isinstance(sampler, multinomial):
        raise NotImplementedError(
            r"The OpenAI API does not support any other sampling algorithm "
            + "than the multinomial sampler."
        )

    if isinstance(schema_object, type(BaseModel)):
        schema = pyjson.dumps(schema_object.model_json_schema())
        format_sequence = lambda x: schema_object.parse_raw(x)
    elif isinstance(schema_object, str):
        schema = schema_object
        format_sequence = lambda x: pyjson.loads(x)
    else:
        raise ValueError(
            f"Cannot parse schema {schema_object}. The schema must be either "
            + "a Pydantic object, a function or a string that contains the JSON "
            + "Schema specification"
        )

    # create copied, patched model with normalized json schema set
    generator = model.new_with_replacements(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "default",
                "strict": True,
                "schema": pyjson.loads(schema),
            },
        }
    )

    generator.format_sequence = format_sequence

    return generator
