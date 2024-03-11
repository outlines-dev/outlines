import json as pyjson
from functools import singledispatch
from typing import Callable, Optional, Union

from pydantic import BaseModel

from outlines.fsm.json_schema import build_regex_from_schema, get_schema_from_signature
from outlines.generate.api import SequenceGenerator
from outlines.models import OpenAI
from outlines.samplers import Sampler, multinomial

from .regex import regex


@singledispatch
def json(
    model,
    schema_object: Union[str, object, Callable],
    sampler: Sampler = multinomial(),
    whitespace_pattern: Optional[str] = None,
    ignore_fields_order: bool = False,
) -> SequenceGenerator:
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
    ignore_fields_order
        If True, the generated JSON will not be constrained by the order of the fields
        in the schema.
    Returns
    -------
    A `SequenceGenerator` instance that generates text constrained by the schema_object and
    transforms the result if BaseModel is used.

    """
    if isinstance(schema_object, type(BaseModel)):
        schema = pyjson.dumps(schema_object.model_json_schema())
        regex_str = build_regex_from_schema(
            schema, whitespace_pattern, ignore_fields_order
        )
        generator = regex(model, regex_str, sampler)
        generator.format_sequence = lambda x: schema_object.parse_raw(x)
    elif callable(schema_object):
        schema = pyjson.dumps(get_schema_from_signature(schema_object))
        regex_str = build_regex_from_schema(
            schema, whitespace_pattern, ignore_fields_order
        )
        generator = regex(model, regex_str, sampler)
        generator.format_sequence = lambda x: pyjson.loads(x)
    elif isinstance(schema_object, str):
        schema = schema_object
        regex_str = build_regex_from_schema(
            schema, whitespace_pattern, ignore_fields_order
        )
        generator = regex(model, regex_str, sampler)
        generator.format_sequence = lambda x: pyjson.loads(x)
    else:
        raise ValueError(
            f"Cannot parse schema {schema_object}. The schema must be either "
            + "a Pydantic object, a function or a string that contains the JSON "
            + "Schema specification"
        )

    return generator


@json.register(OpenAI)
def json_openai(
    model, schema_object: Union[str, object, Callable], sampler: Sampler = multinomial()
):
    raise NotImplementedError(
        "Cannot use JSON Schema-structure generation with an OpenAI model "
        + "due to the limitations of the OpenAI API"
    )
