import inspect
import json
import re
import warnings
from typing import Callable, Optional, Tuple, Type, Union

from jsonschema.protocols import Validator
from pydantic import BaseModel, create_model
from referencing import Registry, Resource
from referencing._core import Resolver
from referencing.jsonschema import DRAFT202012

# allow `\"`, `\\`, or any character which isn't a control sequence
STRING_INNER = r'([^"\\\x00-\x1F\x7F-\x9F]|\\["\\])'
STRING = f'"{STRING_INNER}*"'

INTEGER = r"(-)?(0|[1-9][0-9]*)"
NUMBER = rf"({INTEGER})(\.[0-9]+)?([eE][+-][0-9]+)?"
BOOLEAN = r"(true|false)"
NULL = r"null"
WHITESPACE = r"[ ]?"

type_to_regex = {
    "string": STRING,
    "integer": INTEGER,
    "number": NUMBER,
    "boolean": BOOLEAN,
    "null": NULL,
}

DATE_TIME = r'"(-?(?:[1-9][0-9]*)?[0-9]{4})-(1[0-2]|0[1-9])-(3[01]|0[1-9]|[12][0-9])T(2[0-3]|[01][0-9]):([0-5][0-9]):([0-5][0-9])(\.[0-9]{3})?(Z)?"'
DATE = r'"(?:\d{4})-(?:0[1-9]|1[0-2])-(?:0[1-9]|[1-2][0-9]|3[0-1])"'
TIME = r'"(2[0-3]|[01][0-9]):([0-5][0-9]):([0-5][0-9])(\\.[0-9]+)?(Z)?"'
UUID = r'"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"'

format_to_regex = {
    "uuid": UUID,
    "date-time": DATE_TIME,
    "date": DATE,
    "time": TIME,
}


def build_regex_from_schema(schema: str, whitespace_pattern: Optional[str] = None):
    """Turn a JSON schema into a regex that matches any JSON object that follows
    this schema.

    JSON Schema is a declarative language that allows to annotate JSON documents
    with types and descriptions. These schemas can be generated from any Python
    datastructure that has type annotation: namedtuples, dataclasses, Pydantic
    models. And by ensuring that the generation respects the schema we ensure
    that the output can be parsed into these objects.
    This function parses the provided schema and builds a generation schedule which
    mixes deterministic generation (fixed strings), and sampling with constraints.

    Parameters
    ----------
    schema
        A string that represents a JSON Schema.
    whitespace_pattern
        Pattern to use for JSON syntactic whitespace (doesn't impact string literals)
        Example: allow only a single space or newline with `whitespace_pattern=r"[\n ]?"`

    Returns
    -------
    A generation schedule. A list of strings that represent the JSON
    schema's structure and regular expression that define the structure of
    the fields.

    References
    ----------
    .. [0] JSON Schema. https://json-schema.org/

    """

    schema = json.loads(schema)
    Validator.check_schema(schema)

    # Build reference resolver
    schema = Resource(contents=schema, specification=DRAFT202012)
    uri = schema.id() if schema.id() is not None else ""
    registry = Registry().with_resource(uri=uri, resource=schema)
    resolver = registry.resolver()

    content = schema.contents
    return to_regex(resolver, content, whitespace_pattern)


def convert_json_schema_to_str(json_schema: Union[dict, str, Type[BaseModel]]) -> str:
    """Convert a JSON schema to a string.

    Parameters
    ----------
    json_schema
        The JSON schema.

    Returns
    -------
    str
        The JSON schema converted to a string.

    Raises
    ------
    ValueError
        If the schema is not a dictionary, a string or a Pydantic class.
    """
    if isinstance(json_schema, dict):
        schema_str = json.dumps(json_schema)
    elif isinstance(json_schema, str):
        schema_str = json_schema
    elif issubclass(json_schema, BaseModel):
        schema_str = json.dumps(json_schema.model_json_schema())
    else:
        raise ValueError(
            f"Cannot parse schema {json_schema}. The schema must be either "
            + "a Pydantic class, a dictionary or a string that contains the JSON "
            + "schema specification"
        )
    return schema_str


def _get_num_items_pattern(min_items, max_items, whitespace_pattern):
    # Helper function for arrays and objects
    min_items = int(min_items or 0)
    if max_items is None:
        return rf"{{{max(min_items - 1, 0)},}}"
    else:
        max_items = int(max_items)
        if max_items < 1:
            return None
        return rf"{{{max(min_items - 1, 0)},{max_items - 1}}}"


def validate_quantifiers(
    min_bound: Optional[str], max_bound: Optional[str], start_offset: int = 0
) -> Tuple[str, str]:
    """
    Ensures that the bounds of a number are valid. Bounds are used as quantifiers in the regex.

    Parameters
    ----------
    min_bound
        The minimum value that the number can take.
    max_bound
        The maximum value that the number can take.
    start_offset
        Number of elements that are already present in the regex but still need to be counted.
        ex: if the regex is already "(-)?(0|[1-9][0-9])", we will always have at least 1 digit, so the start_offset is 1.

    Returns
    -------
    min_bound
        The minimum value that the number can take.
    max_bound
        The maximum value that the number can take.

    Raises
    ------
    ValueError
        If the minimum bound is greater than the maximum bound.

    TypeError or ValueError
        If the minimum bound is not an integer or None.
        or
        If the maximum bound is not an integer or None.
    """
    min_bound = "" if min_bound is None else str(int(min_bound) - start_offset)
    max_bound = "" if max_bound is None else str(int(max_bound) - start_offset)
    if min_bound and max_bound:
        if int(max_bound) < int(min_bound):
            raise ValueError("max bound must be greater than or equal to min bound")
    return min_bound, max_bound


def to_regex(
    resolver: Resolver, instance: dict, whitespace_pattern: Optional[str] = None
):
    """Translate a JSON Schema instance into a regex that validates the schema.

    Note
    ----
    Many features of JSON schema are missing:
    - Handle `additionalProperties` keyword
    - Handle types defined as a list
    - Handle constraints on numbers
    - Handle special patterns: `date`, `uri`, etc.

    This does not support recursive definitions.

    Parameters
    ----------
    resolver
        An object that resolves references to other instances within a schema
    instance
        The instance to translate
    whitespace_pattern
        Pattern to use for JSON syntactic whitespace (doesn't impact string literals)
        Example: allow only a single space or newline with `whitespace_pattern=r"[\n ]?"`
    """

    # set whitespace pattern
    if whitespace_pattern is None:
        whitespace_pattern = WHITESPACE

    if instance == {}:
        # JSON Schema Spec: Empty object means unconstrained, any json type is legal
        types = [
            {"type": "boolean"},
            {"type": "null"},
            {"type": "number"},
            {"type": "integer"},
            {"type": "string"},
            {"type": "array"},
            {"type": "object"},
        ]
        regexes = [to_regex(resolver, t, whitespace_pattern) for t in types]
        regexes = [rf"({r})" for r in regexes]
        return rf"{'|'.join(regexes)}"

    elif "properties" in instance:
        regex = ""
        regex += r"\{"
        properties = instance["properties"]
        required_properties = instance.get("required", [])
        is_required = [item in required_properties for item in properties]
        # If at least one property is required, we include the one in the lastest position
        # without any comma.
        # For each property before it (optional or required), we add with a comma after the property.
        # For each property after it (optional), we add with a comma before the property.
        if any(is_required):
            last_required_pos = max([i for i, value in enumerate(is_required) if value])
            for i, (name, value) in enumerate(properties.items()):
                subregex = f'{whitespace_pattern}"{re.escape(name)}"{whitespace_pattern}:{whitespace_pattern}'
                subregex += to_regex(resolver, value, whitespace_pattern)
                if i < last_required_pos:
                    subregex = f"{subregex}{whitespace_pattern},"
                elif i > last_required_pos:
                    subregex = f"{whitespace_pattern},{subregex}"
                regex += subregex if is_required[i] else f"({subregex})?"
        # If no property is required, we have to create a possible pattern for each property in which
        # it's the last one necessarilly present. Then, we add the others as optional before and after
        # following the same strategy as described above.
        # The whole block is made optional to allow the case in which no property is returned.
        else:
            property_subregexes = []
            for i, (name, value) in enumerate(properties.items()):
                subregex = f'{whitespace_pattern}"{name}"{whitespace_pattern}:{whitespace_pattern}'
                subregex += to_regex(resolver, value, whitespace_pattern)
                property_subregexes.append(subregex)
            possible_patterns = []
            for i in range(len(property_subregexes)):
                pattern = ""
                for subregex in property_subregexes[:i]:
                    pattern += f"({subregex}{whitespace_pattern},)?"
                pattern += property_subregexes[i]
                for subregex in property_subregexes[i + 1 :]:
                    pattern += f"({whitespace_pattern},{subregex})?"
                possible_patterns.append(pattern)
            regex += f"({'|'.join(possible_patterns)})?"

        regex += f"{whitespace_pattern}" + r"\}"

        return regex

    # To validate against allOf, the given data must be valid against all of the
    # given subschemas.
    elif "allOf" in instance:
        subregexes = [
            to_regex(resolver, t, whitespace_pattern) for t in instance["allOf"]
        ]
        subregexes_str = [f"{subregex}" for subregex in subregexes]
        return rf"({''.join(subregexes_str)})"

    # To validate against `anyOf`, the given data must be valid against
    # any (one or more) of the given subschemas.
    elif "anyOf" in instance:
        subregexes = [
            to_regex(resolver, t, whitespace_pattern) for t in instance["anyOf"]
        ]
        return rf"({'|'.join(subregexes)})"

    # To validate against oneOf, the given data must be valid against exactly
    # one of the given subschemas.
    elif "oneOf" in instance:
        subregexes = [
            to_regex(resolver, t, whitespace_pattern) for t in instance["oneOf"]
        ]

        xor_patterns = [f"(?:{subregex})" for subregex in subregexes]

        return rf"({'|'.join(xor_patterns)})"

    # Create pattern for Tuples, per JSON Schema spec, `prefixItems` determines types at each idx
    elif "prefixItems" in instance:
        element_patterns = [
            to_regex(resolver, t, whitespace_pattern) for t in instance["prefixItems"]
        ]
        comma_split_pattern = rf"{whitespace_pattern},{whitespace_pattern}"
        tuple_inner = comma_split_pattern.join(element_patterns)
        return rf"\[{whitespace_pattern}{tuple_inner}{whitespace_pattern}\]"

    # The enum keyword is used to restrict a value to a fixed set of values. It
    # must be an array with at least one element, where each element is unique.
    elif "enum" in instance:
        choices = []
        for choice in instance["enum"]:
            if type(choice) in [int, float, bool, type(None), str]:
                choices.append(re.escape(json.dumps(choice)))
            else:
                raise TypeError(f"Unsupported data type in enum: {type(choice)}")
        return f"({'|'.join(choices)})"

    elif "const" in instance:
        const = instance["const"]
        if type(const) in [int, float, bool, type(None), str]:
            const = re.escape(json.dumps(const))
        else:
            raise TypeError(f"Unsupported data type in const: {type(const)}")
        return const

    elif "$ref" in instance:
        path = f"{instance['$ref']}"
        instance = resolver.lookup(path).contents
        return to_regex(resolver, instance, whitespace_pattern)

    # The type keyword may either be a string or an array:
    # - If it's a string, it is the name of one of the basic types.
    # - If it is an array, it must be an array of strings, where each string is
    # the name of one of the basic types, and each element is unique. In this
    # case, the JSON snippet is valid if it matches any of the given types.
    elif "type" in instance:
        instance_type = instance["type"]
        if instance_type == "string":
            if "maxLength" in instance or "minLength" in instance:
                max_items = instance.get("maxLength", "")
                min_items = instance.get("minLength", "")
                try:
                    if int(max_items) < int(min_items):
                        raise ValueError(
                            "maxLength must be greater than or equal to minLength"
                        )  # FIXME this raises an error but is caught right away by the except (meant for int("") I assume)
                except ValueError:
                    pass
                return f'"{STRING_INNER}{{{min_items},{max_items}}}"'
            elif "pattern" in instance:
                pattern = instance["pattern"]
                if pattern[0] == "^" and pattern[-1] == "$":
                    return rf'("{pattern[1:-1]}")'
                else:
                    return rf'("{pattern}")'
            elif "format" in instance:
                format = instance["format"]
                if format == "date-time":
                    return format_to_regex["date-time"]
                elif format == "uuid":
                    return format_to_regex["uuid"]
                elif format == "date":
                    return format_to_regex["date"]
                elif format == "time":
                    return format_to_regex["time"]
                else:
                    raise NotImplementedError(
                        f"Format {format} is not supported by Outlines"
                    )
            else:
                return type_to_regex["string"]

        elif instance_type == "number":
            bounds = {
                "minDigitsInteger",
                "maxDigitsInteger",
                "minDigitsFraction",
                "maxDigitsFraction",
                "minDigitsExponent",
                "maxDigitsExponent",
            }
            if bounds.intersection(set(instance.keys())):
                min_digits_integer, max_digits_integer = validate_quantifiers(
                    instance.get("minDigitsInteger"),
                    instance.get("maxDigitsInteger"),
                    start_offset=1,
                )
                min_digits_fraction, max_digits_fraction = validate_quantifiers(
                    instance.get("minDigitsFraction"), instance.get("maxDigitsFraction")
                )
                min_digits_exponent, max_digits_exponent = validate_quantifiers(
                    instance.get("minDigitsExponent"), instance.get("maxDigitsExponent")
                )
                integers_quantifier = (
                    f"{{{min_digits_integer},{max_digits_integer}}}"
                    if min_digits_integer or max_digits_integer
                    else "*"
                )
                fraction_quantifier = (
                    f"{{{min_digits_fraction},{max_digits_fraction}}}"
                    if min_digits_fraction or max_digits_fraction
                    else "+"
                )
                exponent_quantifier = (
                    f"{{{min_digits_exponent},{max_digits_exponent}}}"
                    if min_digits_exponent or max_digits_exponent
                    else "+"
                )
                return rf"((-)?(0|[1-9][0-9]{integers_quantifier}))(\.[0-9]{fraction_quantifier})?([eE][+-][0-9]{exponent_quantifier})?"
            return type_to_regex["number"]

        elif instance_type == "integer":
            if "minDigits" in instance or "maxDigits" in instance:
                min_digits, max_digits = validate_quantifiers(
                    instance.get("minDigits"), instance.get("maxDigits"), start_offset=1
                )
                return rf"(-)?(0|[1-9][0-9]{{{min_digits},{max_digits}}})"
            return type_to_regex["integer"]

        elif instance_type == "array":
            num_repeats = _get_num_items_pattern(
                instance.get("minItems"), instance.get("maxItems"), whitespace_pattern
            )
            if num_repeats is None:
                return rf"\[{whitespace_pattern}\]"

            allow_empty = "?" if int(instance.get("minItems", 0)) == 0 else ""

            if "items" in instance:
                items_regex = to_regex(resolver, instance["items"], whitespace_pattern)
                return rf"\[{whitespace_pattern}(({items_regex})(,{whitespace_pattern}({items_regex})){num_repeats}){allow_empty}{whitespace_pattern}\]"
            else:
                # Here we need to make the choice to exclude generating list of objects
                # if the specification of the object is not given, even though a JSON
                # object that contains an object here would be valid under the specification.
                legal_types = [
                    {"type": "boolean"},
                    {"type": "null"},
                    {"type": "number"},
                    {"type": "integer"},
                    {"type": "string"},
                ]
                depth = instance.get("depth", 2)
                if depth > 0:
                    legal_types.append({"type": "object", "depth": depth - 1})
                    legal_types.append({"type": "array", "depth": depth - 1})

                regexes = [
                    to_regex(resolver, t, whitespace_pattern) for t in legal_types
                ]
                return rf"\[{whitespace_pattern}({'|'.join(regexes)})(,{whitespace_pattern}({'|'.join(regexes)})){num_repeats}{allow_empty}{whitespace_pattern}\]"

        elif instance_type == "object":
            # pattern for json object with values defined by instance["additionalProperties"]
            # enforces value type constraints recursively, "minProperties", and "maxProperties"
            # doesn't enforce "required", "dependencies", "propertyNames" "any/all/on Of"
            num_repeats = _get_num_items_pattern(
                instance.get("minProperties"),
                instance.get("maxProperties"),
                whitespace_pattern,
            )
            if num_repeats is None:
                return rf"\{{{whitespace_pattern}\}}"

            allow_empty = "?" if int(instance.get("minProperties", 0)) == 0 else ""

            additional_properties = instance.get("additionalProperties")

            if additional_properties is None or additional_properties is True:
                # JSON Schema behavior: If the additionalProperties of an object is
                # unset or True, it is unconstrained object.
                # We handle this by setting additionalProperties to anyOf: {all types}

                legal_types = [
                    {"type": "string"},
                    {"type": "number"},
                    {"type": "boolean"},
                    {"type": "null"},
                ]

                # We set the object depth to 2 to keep the expression finite, but the "depth"
                # key is not a true component of the JSON Schema specification.
                depth = instance.get("depth", 2)
                if depth > 0:
                    legal_types.append({"type": "object", "depth": depth - 1})
                    legal_types.append({"type": "array", "depth": depth - 1})
                additional_properties = {"anyOf": legal_types}

            value_pattern = to_regex(
                resolver, additional_properties, whitespace_pattern
            )
            key_value_pattern = (
                f"{STRING}{whitespace_pattern}:{whitespace_pattern}{value_pattern}"
            )
            key_value_successor_pattern = (
                f"{whitespace_pattern},{whitespace_pattern}{key_value_pattern}"
            )
            multiple_key_value_pattern = f"({key_value_pattern}({key_value_successor_pattern}){num_repeats}){allow_empty}"

            return (
                r"\{"
                + whitespace_pattern
                + multiple_key_value_pattern
                + whitespace_pattern
                + r"\}"
            )

        elif instance_type == "boolean":
            return type_to_regex["boolean"]

        elif instance_type == "null":
            return type_to_regex["null"]

        elif isinstance(instance_type, list):
            # Here we need to make the choice to exclude generating an object
            # if the specification of the object is not give, even though a JSON
            # object that contains an object here would be valid under the specification.
            regexes = [
                to_regex(resolver, {"type": t}, whitespace_pattern)
                for t in instance_type
                if t != "object"
            ]
            return rf"({'|'.join(regexes)})"

    raise NotImplementedError(
        f"""Could not translate the instance {instance} to a
    regular expression. Make sure it is valid to the JSON Schema specification. If
    it is, please open an issue on the Outlines repository"""
    )


def get_schema_from_signature(fn: Callable) -> str:
    """Turn a function signature into a JSON schema.

    Every JSON object valid to the output JSON Schema can be passed
    to `fn` using the ** unpacking syntax.

    """
    signature = inspect.signature(fn)
    arguments = {}
    for name, arg in signature.parameters.items():
        if arg.annotation == inspect._empty:
            raise ValueError("Each argument must have a type annotation")
        else:
            arguments[name] = (arg.annotation, ...)

    try:
        fn_name = fn.__name__
    except Exception as e:
        fn_name = "Arguments"
        warnings.warn(
            f"The function name could not be determined. Using default name 'Arguments' instead. For debugging, here is exact error:\n{e}",
            category=UserWarning,
        )
    model = create_model(fn_name, **arguments)

    return model.model_json_schema()
