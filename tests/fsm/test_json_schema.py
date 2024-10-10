import json
import random
import re
import string as pystring
from typing import List, Literal, Union

import interegular
import pytest
from pydantic import BaseModel, Field, constr

from outlines.fsm.json_schema import (
    BOOLEAN,
    DATE,
    DATE_TIME,
    INTEGER,
    NULL,
    NUMBER,
    SAFE_WHITESPACE,
    STRING,
    STRING_INNER,
    TIME,
    UUID,
    build_regex_from_schema,
    get_int_pattern,
    get_schema_from_signature,
    get_str_pattern,
    to_regex,
)

SAFE_INT = get_int_pattern(safe_subset=True)
SAFE_STR = get_str_pattern(safe_subset=True)


def test_function_basic():
    def test_function(foo: str, bar: List[int]):
        pass

    result = get_schema_from_signature(test_function)
    assert result["type"] == "object"
    assert list(result["properties"].keys()) == ["foo", "bar"]
    assert result["properties"]["foo"]["type"] == "string"
    assert result["properties"]["bar"]["type"] == "array"
    assert result["properties"]["bar"]["items"]["type"] == "integer"


def test_function_no_type():
    def test_function(foo, bar: List[int]):
        pass

    with pytest.raises(ValueError):
        get_schema_from_signature(test_function)


def test_from_pydantic():
    class User(BaseModel):
        user_id: int
        name: str
        maxlength_name: constr(max_length=10)
        minlength_name: constr(min_length=10)
        value: float
        is_true: bool

    schema = json.dumps(User.model_json_schema())
    schedule = build_regex_from_schema(schema)
    assert isinstance(schedule, str)


@pytest.mark.parametrize(
    "pattern,does_match",
    [
        ({"integer": "0"}, True),
        ({"integer": "1"}, True),
        ({"integer": "-1"}, True),
        ({"integer": "01"}, False),
        ({"integer": "1.3"}, False),
        ({"integer": "t"}, False),
    ],
)
def test_match_integer(pattern, does_match):
    step = {"title": "Foo", "type": "integer"}
    regex = to_regex(None, step, safe_subset=False)
    assert regex == INTEGER

    value = pattern["integer"]
    match = re.fullmatch(regex, value)
    if does_match:
        assert match[0] == value
        assert match.span() == (0, len(value))
    else:
        assert match is None


@pytest.mark.parametrize(
    "minimum,maximum",
    [
        (0, 0),
        (-1, 0),
        (0, 1),
        (-15, 0),
        (0, 15),
        (-1, 1),
        (-15, 15),
        (-1234, 56),
        (-56, 1234),
        (-9, 9),
        (-10, 10),
        (-9, 10),
        (-10, 9),
        (123, 199),
        (123, 456),
        (5600, 5678),
        (550, 560),
        (-12345, 3423),
        (50, 10000),
        (0, 1000),
        (-100000, 0),
        (-100000, 100000),
    ],
)
def test_int_range_pattern(minimum, maximum):
    pattern = get_int_pattern(minimum, maximum)
    fsm = interegular.parse_pattern(pattern).to_fsm().reduce()
    pattern_numbers = {"".join(s) for s in fsm.strings()}
    range_numbers = set(map(str, range(minimum, maximum + 1)))
    if "0" in range_numbers:
        range_numbers.add("-0")
    assert pattern_numbers == range_numbers

    # logarithmic space complexity
    assert len(fsm.states) <= (len(str(minimum)) + len(str(maximum))) * 2


def test_int_range_unconstrained():
    # test unconstrained
    pattern = get_int_pattern(float("-inf"), float("inf"))
    fsm = interegular.parse_pattern(pattern).to_fsm().reduce()
    assert get_int_pattern(None, None) == pattern
    assert fsm.accepts("0")
    assert fsm.accepts("-1")
    assert fsm.accepts("1")
    assert fsm.accepts("-98427983498234893274983274892")
    assert fsm.accepts("2994399439493294329432984932")

    assert not fsm.accepts("1.1")
    assert not fsm.accepts("1.0")
    assert not fsm.accepts("1.0")
    assert not fsm.accepts("one")

    assert len(fsm.states) < 5


def test_int_range_min_zero():
    # test min zero
    pattern = get_int_pattern(0, float("inf"))
    fsm = interegular.parse_pattern(pattern).to_fsm()
    assert fsm.accepts("0")
    assert not fsm.accepts("-1")
    assert fsm.accepts("1")
    assert not fsm.accepts("-98427983498234893274983274892")
    assert fsm.accepts("2994399439493294329432984932")


def test_int_range_max_zero():
    # test min zero
    pattern = get_int_pattern(-float("inf"), 0)
    fsm = interegular.parse_pattern(pattern).to_fsm()
    assert fsm.accepts("0")
    assert fsm.accepts("-1")
    assert not fsm.accepts("1")
    assert fsm.accepts("-98427983498234893274983274892")
    assert not fsm.accepts("2994399439493294329432984932")


def test_int_range_max_minus_32():
    # test min zero
    pattern = get_int_pattern(-float("inf"), -32)
    fsm = interegular.parse_pattern(pattern).to_fsm()
    assert not fsm.accepts("0")
    assert not fsm.accepts("-1")
    assert not fsm.accepts("1")
    assert not fsm.accepts("32")
    assert fsm.accepts("-32")
    assert fsm.accepts("-33")
    assert fsm.accepts("-39482929438")
    assert fsm.accepts("-98427983498234893274983274892")
    assert not fsm.accepts("2994399439493294329432984932")


@pytest.mark.parametrize(
    "pattern,does_match",
    [
        ({"number": "1"}, True),
        ({"number": "0"}, True),
        ({"number": "01"}, False),
        ({"number": ".3"}, False),
        ({"number": "1.3"}, True),
        ({"number": "-1.3"}, True),
        ({"number": "1.3e9"}, False),
        ({"number": "1.3e+9"}, True),
    ],
)
def test_match_number(pattern, does_match):
    step = {"title": "Foo", "type": "number"}
    regex = to_regex(None, step)
    assert regex == NUMBER

    value = pattern["number"]
    match = re.fullmatch(regex, value)
    if does_match:
        assert match[0] == value
        assert match.span() == (0, len(value))
    else:
        assert match is None


@pytest.mark.parametrize(
    "min_len,max_len,safe_subset,expected_max,errors",
    [
        # if no max and no safe mode, any length allowed
        (None, None, False, None, False),
        # max_len is None, use safe max_len of 256
        (None, None, True, 256, False),
        (0, None, True, 256, False),
        # if max_len is specified, it overrides safe_subset rules
        (None, 500, True, 500, False),
        (0, 500, True, 500, False),
        # if min_len specification has no effect
        (3, 500, True, 500, False),
        (30, 500, True, 500, False),
        (300, 500, True, 500, False),
        # illegal
        (-1, None, True, None, True),
        (30, 20, True, None, True),
    ],
)
def test_get_str_pattern(min_len, max_len, safe_subset, expected_max, errors):
    if errors:
        with pytest.raises(ValueError):
            get_str_pattern(min_len, max_len, safe_subset)
        return

    pattern = get_str_pattern(min_len, max_len, safe_subset)

    # verify str len in (min_len, max_len)
    def str_of_len(str_len):
        s = "".join(random.choices(pystring.ascii_letters + pystring.digits, k=str_len))
        return f'"{s}"'

    # verify min_len held
    min_len = min_len or 0
    assert re.match(pattern, str_of_len(min_len))
    if min_len != 0:
        assert not re.match(pattern, str_of_len(min_len - 1))

    # verify expected_max accurate
    if expected_max is not None:
        assert re.match(pattern, str_of_len(expected_max))
        assert re.match(pattern, str_of_len(max(expected_max - 1, min_len)))
        assert not re.match(pattern, str_of_len(expected_max + 1))
    else:
        assert re.match(pattern, str_of_len(max(min_len, 100)))
        assert re.match(pattern, str_of_len(max(min_len, 1000)))
        assert re.match(pattern, str_of_len(max(min_len, 100000)))


@pytest.mark.parametrize(
    "schema,regex,examples",
    [
        # String
        (
            {"title": "Foo", "type": "string"},
            STRING,
            [
                ("unquotedstring", False),
                ('"(parenthesized_string)"', True),
                ('"malformed) parenthesis (((() string"', True),
                ('"quoted_string"', True),
                (r'"escape_\character"', False),
                (r'"double_\\escape"', True),
                (r'"\n"', False),
                (r'"\\n"', True),
                (r'"unescaped " quote"', False),
                (r'"escaped \" quote"', True),
            ],
        ),
        # String with maximum length
        (
            {"title": "Foo", "type": "string", "maxLength": 3},
            f'"{STRING_INNER}{{,3}}"',
            [('"ab"', True), ('"a""', False), ('"abcd"', False)],
        ),
        # String with minimum length
        (
            {"title": "Foo", "type": "string", "minLength": 3},
            f'"{STRING_INNER}{{3,}}"',
            [('"ab"', False), ('"abcd"', True), ('"abc""', False)],
        ),
        # String with both minimum and maximum length
        (
            {"title": "Foo", "type": "string", "minLength": 3, "maxLength": 5},
            f'"{STRING_INNER}{{3,5}}"',
            [('"ab"', False), ('"abcd"', True), ('"abcdef""', False)],
        ),
        # String defined by a regular expression
        (
            {"title": "Foo", "type": "string", "pattern": r"^[a-z]$"},
            r'("[a-z]")',
            [('"a"', True), ('"1"', False)],
        ),
        # Boolean
        (
            {"title": "Foo", "type": "boolean"},
            BOOLEAN,
            [
                ("true", True),
                ("false", True),
                ("null", False),
                ("0", False),
            ],
        ),
        # Null
        (
            {"title": "Foo", "type": "null"},
            NULL,
            [
                ("null", True),
                ("true", False),
                ("0", False),
            ],
        ),
        # Const string
        (
            {"title": "Foo", "const": "Marc", "type": "string"},
            '"Marc"',
            [('"Marc"', True), ('"Jean"', False), ('"John"', False)],
        ),
        # Make sure strings are escaped with regex escaping
        (
            {"title": "Foo", "const": ".*", "type": "string"},
            r'"\.\*"',
            [('".*"', True), (r'"\s*"', False), (r'"\.\*"', False)],
        ),
        # Make sure strings are escaped with JSON escaping
        (
            {"title": "Foo", "const": '"', "type": "string"},
            r'"\\""',
            [('"\\""', True), ('"""', False)],
        ),
        # Const integer
        (
            {"title": "Foo", "const": 0, "type": "integer"},
            "0",
            [("0", True), ("1", False), ("a", False)],
        ),
        # Const float
        (
            {"title": "Foo", "const": 0.2, "type": "float"},
            r"0\.2",
            [("0.2", True), ("032", False)],
        ),
        # Const boolean
        (
            {"title": "Foo", "const": True, "type": "boolean"},
            "true",
            [("true", True), ("True", False)],
        ),
        # Const null
        (
            {"title": "Foo", "const": None, "type": "null"},
            "null",
            [("null", True), ("None", False), ("", False)],
        ),
        # Enum string
        (
            {"title": "Foo", "enum": ["Marc", "Jean"], "type": "string"},
            '("Marc"|"Jean")',
            [('"Marc"', True), ('"Jean"', True), ('"John"', False)],
        ),
        # Make sure strings are escaped with regex and JSON escaping
        (
            {"title": "Foo", "enum": [".*", r"\s*"], "type": "string"},
            r'("\.\*"|"\\\\s\*")',
            [('".*"', True), (r'"\\s*"', True), (r'"\.\*"', False)],
        ),
        # Enum integer
        (
            {"title": "Foo", "enum": [0, 1], "type": "integer"},
            "(0|1)",
            [("0", True), ("1", True), ("a", False)],
        ),
        # Enum mix of types
        (
            {"title": "Foo", "enum": [6, 5.3, "potato", True, None]},
            r'(6|5\.3|"potato"|true|null)',
            [
                ("6", True),
                ("5.3", True),
                ('"potato"', True),
                ("true", True),
                ("null", True),
                ("523", False),
                ("True", False),
                ("None", False),
            ],
        ),
        # integer
        (
            {
                "title": "Foo",
                "type": "object",
                "properties": {"count": {"title": "Count", "type": "integer"}},
                "required": ["count"],
            },
            '\\{[ ]?"count"[ ]?:[ ]?(-)?(0|[1-9][0-9]*)[ ]?\\}',
            [('{ "count": 100 }', True)],
        ),
        # integer with minimum digits
        (
            {
                "title": "Foo",
                "type": "object",
                "properties": {
                    "count": {"title": "Count", "type": "integer", "minimum": 100}
                },
                "required": ["count"],
            },
            '\\{[ ]?"count"[ ]?:[ ]?(([\\d]{4,}|([1-9][0-9]{2})))[ ]?\\}',
            [('{ "count": 10 }', False), ('{ "count": 100 }', True)],
        ),
        # integer with maximum digits
        (
            {
                "title": "Foo",
                "type": "object",
                "properties": {
                    "count": {"title": "Count", "type": "integer", "maximum": 999}
                },
                "required": ["count"],
            },
            '\\{[ ]?"count"[ ]?:[ ]?((-(([\\d]{2,}|([1-9]))))|((0|(([1-9])|([1-9][0-9]{1})|([1-9][0-9]{2})))))[ ]?\\}',
            [('{ "count": 100 }', True), ('{ "count": 1000 }', False)],
        ),
        # integer with minimum and maximum
        (
            {
                "title": "Foo",
                "type": "object",
                "properties": {
                    "count": {
                        "title": "Count",
                        "type": "integer",
                        "minimum": 50,
                        "maximum": 50000,
                    }
                },
                "required": ["count"],
            },
            '\\{[ ]?"count"[ ]?:[ ]?(([5-9][0-9]{1})|([1-9][0-9]{2})|([1-9][0-9]{3})|([1-4][0-9]{4})|(5000[0-0]))[ ]?\\}',
            [
                ('{ "count": 10 }', False),
                ('{ "count": 100 }', True),
                ('{ "count": 10000 }', True),
                ('{ "count": 100000 }', False),
            ],
        ),
        # number
        (
            {
                "title": "Foo",
                "type": "object",
                "properties": {"count": {"title": "Count", "type": "number"}},
                "required": ["count"],
            },
            '\\{[ ]?"count"[ ]?:[ ]?((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?[ ]?\\}',
            [('{ "count": 100 }', True), ('{ "count": 100.5 }', True)],
        ),
        # number with min and max integer digits
        (
            {
                "title": "Foo",
                "type": "object",
                "properties": {
                    "count": {
                        "title": "Count",
                        "type": "number",
                        "minDigitsInteger": 3,
                        "maxDigitsInteger": 5,
                    }
                },
                "required": ["count"],
            },
            '\\{[ ]?"count"[ ]?:[ ]?((-)?(0|[1-9][0-9]{2,4}))(\\.[0-9]+)?([eE][+-][0-9]+)?[ ]?\\}',
            [
                ('{ "count": 10.005 }', False),
                ('{ "count": 100.005 }', True),
                ('{ "count": 10000.005 }', True),
                ('{ "count": 100000.005 }', False),
            ],
        ),
        # number with min and max fraction digits
        (
            {
                "title": "Foo",
                "type": "object",
                "properties": {
                    "count": {
                        "title": "Count",
                        "type": "number",
                        "minDigitsFraction": 3,
                        "maxDigitsFraction": 5,
                    }
                },
                "required": ["count"],
            },
            '\\{[ ]?"count"[ ]?:[ ]?((-)?(0|[1-9][0-9]*))(\\.[0-9]{3,5})?([eE][+-][0-9]+)?[ ]?\\}',
            [
                ('{ "count": 1.05 }', False),
                ('{ "count": 1.005 }', True),
                ('{ "count": 1.00005 }', True),
                ('{ "count": 1.000005 }', False),
            ],
        ),
        # number with min and max exponent digits
        (
            {
                "title": "Foo",
                "type": "object",
                "properties": {
                    "count": {
                        "title": "Count",
                        "type": "number",
                        "minDigitsExponent": 3,
                        "maxDigitsExponent": 5,
                    }
                },
                "required": ["count"],
            },
            '\\{[ ]?"count"[ ]?:[ ]?((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]{3,5})?[ ]?\\}',
            [
                ('{ "count": 1.05e1 }', False),
                ('{ "count": 1.05e+001 }', True),
                ('{ "count": 1.05e-00001 }', True),
                ('{ "count": 1.05e0000001 }', False),
            ],
        ),
        # number with min and max integer, fraction and exponent digits
        (
            {
                "title": "Foo",
                "type": "object",
                "properties": {
                    "count": {
                        "title": "Count",
                        "type": "number",
                        "minDigitsInteger": 3,
                        "maxDigitsInteger": 5,
                        "minDigitsFraction": 3,
                        "maxDigitsFraction": 5,
                        "minDigitsExponent": 3,
                        "maxDigitsExponent": 5,
                    }
                },
                "required": ["count"],
            },
            '\\{[ ]?"count"[ ]?:[ ]?((-)?(0|[1-9][0-9]{2,4}))(\\.[0-9]{3,5})?([eE][+-][0-9]{3,5})?[ ]?\\}',
            [
                ('{ "count": 1.05e1 }', False),
                ('{ "count": 100.005e+001 }', True),
                ('{ "count": 10000.00005e-00001 }', True),
                ('{ "count": 100000.000005e0000001 }', False),
            ],
        ),
        # array
        (
            {"title": "Foo", "type": "array", "items": {"type": "number"}},
            rf"\[{SAFE_WHITESPACE}(({NUMBER})(,{SAFE_WHITESPACE}({NUMBER})){{0,}})?{SAFE_WHITESPACE}\]",
            [("[1e+9,1.3]", True), ("[]", True), ("[1", False)],
        ),
        # array with a set length of 1
        (
            {
                "title": "Foo",
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 1,
                "maxItems": 1,
            },
            rf"\[{SAFE_WHITESPACE}(({INTEGER})(,{SAFE_WHITESPACE}({INTEGER})){{0,0}}){SAFE_WHITESPACE}\]",
            [("[1]", True), ("[1,2]", False), ('["a"]', False), ("[]", False)],
        ),
        # array with a set length greather than 1
        (
            {
                "title": "Foo",
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 3,
                "maxItems": 3,
            },
            rf"\[{SAFE_WHITESPACE}(({INTEGER})(,{SAFE_WHITESPACE}({INTEGER})){{2,2}}){SAFE_WHITESPACE}\]",
            [("[1]", False), ("[]", False), ("[1,2,3]", True), ("[1,2,3,4]", False)],
        ),
        # array with length 0
        (
            {
                "title": "Foo",
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 0,
                "maxItems": 0,
            },
            rf"\[{SAFE_WHITESPACE}\]",
            [("[1]", False), ("[]", True), ("[1,2,3]", False), ("[1,2,3,4]", False)],
        ),
        # object
        (
            {
                "title": "TestSchema",
                "type": "object",
                "properties": {
                    "test_dict": {
                        "title": "Test Dict",
                        "additionalProperties": {"type": "string"},
                        "type": "object",
                    }
                },
                "required": ["test_dict"],
            },
            rf"""\{{{SAFE_WHITESPACE}"test_dict"{SAFE_WHITESPACE}:{SAFE_WHITESPACE}\{{{SAFE_WHITESPACE}({STRING}{SAFE_WHITESPACE}:{SAFE_WHITESPACE}{STRING}({SAFE_WHITESPACE},{SAFE_WHITESPACE}{STRING}{SAFE_WHITESPACE}:{SAFE_WHITESPACE}{STRING}){{0,}})?{SAFE_WHITESPACE}\}}{SAFE_WHITESPACE}\}}""",
            [
                ("""{ "test_dict":{"foo":"bar","baz": "bif"}}""", True),
                ("""{ "test_dict":{"foo":"bar" }}""", True),
                ("""{ "test_dict":{}}""", True),
                ("""{ "WRONG_KEY":{}}""", False),
                ("""{ "test_dict":{"wrong_type" 1}}""", False),
            ],
        ),
        # object containing object
        (
            {
                "title": "TestSchema",
                "type": "object",
                "properties": {
                    "test_dict": {
                        "title": "Test Dict",
                        "additionalProperties": {
                            "additionalProperties": {"type": "integer"},
                            "type": "object",
                        },
                        "type": "object",
                    }
                },
                "required": ["test_dict"],
            },
            rf"""\{{{SAFE_WHITESPACE}"test_dict"{SAFE_WHITESPACE}:{SAFE_WHITESPACE}\{{{SAFE_WHITESPACE}({STRING}{SAFE_WHITESPACE}:{SAFE_WHITESPACE}\{{{SAFE_WHITESPACE}({STRING}{SAFE_WHITESPACE}:{SAFE_WHITESPACE}{INTEGER}({SAFE_WHITESPACE},{SAFE_WHITESPACE}{STRING}{SAFE_WHITESPACE}:{SAFE_WHITESPACE}{INTEGER}){{0,}})?{SAFE_WHITESPACE}\}}({SAFE_WHITESPACE},{SAFE_WHITESPACE}{STRING}{SAFE_WHITESPACE}:{SAFE_WHITESPACE}\{{{SAFE_WHITESPACE}({STRING}{SAFE_WHITESPACE}:{SAFE_WHITESPACE}{INTEGER}({SAFE_WHITESPACE},{SAFE_WHITESPACE}{STRING}{SAFE_WHITESPACE}:{SAFE_WHITESPACE}{INTEGER}){{0,}})?{SAFE_WHITESPACE}\}}){{0,}})?{SAFE_WHITESPACE}\}}{SAFE_WHITESPACE}\}}""",
            [
                (
                    """{"test_dict": {"foo": {"bar": 123, "apple": 99}, "baz": {"bif": 456}}}""",
                    True,
                ),
                (
                    """{"test_dict": {"anykey": {"anykey": 123}, "anykey2": {"bif": 456}}}""",
                    True,
                ),
                ("""{"test_dict": {}}""", True),
                ("""{"test_dict": {"dict of empty dicts are ok": {} }}""", True),
                (
                    """{"test_dict": {"anykey": {"ONLY Dict[Dict]": 123}, "No Dict[int]" 1: }}""",
                    False,
                ),
            ],
        ),
        # oneOf
        (
            {
                "title": "Foo",
                "oneOf": [{"type": "string"}, {"type": "number"}, {"type": "boolean"}],
            },
            rf'((?:"{STRING_INNER}*")|(?:{NUMBER})|(?:{BOOLEAN}))',
            [
                ("12.3", True),
                ("true", True),
                ('"a"', True),
                ("null", False),
                ("", False),
                ("12true", False),
                ('1.3"a"', False),
                ('12.3true"a"', False),
            ],
        ),
        # anyOf
        (
            {
                "title": "Foo",
                "anyOf": [{"type": "string"}, {"type": "integer"}],
            },
            rf"({STRING}|{INTEGER})",
            [("12", True), ('"a"', True), ('1"a"', False)],
        ),
        # allOf
        (
            {
                "title": "Foo",
                "allOf": [{"type": "string"}, {"type": "integer"}],
            },
            rf"({STRING}{INTEGER})",
            [('"a"1', True), ('"a"', False), ('"1"', False)],
        ),
        # Tuple / prefixItems
        (
            {
                "title": "Foo",
                "prefixItems": [{"type": "string"}, {"type": "integer"}],
            },
            rf"\[{SAFE_WHITESPACE}{STRING}{SAFE_WHITESPACE},{SAFE_WHITESPACE}{INTEGER}{SAFE_WHITESPACE}\]",
            [('["a", 1]', True), ('["a", 1, 1]', False), ("[]", False)],
        ),
        # Nested schema
        (
            {
                "title": "Bar",
                "type": "object",
                "properties": {
                    "fuzz": {
                        "title": "Foo",
                        "type": "object",
                        "properties": {"spam": {"title": "Spam", "type": "integer"}},
                        "required": ["spam"],
                    }
                },
                "required": ["fuzz"],
            },
            f'\\{{[ ]?"fuzz"[ ]?:[ ]?\\{{[ ]?"spam"[ ]?:[ ]?{INTEGER}[ ]?\\}}[ ]?\\}}',
            [('{ "fuzz": { "spam": 100 }}', True)],
        ),
        # Schema with a reference
        (
            {
                "title": "User",
                "type": "object",
                "properties": {
                    "user_id": {"title": "User Id", "type": "integer"},
                    "name": {"title": "Name", "type": "string"},
                    "a": {"$ref": "#/properties/name"},
                },
                "required": ["user_id", "name", "a"],
            },
            f'\\{{[ ]?"user_id"[ ]?:[ ]?{INTEGER}[ ]?,[ ]?"name"[ ]?:[ ]?{STRING}[ ]?,[ ]?"a"[ ]?:[ ]?{STRING}[ ]?\\}}',
            [('{"user_id": 100, "name": "John", "a": "Marc"}', True)],
        ),
        (
            {
                "title": "User",
                "type": "object",
                "$defs": {"name": {"title": "Name2", "type": "string"}},
                "properties": {
                    "user_id": {"title": "User Id", "type": "integer"},
                    "name": {"title": "Name", "type": "string"},
                    "name2": {"$ref": "#/$defs/name"},
                },
                "required": ["user_id", "name", "name2"],
            },
            f'\\{{[ ]?"user_id"[ ]?:[ ]?{INTEGER}[ ]?,[ ]?"name"[ ]?:[ ]?{STRING}[ ]?,[ ]?"name2"[ ]?:[ ]?{STRING}[ ]?\\}}',
            [('{"user_id": 100, "name": "John", "name2": "Marc"}', True)],
        ),
        (
            {
                "$id": "customer",
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "title": "Customer",
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "last_name": {"type": "string"},
                    "address": {"$ref": "customer#/$defs/address"},
                },
                "required": [
                    "name",
                    "first_name",
                    "last_name",
                    "address",
                    "shipping_address",
                    "billing_address",
                ],
                "$defs": {
                    "address": {
                        "title": "Address",
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                        "required": ["street_address", "city", "state"],
                        "definitions": {
                            "state": {
                                "type": "object",
                                "title": "State",
                                "properties": {"name": {"type": "string"}},
                                "required": ["name"],
                            }
                        },
                    }
                },
            },
            f'\\{{[ ]?"name"[ ]?:[ ]?{STRING}[ ]?,[ ]?"last_name"[ ]?:[ ]?{STRING}[ ]?,[ ]?"address"[ ]?:[ ]?\\{{[ ]?"city"[ ]?:[ ]?{STRING}[ ]?\\}}[ ]?\\}}',
            [
                (
                    '{"name": "John", "last_name": "Doe", "address": {"city": "Paris"}}',
                    True,
                )
            ],
        ),
        # Optional properties
        # Last required property in first position
        (
            {
                "properties": {
                    "name": {"type": "string"},
                    "age": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "weapon": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["name"],
                "title": "Character",
                "type": "object",
            },
            f'\\{{[ ]?"name"[ ]?:[ ]?{STRING}([ ]?,[ ]?"age"[ ]?:[ ]?({INTEGER}|null))?([ ]?,[ ]?"weapon"[ ]?:[ ]?({STRING}|null))?[ ]?\\}}',
            [
                ('{ "name" : "Player" }', True),
                ('{ "name" : "Player", "weapon" : "sword" }', True),
                ('{ "age" : 10, "weapon" : "sword" }', False),
            ],
        ),
        # Last required property in middle position
        (
            {
                "properties": {
                    "name": {"type": "string"},
                    "age": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "weapon": {"type": "string"},
                    "strength": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                },
                "required": ["name", "weapon"],
                "title": "Character",
                "type": "object",
            },
            f'\\{{[ ]?"name"[ ]?:[ ]?{STRING}[ ]?,([ ]?"age"[ ]?:[ ]?({INTEGER}|null)[ ]?,)?[ ]?"weapon"[ ]?:[ ]?{STRING}([ ]?,[ ]?"strength"[ ]?:[ ]?({INTEGER}|null))?[ ]?\\}}',
            [
                ('{ "name" : "Player" , "weapon" : "sword" }', True),
                (
                    '{ "name" : "Player", "age" : 10, "weapon" : "sword" , "strength" : 10 }',
                    True,
                ),
                ('{ "weapon" : "sword" }', False),
            ],
        ),
        # Last required property in last position
        (
            {
                "properties": {
                    "name": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "age": {"type": "integer"},
                    "armor": {"type": "string"},
                    "strength": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "weapon": {"title": "Weapon", "type": "string"},
                },
                "required": ["age", "armor", "weapon"],
                "title": "Character",
                "type": "object",
            },
            f'\\{{([ ]?"name"[ ]?:[ ]?({STRING}|null)[ ]?,)?[ ]?"age"[ ]?:[ ]?{INTEGER}[ ]?,[ ]?"armor"[ ]?:[ ]?{STRING}[ ]?,([ ]?"strength"[ ]?:[ ]?({INTEGER}|null)[ ]?,)?[ ]?"weapon"[ ]?:[ ]?{STRING}[ ]?\\}}',
            [
                (
                    '{ "name" : "Player", "age" : 10, "armor" : "plate", "strength" : 11, "weapon" : "sword" }',
                    True,
                ),
                ('{ "age" : 10, "armor" : "plate", "weapon" : "sword" }', True),
                (
                    '{ "name" : "Kahlhanbeh", "armor" : "plate", "weapon" : "sword" }',
                    False,
                ),
            ],
        ),
        # All properties are optional
        (
            {
                "properties": {
                    "name": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "age": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "strength": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                },
                "title": "Character",
                "type": "object",
            },
            f'\\{{([ ]?"name"[ ]?:[ ]?({STRING}|null)([ ]?,[ ]?"age"[ ]?:[ ]?({INTEGER}|null))?([ ]?,[ ]?"strength"[ ]?:[ ]?({INTEGER}|null))?|([ ]?"name"[ ]?:[ ]?({STRING}|null)[ ]?,)?[ ]?"age"[ ]?:[ ]?({INTEGER}|null)([ ]?,[ ]?"strength"[ ]?:[ ]?({INTEGER}|null))?|([ ]?"name"[ ]?:[ ]?({STRING}|null)[ ]?,)?([ ]?"age"[ ]?:[ ]?({INTEGER}|null)[ ]?,)?[ ]?"strength"[ ]?:[ ]?({INTEGER}|null))?[ ]?\\}}',
            [
                ('{ "name" : "Player" }', True),
                ('{ "name" : "Player", "age" : 10, "strength" : 10 }', True),
                ('{ "age" : 10, "strength" : 10 }', True),
                ("{ }", True),
            ],
        ),
    ],
)
def test_match(schema, regex, examples):
    interegular.parse_pattern(regex)
    schema = json.dumps(schema)
    test_regex = build_regex_from_schema(
        schema, whitespace_pattern=SAFE_WHITESPACE, safe_subset=False
    )
    assert test_regex == regex

    for string, does_match in examples:
        match = re.fullmatch(test_regex, string)
        if does_match:
            if match is None:
                raise ValueError(f"Expected match for '{string}'")
            assert match[0] == string
            assert match.span() == (0, len(string))
        else:
            assert match is None


@pytest.mark.parametrize(
    "schema,regex,examples",
    [
        # UUID
        (
            {"title": "Foo", "type": "string", "format": "uuid"},
            UUID,
            [
                ("123e4567-e89b-12d3-a456-426614174000", False),
                ('"123e4567-e89b-12d3-a456-426614174000"', True),
                ('"123e4567-e89b-12d3-a456-42661417400"', False),
                ('"123e4567-e89b-12d3-a456-42661417400g"', False),
                ('"123e4567-e89b-12d3-a456-42661417400-"', False),
                ('""', False),
            ],
        ),
        # DATE-TIME
        (
            {"title": "Foo", "type": "string", "format": "date-time"},
            DATE_TIME,
            [
                ("2018-11-13T20:20:39Z", False),
                ('"2018-11-13T20:20:39Z"', True),
                ('"2016-09-18T17:34:02.666Z"', True),
                ('"2008-05-11T15:30:00Z"', True),
                ('"2021-01-01T00:00:00"', True),
                ('"2022-01-10 07:19:30"', False),  # missing T
                ('"2022-12-10T10-04-29"', False),  # incorrect separator
                ('"2023-01-01"', False),
            ],
        ),
        # DATE
        (
            {"title": "Foo", "type": "string", "format": "date"},
            DATE,
            [
                ("2018-11-13", False),
                ('"2018-11-13"', True),
                ('"2016-09-18"', True),
                ('"2008-05-11"', True),
                ('"2015-13-01"', False),  # incorrect month
                ('"2022-01"', False),  # missing day
                ('"2022/12/01"', False),  # incorrect separator"
            ],
        ),
        # TIME
        (
            {"title": "Foo", "type": "string", "format": "time"},
            TIME,
            [
                ("20:20:39Z", False),
                ('"20:20:39Z"', True),
                ('"15:30:00Z"', True),
                ('"25:30:00"', False),  # incorrect hour
                ('"15:30"', False),  # missing seconds
                ('"15:30:00.000"', False),  # missing Z
                ('"15-30-00"', False),  # incorrect separator
                ('"15:30:00+01:00"', False),  # incorrect separator
            ],
        ),
    ],
)
def test_format(schema, regex, examples):
    interegular.parse_pattern(regex)
    schema = json.dumps(schema)
    test_regex = build_regex_from_schema(schema)
    assert test_regex == regex

    for string, does_match in examples:
        match = re.fullmatch(test_regex, string)
        if does_match:
            assert match[0] == string
            assert match.span() == (0, len(string))
        else:
            assert match is None


@pytest.mark.parametrize(
    "schema,examples",
    [
        # NESTED UUID
        (
            {
                "title": "Foo",
                "type": "object",
                "properties": {"uuid": {"type": "string", "format": "uuid"}},
            },
            [
                ('{"uuid": "123e4567-e89b-12d3-a456-426614174000"}', True),
                ('{"uuid":"123e4567-e89b-12d3-a456-42661417400"}', False),
                ('{"uuid":"123e4567-e89b-12d3-a456-42661417400g"}', False),
                ('{"uuid":"123e4567-e89b-12d3-a456-42661417400-"}', False),
                (
                    '{"uuid":123e4567-e89b-12d3-a456-426614174000}',
                    False,
                ),  # missing quotes for value
                ('{"uuid":""}', False),
            ],
        ),
        # NESTED DATE-TIME
        (
            {
                "title": "Foo",
                "type": "object",
                "properties": {"dateTime": {"type": "string", "format": "date-time"}},
            },
            [
                ('{"dateTime": "2018-11-13T20:20:39Z"}', True),
                ('{"dateTime":"2016-09-18T17:34:02.666Z"}', True),
                ('{"dateTime":"2008-05-11T15:30:00Z"}', True),
                ('{"dateTime":"2021-01-01T00:00:00"}', True),
                ('{"dateTime":"2022-01-10 07:19:30"}', False),  # missing T
                ('{"dateTime":"2022-12-10T10-04-29"}', False),  # incorrect separator
                (
                    '{"dateTime":2018-11-13T20:20:39Z}',
                    False,
                ),  # missing quotes for value
                ('{"dateTime":"2023-01-01"}', False),
            ],
        ),
        # NESTED DATE
        (
            {
                "title": "Foo",
                "type": "object",
                "properties": {"date": {"type": "string", "format": "date"}},
            },
            [
                ('{"date": "2018-11-13"}', True),
                ('{"date":"2016-09-18"}', True),
                ('{"date":"2008-05-11"}', True),
                ('{"date":"2015-13-01"}', False),  # incorrect month
                ('{"date":"2022-01"}', False),  # missing day
                ('{"date":"2022/12/01"}', False),  # incorrect separator"
                ('{"date":2018-11-13}', False),  # missing quotes for value
            ],
        ),
        # NESTED TIME
        (
            {
                "title": "Foo",
                "type": "object",
                "properties": {"time": {"type": "string", "format": "time"}},
            },
            [
                ('{"time": "20:20:39Z"}', True),
                ('{"time":"15:30:00Z"}', True),
                ('{"time":"25:30:00"}', False),  # incorrect hour
                ('{"time":"15:30"}', False),  # missing seconds
                ('{"time":"15:30:00.000"}', False),  # missing Z
                ('{"time":"15-30-00"}', False),  # incorrect separator
                ('{"time":"15:30:00+01:00"}', False),  # incorrect separator
                ('{"time":20:20:39Z}', False),  # missing quotes for value
            ],
        ),
        # Unconstrained Object
        (
            {
                "title": "Foo",
                "type": "object",
            },
            [
                ("{}", True),
                ('{"a": 1, "b": null}', True),
                ('{"a": {"z": {"g": 4}}, "b": null}', True),
                ("1234", False),  # not an object
                ('["a", "a"]', False),  # not an array
            ],
        ),
        # Unconstrained Array
        (
            {
                "type": "array",
            },
            [
                ("[1, {}, false]", True),
                ("[{}]", True),
                ('[{"a": {"z": "q"}, "b": null}]', True),
                ('[{"a": [1, 2, true], "b": null}]', True),
                ('[{"a": [1, 2, true], "b": {"a": "b"}}, 1, true, [1, [2]]]', True),
                # too deep, default unconstrained depth limit = 2
                (
                    '[{"a": [1, 2, true], "b": {"a": "b"}}, 1, true, [1, [2, [3]]]]',
                    False,
                ),
                ('[{"a": {"z": {"g": 4}}, "b": null}]', False),
                ("[[[[1]]]]", False),
                # not an array
                ("{}", False),
                ('{"a": 1, "b": null}', False),
                ('{"a": {"z": {"g": 4}}, "b": null}', False),
                ("1234", False),  # not an array
                ('{"a": "a"}', False),  # not an array
            ],
        ),
        # No schema / unconstrained value
        (
            {},
            [
                ('"aaabbuecuh"', True),  # string
                ("5.554", True),  # number
                ("true", True),  # boolean
                ("null", True),  # null
                ("5999", True),  # integer
                ('["a", "b"]', True),  # array
                ('{"key": {"k2": "value"}}', True),  # nested object
                ("this isnt valid json", False),
            ],
        ),
    ],
)
def test_format_without_regex(schema, examples):
    schema = json.dumps(schema)
    test_regex = build_regex_from_schema(schema)
    for string, does_match in examples:
        match = re.fullmatch(test_regex, string)
        if does_match:
            assert match[0] == string
            assert match.span() == (0, len(string))
        else:
            assert match is None


@pytest.mark.parametrize("whitespace_pattern", [None, r"[\n ]*", "abc"])
def test_json_schema_custom_whitespace_pattern(whitespace_pattern):
    """assert whitespace_pattern setting respected"""

    class MockModel(BaseModel):
        foo: int
        bar: str

    schema = json.dumps(MockModel.model_json_schema())

    # assert any ws pattern can be used
    if whitespace_pattern == "abc":
        build_regex_from_schema(schema, whitespace_pattern, safe_subset=False)
        return

    pattern = build_regex_from_schema(schema, whitespace_pattern, safe_subset=False)

    mock_result_mult_ws = (
        """{     "foo"   :   4, \n\n\n   "bar": "baz    baz baz bar"\n\n}"""
    )
    mock_result_maybe_ws = """{"foo" : 4 ,"bar":"baz    baz baz bar"}"""

    match_default_ws = re.fullmatch(pattern, mock_result_maybe_ws)
    if whitespace_pattern is None:
        assert match_default_ws
    else:
        assert re.fullmatch(pattern, mock_result_mult_ws)


def test_one_of_doesnt_produce_illegal_lookaround():
    """Reproduces failure in https://github.com/dottxt-ai/outlines/issues/823"""

    class Cat(BaseModel):
        pet_type: Literal["cat"]
        meows: int

    class Dog(BaseModel):
        pet_type: Literal["dog"]
        barks: float

    class Model(BaseModel):
        pet: Union[Cat, Dog] = Field(..., discriminator="pet_type")
        n: int

    json_schema = Model.schema_json()

    json_schema = Model.schema_json()
    pattern = build_regex_from_schema(json_schema, whitespace_pattern=None)

    # check if the pattern uses lookarounds incompatible with interegular.Pattern.to_fsm()
    interegular.parse_pattern(pattern).to_fsm()
