import json
import re
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
    STRING,
    STRING_INNER,
    TIME,
    UUID,
    WHITESPACE,
    build_regex_from_schema,
    get_schema_from_signature,
    optimize_schema,
    to_regex,
)


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
    regex = to_regex(None, step)
    assert regex == INTEGER

    value = pattern["integer"]
    match = re.fullmatch(regex, value)
    if does_match:
        assert match[0] == value
        assert match.span() == (0, len(value))
    else:
        assert match is None


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
            r'(^"[a-z]"$)',
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
        # Make sure strings are escaped
        (
            {"title": "Foo", "const": ".*", "type": "string"},
            r'"\.\*"',
            [('".*"', True), (r'"\s*"', False), (r'"\.\*"', False)],
        ),
        # Const integer
        (
            {"title": "Foo", "const": 0, "type": "integer"},
            "0",
            [("0", True), ("1", False), ("a", False)],
        ),
        # Enum string
        (
            {"title": "Foo", "enum": ["Marc", "Jean"], "type": "string"},
            '("Marc"|"Jean")',
            [('"Marc"', True), ('"Jean"', True), ('"John"', False)],
        ),
        # Make sure strings are escaped
        (
            {"title": "Foo", "enum": [".*", r"\s*"], "type": "string"},
            r'("\.\*"|"\\s\*")',
            [('".*"', True), (r'"\s*"', True), (r'"\.\*"', False)],
        ),
        # Enum integer
        (
            {"title": "Foo", "enum": [0, 1], "type": "integer"},
            "(0|1)",
            [("0", True), ("1", True), ("a", False)],
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
        # array
        (
            {"title": "Foo", "type": "array", "items": {"type": "number"}},
            rf"\[{WHITESPACE}(({NUMBER})(,{WHITESPACE}({NUMBER})){{0,}})?{WHITESPACE}\]",
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
            rf"\[{WHITESPACE}(({INTEGER})(,{WHITESPACE}({INTEGER})){{0,0}}){WHITESPACE}\]",
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
            rf"\[{WHITESPACE}(({INTEGER})(,{WHITESPACE}({INTEGER})){{2,2}}){WHITESPACE}\]",
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
            rf"\[{WHITESPACE}\]",
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
            rf"""\{{{WHITESPACE}"test_dict"{WHITESPACE}:{WHITESPACE}\{{{WHITESPACE}({STRING}{WHITESPACE}:{WHITESPACE}{STRING}({WHITESPACE},{WHITESPACE}{STRING}{WHITESPACE}:{WHITESPACE}{STRING}){{0,}})?{WHITESPACE}\}}{WHITESPACE}\}}""",
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
            rf"""\{{{WHITESPACE}"test_dict"{WHITESPACE}:{WHITESPACE}\{{{WHITESPACE}({STRING}{WHITESPACE}:{WHITESPACE}\{{{WHITESPACE}({STRING}{WHITESPACE}:{WHITESPACE}{INTEGER}({WHITESPACE},{WHITESPACE}{STRING}{WHITESPACE}:{WHITESPACE}{INTEGER}){{0,}})?{WHITESPACE}\}}({WHITESPACE},{WHITESPACE}{STRING}{WHITESPACE}:{WHITESPACE}\{{{WHITESPACE}({STRING}{WHITESPACE}:{WHITESPACE}{INTEGER}({WHITESPACE},{WHITESPACE}{STRING}{WHITESPACE}:{WHITESPACE}{INTEGER}){{0,}})?{WHITESPACE}\}}){{0,}})?{WHITESPACE}\}}{WHITESPACE}\}}""",
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
            rf"\[{WHITESPACE}{STRING}{WHITESPACE},{WHITESPACE}{INTEGER}{WHITESPACE}\]",
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
    schema = json.dumps(schema)
    test_regex = build_regex_from_schema(schema)
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
        build_regex_from_schema(schema, whitespace_pattern)
        return

    pattern = build_regex_from_schema(schema, whitespace_pattern)

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
    """Reproduces failure in https://github.com/outlines-dev/outlines/issues/823"""

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


@pytest.mark.parametrize(
    "schema,expected_schema",
    [
        # No optimizations possible
        (
            {
                "properties": {"field_a": {"title": "Field A", "type": "integer"}},
                "required": ["field_a"],
                "title": "Test",
                "type": "object",
            },
            {
                "properties": {"field_a": {"title": "Field A", "type": "integer"}},
                "required": ["field_a"],
                "title": "Test",
                "type": "object",
            },
        ),
        # Makes fields with null type in anyOf optional
        # and removes null fields
        (
            {
                "properties": {
                    "field_a": {"title": "Field A", "type": "integer"},
                    "field_b": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                        "title": "Field B",
                    },
                    "field_c": {"title": "Field C", "type": "null"},
                },
                "required": ["field_a", "field_b", "field_c"],
                "title": "Test",
                "type": "object",
            },
            {
                "properties": {
                    "field_a": {"title": "Field A", "type": "integer"},
                    "field_b": {"title": "Field B", "type": "integer"},
                },
                "required": ["field_a"],
                "title": "Test",
                "type": "object",
            },
        ),
        # Multilevel example
        (
            {
                "$defs": {
                    "TestCell": {
                        "properties": {
                            "g": {"title": "G", "type": "integer"},
                            "h": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "title": "H",
                            },
                        },
                        "required": ["g", "h"],
                        "title": "TestCell",
                        "type": "object",
                    },
                    "TestLineItem": {
                        "properties": {
                            "a": {"title": "A", "type": "integer"},
                            "b": {
                                "anyOf": [{"type": "integer"}, {"type": "null"}],
                                "title": "B",
                            },
                            "c": {"title": "C", "type": "string"},
                            "d": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "title": "D",
                            },
                            "e": {"title": "E", "type": "null"},
                            "f": {
                                "anyOf": [{"type": "integer"}, {"type": "string"}],
                                "title": "F",
                            },
                            "i": {"$ref": "#/$defs/TestCell"},
                        },
                        "required": ["a", "b", "c", "d", "e", "f", "i"],
                        "title": "TestLineItem",
                        "type": "object",
                    },
                },
                "properties": {
                    "line_items": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/TestLineItem"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "title": "Line Items",
                    }
                },
                "required": ["line_items"],
                "title": "TestTable",
                "type": "object",
            },
            {
                "$defs": {
                    "TestCell": {
                        "properties": {
                            "g": {"title": "G", "type": "integer"},
                            "h": {"title": "H", "type": "string"},
                        },
                        "required": ["g"],
                        "title": "TestCell",
                        "type": "object",
                    },
                    "TestLineItem": {
                        "properties": {
                            "a": {"title": "A", "type": "integer"},
                            "b": {"title": "B", "type": "integer"},
                            "c": {"title": "C", "type": "string"},
                            "d": {"title": "D", "type": "string"},
                            "f": {
                                "anyOf": [{"type": "integer"}, {"type": "string"}],
                                "title": "F",
                            },
                            "i": {"$ref": "#/$defs/TestCell"},
                        },
                        "required": ["a", "c", "f", "i"],
                        "title": "TestLineItem",
                        "type": "object",
                    },
                },
                "properties": {
                    "line_items": {
                        "title": "Line Items",
                        "items": {"$ref": "#/$defs/TestLineItem"},
                        "type": "array",
                    },
                },
                "required": [],
                "title": "TestTable",
                "type": "object",
            },
        ),
        # From function signature
        (
            {
                "properties": {
                    "a": {"title": "A", "type": "integer"},
                    "b": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                        "title": "B",
                    },
                },
                "required": ["a", "b"],
                "title": "Arguments",
                "type": "object",
            },
            {
                "properties": {
                    "a": {"title": "A", "type": "integer"},
                    "b": {"title": "B", "type": "integer"},
                },
                "required": ["a"],
                "title": "Arguments",
                "type": "object",
            },
        ),
    ],
)
def test_json_schema_optimization(schema: dict, expected_schema: dict):
    optimized_schema = optimize_schema(schema)
    assert optimized_schema == expected_schema
