import json
import string

import pytest
from pydantic import BaseModel, ValidationError

from outlines import generate


class MockCharacterTokenizer:
    def __init__(self):
        characters = set(
            string.ascii_letters
            + string.digits
            + string.punctuation
            + string.whitespace
        )
        self.vocabulary = {tok: tok_id for tok_id, tok in enumerate(characters)}
        self.vocabulary["eos"] = len(characters)
        self.special_tokens = {"eos"}
        self.eos_token_id = len(characters)

    def convert_token_to_string(self, token):
        return token


class MockModel:
    def __init__(self, generated):
        self.generated = generated
        self.tokenizer = MockCharacterTokenizer()

    def generate(self, *args, **kwargs):
        return self.generated


mock_json_schema = json.dumps(
    {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
        "additionalProperties": False,
    }
)


class MockPydanticModel(BaseModel):
    message: str


@pytest.mark.parametrize("schema", [mock_json_schema, MockPydanticModel])
def test_generate_strict_success(schema):
    model = MockModel(generated='{"message": "foo"}')
    generator = generate.json(model, schema)
    generator("hi")


@pytest.mark.parametrize("schema", [mock_json_schema, MockPydanticModel])
def test_generate_strict_success_batch(schema):
    model = MockModel(
        generated=[
            '{"message": "foo"}',
            '{"message": "basteuhotuhnoethunoteuhntoeuhntoehuotn"}',
        ]
    )
    generator = generate.json(model, schema)
    for output in generator("hi"):
        pass


@pytest.mark.parametrize("schema", [mock_json_schema, MockPydanticModel])
def test_generate_strict_fail(schema):
    model = MockModel(generated='{"message": "foo')
    generator = generate.json(model, schema)
    with pytest.raises((json.decoder.JSONDecodeError, ValidationError)):
        generator("hi")


@pytest.mark.parametrize("schema", [mock_json_schema, MockPydanticModel])
def test_generate_strict_fail_batch(schema):
    model = MockModel(
        generated=[
            '{"message": "foo"}',
            '{"message": "basteuhotuhnoethunoteuhntoeuhntoehuotn"',
        ]
    )
    generator = generate.json(model, schema)
    with pytest.raises((json.decoder.JSONDecodeError, ValidationError)):
        generator("hi")


@pytest.mark.parametrize("schema", [mock_json_schema, MockPydanticModel])
def test_generate_non_strict_evade_failure(schema):
    model = MockModel(generated='{"message": "foo')
    generator = generate.json(model, schema, strict=False)
    result = generator("hi")
    assert result["error_type"] in ("JSONDecodeError", "ValidationError")
    assert result["output"] == model.generated


@pytest.mark.parametrize("schema", [mock_json_schema, MockPydanticModel])
def test_generate_non_strict_evade_failure_batch(schema):
    model = MockModel(
        generated=[
            '{"message": "foo"}',
            '{"message": "basteuhotuhnoethunoteuhntoeuhntoehuotn"',
        ]
    )
    generator = generate.json(model, schema, strict=False)
    result = generator("hi")
    if isinstance(schema, str):
        assert result[0] == json.loads(model.generated[0])
    else:
        assert result[0] == schema.parse_raw(model.generated[0])
    assert result[1]["error_type"] in ("JSONDecodeError", "ValidationError")
    assert result[1]["output"] == model.generated[1]
