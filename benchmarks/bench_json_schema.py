from outlines.caching import cache_disabled
from outlines.fsm.guide import RegexGuide
from outlines.fsm.json_schema import build_regex_from_schema

from .common import setup_tokenizer  # noqa: E402

simple_schema = """{
        "$defs": {
            "Armor": {
                "enum": ["leather", "chainmail", "plate"],
                "title": "Armor",
                "type": "string"
            }
        },
        "properties": {
            "name": {"maxLength": 10, "title": "Name", "type": "string"},
            "age": {"title": "Age", "type": "integer"},
            "armor": {"$ref": "#/$defs/Armor"},
            "strength": {"title": "Strength", "type": "integer"}\
        },
        "required": ["name", "age", "armor", "strength"],
        "title": "Character",
        "type": "object"
    }"""


complex_schema = """{
  "$schema": "http://json-schema.org/draft-04/schema#",
  "title": "Schema for a recording",
  "type": "object",
  "definitions": {
    "artist": {
      "type": "object",
      "properties": {
        "id": {"type": "number"},
        "name": {"type": "string"},
        "functions": {
          "type": "array",
          "items": {"type": "string"}
        }
      },
      "required": ["id", "name", "functions"]
    }
  },
  "properties": {
    "id": {"type": "number"},
    "work": {
      "type": "object",
      "properties": {
        "id": {"type": "number"},
        "name": {"type": "string"},
        "composer": {"$ref": "#/definitions/artist"}
      }
    },
    "recording_artists": {
      "type": "array",
      "items": {"$ref": "#/definitions/artist"}
    }
  },
  "required": ["id", "work", "recording_artists"]
}"""

schemas = dict(simple_schema=simple_schema, complex_schema=complex_schema)


class JsonSchemaBenchmark:
    params = schemas.keys()

    def setup(self, schema_name):
        self.tokenizer = setup_tokenizer()
        self.schema = schemas[schema_name]

    @cache_disabled()
    def time_json_schema_to_regex(self, schema_name):
        build_regex_from_schema(self.schema)

    @cache_disabled()
    def time_json_schema_to_fsm(self, schema_name):
        regex = build_regex_from_schema(self.schema)
        RegexGuide.from_regex(regex, self.tokenizer)
