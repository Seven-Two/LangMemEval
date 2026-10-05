"""Pure A-Mem response policy, independent of SDKs, logging and memory state."""
from dataclasses import dataclass
import json
from typing import Literal


ResponsePurpose = Literal["analysis", "evolution", "query", "generic"]


class InvalidJSONResponse(ValueError):
    """A JSON decoding failure, distinct from transport and schema failures."""


@dataclass(frozen=True)
class PreparedResponse:
    data: dict
    ignored_extra_fields: int = 0


def validate(value, schema, path="$"):
    kind = schema.get("type")
    valid = {"object": isinstance(value, dict), "array": isinstance(value, list),
             "string": isinstance(value, str), "boolean": type(value) is bool,
             "integer": type(value) is int}
    if kind in valid and not valid[kind]:
        raise ValueError(f"A-Mem response field {path} must have type {kind}")
    if kind == "object":
        missing = [k for k in schema.get("required", []) if k not in value]
        if missing:
            raise ValueError(f"A-Mem response field {path} is missing required fields: {', '.join(missing)}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            raise ValueError("A-Mem response contains unknown fields")
        for key, item in value.items():
            if key in properties:
                validate(item, properties[key], f"{path}.{key}")
    elif kind == "array":
        for index, item in enumerate(value):
            validate(item, schema["items"], f"{path}[{index}]")


def prepare_response(value, schema, *, purpose: ResponsePurpose = "generic") -> PreparedResponse:
    """Validate consumed fields. Purpose is explicit, never inferred from schema equality."""
    if not isinstance(value, dict):
        validate(value, schema)
        return PreparedResponse(value)
    properties = schema.get("properties", {})
    extra_count = len(set(value) - set(properties))
    if purpose == "evolution":
        validate(value.get("should_evolve"), properties["should_evolve"], "$.should_evolve")
        fields = ["should_evolve"]
        if value["should_evolve"]:
            validate(value.get("actions"), properties["actions"], "$.actions")
            fields.append("actions")
            if "strengthen" in value["actions"]:
                fields.extend(["suggested_connections", "tags_to_update"])
            if "update_neighbor" in value["actions"]:
                fields.extend(["new_context_neighborhood", "new_tags_neighborhood"])
        runtime_schema = {"type": "object", "properties": {k: properties[k] for k in fields},
                          "required": fields, "additionalProperties": False}
    else:
        fields, runtime_schema = list(properties), schema
    result = {key: value[key] for key in fields if key in value}
    validate(result, runtime_schema)
    return PreparedResponse(result, extra_count)


def decode_response(content, *, purpose: ResponsePurpose = "generic"):
    if not isinstance(content, str) or (purpose != "evolution" and not content.strip()):
        raise ValueError("A-Mem received an empty structured response")
    content = content.strip()
    if content.startswith("```") and content.endswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    if purpose == "evolution":
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end != -1:
            content = content[start:end + 1]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        raise InvalidJSONResponse("A-Mem received invalid JSON") from None
