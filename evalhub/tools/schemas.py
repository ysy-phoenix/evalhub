import json
from typing import Any, Literal

from pydantic import BaseModel, Field


# adapted from verl.tools.schemas
class OpenAIFunctionPropertySchema(BaseModel):
    r"""The schema of a parameter in OpenAI format."""

    type: str
    description: str | None = None
    enum: list[str] | None = None


class OpenAIFunctionParametersSchema(BaseModel):
    r"""The schema of parameters in OpenAI format."""

    type: str
    properties: dict[str, OpenAIFunctionPropertySchema]
    required: list[str]


class OpenAIFunctionSchema(BaseModel):
    r"""The schema of a function in OpenAI format."""

    name: str
    description: str
    parameters: OpenAIFunctionParametersSchema
    strict: bool = False


class OpenAIFunctionToolSchema(BaseModel):
    r"""The schema of a tool in OpenAI format."""

    type: str
    function: OpenAIFunctionSchema


class OpenAIFunctionParsedSchema(BaseModel):
    r"""The parsed schema of a tool in OpenAI format."""

    name: str
    arguments: str  # JSON string


class OpenAIFunctionCallSchema(BaseModel):
    r"""The parsed schema of a tool in OpenAI format."""

    name: str
    arguments: dict[str, Any]

    @staticmethod
    def from_openai_function_parsed_schema(
        parsed_schema: OpenAIFunctionParsedSchema,
    ) -> tuple["OpenAIFunctionCallSchema", bool]:
        has_decode_error = False
        try:
            arguments = json.loads(parsed_schema.arguments)
        except json.JSONDecodeError:
            arguments = {}
            has_decode_error = True
        # If the arguments is not a dict, it means the arguments is not a valid JSON string
        if not isinstance(arguments, dict):
            arguments = {}
            has_decode_error = True

        return OpenAIFunctionCallSchema(
            name=parsed_schema.name, arguments=arguments
        ), has_decode_error


class OpenAIFunctionToolCall(BaseModel):
    r"""The tool call in OpenAI format."""

    id: str
    type: Literal["function"] = "function"
    function: OpenAIFunctionCallSchema


# adapted from sglang.srt.openai_api.protocol
class Function(BaseModel):
    r"""Function descriptions."""

    description: str | None = Field(default=None, examples=[None])
    name: str | None = None
    parameters: object | None = None
    strict: bool = False


class Tool(BaseModel):
    r"""Function wrapper."""

    type: str = Field(default="function", examples=["function"])
    function: Function
