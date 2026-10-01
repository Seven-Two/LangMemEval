from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

NonEmpty = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: NonEmpty
    timestamp: int | None = None


class AddRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: NonEmpty
    user_id: NonEmpty
    session_id: NonEmpty
    messages: list[Message] = Field(min_length=1)


class AddResponse(BaseModel):
    success: Literal[True] = True
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: NonEmpty
    user_id: NonEmpty
    top_k: int = Field(ge=1, strict=True)
    options: list[NonEmpty] | None = None


class Evidence(BaseModel):
    id: NonEmpty
    content: NonEmpty
    score: float | None = Field(default=None, allow_inf_nan=False)


class SearchResponse(BaseModel):
    data: list[Evidence]
