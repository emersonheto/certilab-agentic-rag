from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


RoleName = Literal["admin", "technician", "client"]


class AskRequest(BaseModel):
    """Public question payload.

    Caller identity is intentionally not accepted here. The API derives it from
    the demo authentication adapter so tenant isolation cannot be bypassed by
    request-body fields.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    question: str = Field(min_length=3, max_length=500)


class Source(BaseModel):
    certificate_id: int | None
    code: str | None
    customer_id: int | None
    source_type: str
    source_id: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    route: Literal["structured", "semantic", "combined", "web_search"]
    sources: list[Source]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    environment: str
