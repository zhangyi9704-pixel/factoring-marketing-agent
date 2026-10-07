"""Explicit human input; filesystem locations are never accepted."""
from typing import Literal
from datetime import date
from pydantic import BaseModel, ConfigDict, Field, field_validator


class VersionRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_id: str = Field(min_length=1, max_length=64)
    version: str = Field(min_length=1, max_length=64)


class FeedbackInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["adopt", "supplement"]
    content: str = Field(min_length=1, max_length=20000)
    source: str = Field(min_length=1, max_length=512)
    category: Literal["product", "industry", "compliance", "process"]
    industry: Literal["all", "基建", "医药", "电商"]
    topic: str = Field(min_length=1, max_length=128)
    effective_date: date
    doc_id: str | None = Field(default=None, min_length=1, max_length=64)
    adopted: VersionRef | None = None

    @field_validator("content", "source", "topic")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("nonblank_required")
        return value.strip()


class DraftConfirmation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    content_confirmed: Literal[True]
    source_confirmed: Literal[True]
