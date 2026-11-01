"""Opaque identities for routing to an existing Console execution owner."""

import re

from pydantic import BaseModel, ConfigDict, field_validator

from tldw_chatbook.Canvas.limits import validate_opaque_identifier


class ConsoleHostIdentity(BaseModel):
    """One retained child, scoped to its parent server lifetime."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    server_epoch: str
    console_id: str

    @field_validator("*", mode="before")
    @classmethod
    def validate_identity(cls, value: object) -> object:
        """Reject paths/display labels as well as malformed opaque identities."""
        if value is not None:
            validate_opaque_identifier(value)
            if re.fullmatch(r"[A-Za-z0-9_:.-]+", value) is None or value in {".", ".."}:
                raise ValueError("identity must be opaque")
        return value


class ConsoleTarget(ConsoleHostIdentity):
    """Exact live chat incarnation and optional native work identities."""

    session_id: str
    incarnation_id: str
    turn_id: str | None = None
    run_id: str | None = None
    handle_id: str | None = None
    round_id: str | None = None
