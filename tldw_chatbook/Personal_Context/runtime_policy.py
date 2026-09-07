"""Encrypted peer-local Personal Context runtime authority."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, StrictBool
from tldw_profile_core.canonical import VersionOne


GLOBAL_POLICY_ID = "personal-context-global-policy"


class AgentAuthority(StrEnum):
    READ_ONLY = "read_only"
    PROPOSE = "propose"
    DIRECT_WRITE = "direct_write"


class GlobalRuntimePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: VersionOne = 1
    enabled: StrictBool = False


class ScopeRuntimePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: VersionOne = 1
    authority: AgentAuthority


class PersonalContextAuthorityError(PermissionError):
    """A path-free fail-closed runtime authority denial."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


_AUTHORITY_RANK = {
    AgentAuthority.READ_ONLY: 1,
    AgentAuthority.PROPOSE: 2,
    AgentAuthority.DIRECT_WRITE: 3,
}


def authority_allows(actual: AgentAuthority, required: AgentAuthority) -> bool:
    return _AUTHORITY_RANK[actual] >= _AUTHORITY_RANK[required]
