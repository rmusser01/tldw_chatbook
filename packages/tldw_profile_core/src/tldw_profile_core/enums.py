from enum import StrEnum


class RecordKind(StrEnum):
    PREFERENCE = "preference"
    FACT = "fact"
    GOAL = "goal"
    WORKING_CONTEXT = "working_context"
    LEGACY_UNCLASSIFIED = "legacy_unclassified"


class RecordState(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    EXPIRED = "expired"


class ScopeKind(StrEnum):
    GLOBAL = "global"
    WORKSPACE = "workspace"


class SyncMode(StrEnum):
    DEVICE_ONLY = "device_only"
    SYNCABLE = "syncable"


class AgentVisibility(StrEnum):
    AGENT_VISIBLE = "agent_visible"
    USER_ONLY = "user_only"


class ProposalState(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ProposalOperation(StrEnum):
    CREATE = "create"
    UPDATE = "update"
    ARCHIVE = "archive"


class ToolOperation(StrEnum):
    SEARCH = "search"
    GET = "get"
    PROPOSE = "propose"
    UPDATE = "update"
    PROMOTE = "promote"
    DELETE = "delete"
    PURGE = "purge"
    PRIVACY_CONTROL = "privacy_control"
    CROSS_WORKSPACE = "cross_workspace"
