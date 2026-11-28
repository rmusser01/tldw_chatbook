"""Qualified navigation context shared by Research Workspace and Runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime

from .chat_models import MAX_CHAT_SOURCES, ResearchSourceVersionRef, validate_chat_id
from .contracts import QualifiedWorkspaceRef, WorkspaceDataSource

MAX_RESEARCH_QUERY_CHARS = 4000
RESEARCH_WORKSPACE_RETURN_ROUTE = "research_workspace"

_OUTPUT_OWNERS = {
    WorkspaceDataSource.LOCAL: {"local_chatbook", "local_study_deck", "local_quiz"},
    WorkspaceDataSource.SERVER: {"server_artifact", "server_study_deck", "server_quiz"},
}


def _qualified_origin(origin: QualifiedWorkspaceRef) -> None:
    if not isinstance(origin, QualifiedWorkspaceRef):
        raise TypeError("Deep Research requires a qualified origin workspace")
    validate_chat_id(origin.workspace_id, "workspace ID")
    if origin.data_source is WorkspaceDataSource.SERVER:
        validate_chat_id(origin.server_profile_id, "server profile ID")
        validate_chat_id(origin.principal_id, "server account ID")


@dataclass(frozen=True, slots=True)
class ResearchWorkspaceRunRef:
    """One backend-owned run and its exact workspace launch identity."""

    origin: QualifiedWorkspaceRef
    launch_id: str
    run_id: str

    def __post_init__(self) -> None:
        _qualified_origin(self.origin)
        for name in ("launch_id", "run_id"):
            object.__setattr__(self, name, validate_chat_id(getattr(self, name), name))

    def provenance(self) -> dict[str, str]:
        return {
            "backend": self.origin.data_source.value,
            "workspace_id": self.origin.workspace_id,
            "launch_id": self.launch_id,
            "run_id": self.run_id,
        }


@dataclass(frozen=True, slots=True)
class ResearchLaunchOutputRef:
    owner_kind: str
    owner_id: str
    owner_version: int | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "owner_id", validate_chat_id(self.owner_id, "output ID")
        )
        if self.owner_version is not None and (
            type(self.owner_version) is not int or self.owner_version < 0
        ):
            raise ValueError("The initiating output version is invalid")


@dataclass(frozen=True, slots=True)
class ResearchLaunchContext:
    """Origin and selected versions captured when a workspace launches Runs."""

    origin: QualifiedWorkspaceRef
    launch_id: str
    source_snapshot: tuple[ResearchSourceVersionRef, ...]
    query: str = field(repr=False)
    created_at: str
    initiating_output: ResearchLaunchOutputRef | None = None
    initiating_message_id: str = ""
    local_conversation_id: str = ""
    server_chat_id: str = ""
    return_route: str = RESEARCH_WORKSPACE_RETURN_ROUTE

    def __post_init__(self) -> None:
        _qualified_origin(self.origin)
        if self.return_route != RESEARCH_WORKSPACE_RETURN_ROUTE:
            raise ValueError("The Deep Research return route is invalid")
        for name in (
            "launch_id",
            "local_conversation_id",
            "server_chat_id",
            "initiating_message_id",
        ):
            value = validate_chat_id(
                getattr(self, name), name, allow_empty=name != "launch_id"
            )
            object.__setattr__(self, name, value)
        local = self.origin.data_source is WorkspaceDataSource.LOCAL
        if (local and self.server_chat_id) or (
            not local and self.local_conversation_id
        ):
            raise ValueError(
                "The conversation belongs to a different workspace authority"
            )
        if self.initiating_message_id and (
            self.initiating_output is not None
            or not (self.local_conversation_id or self.server_chat_id)
        ):
            raise ValueError("The initiating message requires its own conversation")
        if self.initiating_output is not None and (
            not isinstance(self.initiating_output, ResearchLaunchOutputRef)
            or self.initiating_output.owner_kind
            not in _OUTPUT_OWNERS[self.origin.data_source]
        ):
            raise ValueError("The initiating output belongs to a different authority")
        sources = self.source_snapshot
        if not isinstance(sources, tuple) or not 1 <= len(sources) <= MAX_CHAT_SOURCES:
            raise ValueError(
                f"Select between 1 and {MAX_CHAT_SOURCES} sources for Deep Research"
            )
        if any(
            not isinstance(row, ResearchSourceVersionRef) or row.ref != self.origin
            for row in sources
        ):
            raise ValueError("A Deep Research source belongs to a different workspace")
        if (
            len({row.source_id for row in sources}) != len(sources)
            or len({row.catalog_item_id for row in sources}) != len(sources)
            or not any(row.ready_modes for row in sources)
        ):
            raise ValueError("Select distinct sources with at least one source ready")
        if not isinstance(self.query, str) or "\x00" in self.query:
            raise ValueError("Enter a valid Deep Research question")
        query = " ".join(self.query.split())
        if not 1 <= len(query) <= MAX_RESEARCH_QUERY_CHARS:
            raise ValueError(
                f"Enter a question of at most {MAX_RESEARCH_QUERY_CHARS} characters"
            )
        object.__setattr__(self, "query", query)
        if (
            not isinstance(self.created_at, str)
            or len(self.created_at) > 64
            or datetime.fromisoformat(self.created_at).utcoffset() is None
        ):
            raise ValueError(
                "A Deep Research launch must have a timestamp with time zone"
            )
        created = datetime.fromisoformat(self.created_at).astimezone(UTC)
        object.__setattr__(
            self, "created_at", created.isoformat().replace("+00:00", "Z")
        )

    def target(self, run_id: str) -> ResearchWorkspaceRunRef:
        return ResearchWorkspaceRunRef(self.origin, self.launch_id, run_id)

    def to_payload(self) -> dict:
        payload = asdict(self)
        rows = []
        for source in payload["source_snapshot"]:
            del source["ref"]
            source["ready_modes"] = list(source["ready_modes"])
            rows.append(source)
        payload["source_snapshot"] = rows
        return payload

    def server_payload(self) -> dict:
        """Send the workspace-owned references without the device's server profile."""
        if self.origin.data_source is not WorkspaceDataSource.SERVER:
            raise ValueError("Choose a Server workspace for this launch")
        payload = self.to_payload()
        for name in ("origin", "local_conversation_id", "return_route"):
            del payload[name]
        return payload

    @classmethod
    def from_server_projection(
        cls, origin: QualifiedWorkspaceRef, raw: object
    ) -> ResearchLaunchContext:
        if (
            origin.data_source is not WorkspaceDataSource.SERVER
            or not isinstance(raw, dict)
            or set(raw) != {"workspace_id", "owner_user_id", "launch"}
            or raw["workspace_id"] != origin.workspace_id
            or not isinstance(raw["owner_user_id"], str)
            or not raw["owner_user_id"]
            or not isinstance(raw["launch"], dict)
        ):
            raise ValueError(
                "The Server run belongs to a different workspace or account"
            )
        if origin.principal_id.startswith("jwt-sub:") and (
            origin.principal_id.removeprefix("jwt-sub:") != raw["owner_user_id"]
        ):
            raise ValueError("The Server run belongs to a different account")
        return cls.from_payload(
            {
                **raw["launch"],
                "origin": asdict(origin),
                "local_conversation_id": "",
                "return_route": RESEARCH_WORKSPACE_RETURN_ROUTE,
            }
        )

    @classmethod
    def from_payload(cls, raw: object) -> ResearchLaunchContext:
        if not isinstance(raw, dict) or set(raw) != {item.name for item in fields(cls)}:
            raise ValueError("The Deep Research launch has unsupported fields")
        payload = dict(raw)
        origin = QualifiedWorkspaceRef(**payload.pop("origin"))
        sources = payload.pop("source_snapshot")
        if not isinstance(sources, list) or len(sources) > MAX_CHAT_SOURCES:
            raise ValueError("The Deep Research source snapshot is invalid")
        output = payload.pop("initiating_output")
        return cls(
            origin=origin,
            source_snapshot=tuple(
                ResearchSourceVersionRef(ref=origin, **row) for row in sources
            ),
            initiating_output=ResearchLaunchOutputRef(**output)
            if output is not None
            else None,
            **payload,
        )
