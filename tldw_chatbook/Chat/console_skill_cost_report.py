"""Bounded, content-free skill observations for a live Console session."""

from __future__ import annotations

import json
import re
import time
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from hashlib import sha256
from threading import Lock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tldw_chatbook.Agents.agent_models import ToolSchema

    from .console_prepared_request import (
        ConsoleRequestTokenAccounting,
        PreparedProviderRequest,
    )

SkillReportScope = tuple[str, int, str]
SchemaWeights = tuple[tuple[str | None, int], ...]
_MAX_SKILLS = 64
_RECENT_TURNS = 128
_SKILL_NAME = re.compile(r"[A-Za-z0-9_:.-]{1,128}")
_LOCK = Lock()


@dataclass(frozen=True, slots=True)
class _Cost:
    at: float
    prepared: bool
    accounting: ConsoleRequestTokenAccounting
    native: bool
    schemas: int | None
    skills: tuple[tuple[str, int], ...]
    conflicts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Use:
    count: int
    last: float
    body_tokens: int
    unmeasured: int


@dataclass(slots=True)
class _Observation:
    scope: SkillReportScope
    since: float
    cost: _Cost | None = None
    usage: dict[str, _Use] = field(default_factory=dict)
    turns: deque[str] = field(default_factory=lambda: deque(maxlen=_RECENT_TURNS))
    capped: bool = False
    base_system_tokens: int | None = None


@dataclass(slots=True)
class SkillChildObservation:
    """Match the existing child request against the rendered skill's prefix."""

    size: int
    digest: bytes
    on_prepared: Callable[[int | None], None] | None = None
    prepared: bool = False

    def observe(self, request: PreparedProviderRequest) -> None:
        if self.prepared:
            return
        self.prepared = True
        body_tokens = None
        for row in request.semantic.active_request:
            content = row.get("content")
            if (
                row.get("role") == "user"
                and isinstance(content, str)
                and len(content) >= self.size
                and skill_body_digest(content[: self.size]) == self.digest
            ):
                body_tokens = (self.size + 3) // 4
                break
        if self.on_prepared is not None:
            self.on_prepared(body_tokens)


def skill_body_digest(text: str) -> bytes:
    """Fingerprint ephemeral text without retaining it in the observation."""
    return sha256(text.encode("utf-8", errors="surrogatepass")).digest()


SKILL_CHILD_OBSERVATION: ContextVar[SkillChildObservation | None] = ContextVar(
    "console_skill_child_observation", default=None
)


def skill_report_scope(session: Any) -> SkillReportScope:
    """Identify one live workspace and conversation binding."""
    return (
        session.incarnation_id,
        session.conversation_binding_revision,
        session.workspace_id,
    )


def _observation(session: Any, scope: SkillReportScope, now: float) -> _Observation:
    state = getattr(session, "_skill_cost_observation", None)
    if state is None or state.scope != scope:
        state = _Observation(scope, now)
        session._skill_cost_observation = state
    return state


def _safe_name(name: object) -> str | None:
    return name if isinstance(name, str) and _SKILL_NAME.fullmatch(name) else None


def schema_cost_weights(
    schemas: Sequence[ToolSchema], skill_names: Mapping[str, str]
) -> SchemaWeights | None:
    """Approximate relative schema contributions from one selected plan."""
    if len(schemas) > _MAX_SKILLS:
        return None
    if len({schema.name.casefold() for schema in schemas}) != len(schemas):
        return None
    weights = []
    for schema in schemas:
        name = _safe_name(skill_names.get(schema.id))
        if schema.id in skill_names and name is None:
            return None
        # Serialized lengths are weights, never reported as provider token counts.
        payload_size = (
            len(schema.name)
            + len(schema.description)
            + len(
                json.dumps(schema.parameters, ensure_ascii=False, separators=(",", ":"))
            )
        )
        weights.append((name, max(1, payload_size)))
    return tuple(weights)


def record_skill_cost(
    session: Any,
    scope: SkillReportScope,
    accounting: ConsoleRequestTokenAccounting,
    *,
    native: bool,
    schema_weights: SchemaWeights | None = None,
    fenced_tokens: int | None = None,
    prepared: bool = False,
    conflicts: Sequence[str] = (),
    now: float | None = None,
) -> None:
    """Record an estimate partitioned inside one original accounting bucket."""
    if now is None:
        now = time.monotonic()
    budget = accounting.tool_schema_tokens if native else fenced_tokens
    budget = (
        None
        if budget is None
        else max(
            0,
            min(
                budget,
                accounting.tool_schema_tokens if native else accounting.system_tokens,
            ),
        )
    )
    skills: list[tuple[str, int]] = []
    if schema_weights is None or budget and not schema_weights:
        budget = None
    elif budget is not None and schema_weights:
        total_weight = sum(weight for _, weight in schema_weights)
        if total_weight > 0:
            cumulative = 0
            for name, weight in schema_weights:
                share = budget * (cumulative + weight) // total_weight
                share -= budget * cumulative // total_weight
                cumulative += weight
                if name is not None:
                    skills.append((name, share))
        else:
            budget = None
    sample = _Cost(
        now,
        prepared,
        accounting,
        native,
        budget,
        tuple(skills),
        tuple(name for item in conflicts if (name := _safe_name(item)))[:_MAX_SKILLS],
    )
    with _LOCK:
        if skill_report_scope(session) != scope:
            return
        state = _observation(session, scope, now)
        if not prepared:
            state.base_system_tokens = accounting.system_tokens - getattr(
                accounting, "system_prompt_delta_tokens", 0
            )
        elif state.base_system_tokens is not None:
            # Keep the Context forecast's signed rendering adjustment while
            # presenting the final prepared request's actual categories.
            sample = replace(
                sample,
                accounting=replace(
                    accounting,
                    system_prompt_delta_tokens=accounting.system_tokens
                    - state.base_system_tokens,
                ),
            )
        state.cost = sample


def skill_report_accounting(session: Any) -> ConsoleRequestTokenAccounting | None:
    """Return the same live request accounting projected by the report."""
    with _LOCK:
        state = getattr(session, "_skill_cost_observation", None)
        if (
            state is not None
            and state.scope == skill_report_scope(session)
            and state.cost is not None
        ):
            return state.cost.accounting
    return None


def observe_skill_use(
    session: Any,
    scope: SkillReportScope,
    names: Sequence[str],
    *,
    turn_id: str | None = None,
    body_tokens: int | None = None,
    now: float | None = None,
) -> None:
    """Count accepted skill uses, once per primary turn and skill."""
    names = tuple(dict.fromkeys(name for name in names if _safe_name(name) is not None))
    if not names:
        return
    if now is None:
        now = time.monotonic()
    with _LOCK:
        if skill_report_scope(session) != scope:
            return
        state = _observation(session, scope, now)
        if turn_id is not None:
            if turn_id in state.turns:
                return
            state.turns.append(turn_id)
        for name in names:
            previous = state.usage.get(name)
            if previous is None:
                if len(state.usage) >= _MAX_SKILLS:
                    state.capped = True
                    continue
                previous = _Use(0, now, 0, 0)
            state.usage[name] = _Use(
                previous.count + 1,
                max(now, previous.last),
                previous.body_tokens + (body_tokens or 0),
                previous.unmeasured + (body_tokens is None),
            )


def format_skill_cost_report(
    session: Any,
    catalog: Mapping[str, Any],
    *,
    disabled_builtins: Sequence[str] = (),
    mcp_enabled: bool | None = None,
    plugin_components: Sequence[Mapping[str, Any]] | None = (),
    now: float | None = None,
) -> str:
    """Render a live snapshot with no reads from stored history or providers."""
    if now is None:
        now = time.monotonic()

    def age(at: float) -> str:
        seconds = max(0, int(now - at))
        return (
            f"{seconds // 60}m {seconds % 60}s ago"
            if seconds >= 60
            else f"{seconds}s ago"
        )

    with _LOCK:
        state = getattr(session, "_skill_cost_observation", None)
        if state is not None and state.scope != skill_report_scope(session):
            state = None
        cost = state.cost if state is not None else None
        usage = dict(state.usage) if state is not None else {}
        since = state.since if state is not None else None
        capped = bool(state is not None and state.capped)

    available = catalog.get("available_skills") or ()
    blocked = catalog.get("blocked_skills") or ()
    active_plugin_skills = tuple(
        row
        for row in (plugin_components or ())
        if row.get("plugin_available") and row.get("plugin_kind") == "skill"
    )
    available = (*available, *active_plugin_skills)
    rows = [row for row in (*available, *blocked) if isinstance(row, Mapping)]
    counts = Counter(
        name.casefold() for row in rows if (name := _safe_name(row.get("name")))
    )
    catalog_names = {
        name
        for row in available
        if isinstance(row, Mapping) and (name := _safe_name(row.get("name")))
    }

    lines = ["Current Console session · Input token estimates"]
    advertised: dict[str, int] = {}
    if cost is None:
        lines.append("Prepared request costs unavailable.")
    else:
        accounting = cost.accounting
        source = "Prepared request" if cost.prepared else "Context preflight"
        lines.append(
            f"{source} ({age(cost.at)}): {accounting.total_input_tokens:,} tokens"
        )
        if cost.schemas is None:
            lines.append(
                f"Tool schemas: {accounting.tool_schema_tokens:,} tokens · skill attribution unavailable"
            )
            lines.append(f"System prompt: {accounting.system_tokens:,} tokens")
        else:
            advertised = dict(cost.skills)
            skills = sum(advertised.values())
            lines.append(f"Advertised skills: ~{skills:,} tokens")
            lines.append(
                f"Other tool schemas and framing: ~{cost.schemas - skills:,} tokens"
            )
            system = (
                accounting.system_tokens
                if cost.native
                else accounting.system_tokens - cost.schemas
            )
            lines.append(f"Remaining system prompt: {system:,} tokens")
            if not cost.native and accounting.tool_schema_tokens:
                lines.append(
                    f"Native tools (unattributed): {accounting.tool_schema_tokens:,} tokens"
                )

            lines.append("Schema shares are estimates from this request sample.")
        remainder = (
            accounting.total_input_tokens
            - accounting.system_tokens
            - accounting.tool_schema_tokens
        )
        lines.append(f"Other request categories: {remainder:,} tokens")

    lines.append("")
    lines.append(
        "Skill usage unknown · no session observations yet."
        if since is None
        else f"Accepted uses since {age(since)} · earlier history unknown"
    )
    if capped:
        lines.append("Observation limit reached · additional skills may be missing.")
    names = sorted(set(advertised) | set(usage))
    remaining = sorted(catalog_names.difference(names))
    if len(remaining) > _MAX_SKILLS:
        lines.append("Catalog listing limited to 64 additional skills.")
    names = sorted((*names, *remaining[:_MAX_SKILLS]), key=str.casefold)
    for name in names:
        fields = []
        if name in advertised:
            fields.append(f"~{advertised[name]:,} advertised tokens")
        elif cost is not None and cost.schemas is not None:
            fields.append("no advertised schema in sample")
        if (use := usage.get(name)) is not None:
            fields.extend((f"{use.count} observed", f"last used {age(use.last)}"))
            if use.count > use.unmeasured:
                fields.append(
                    f"~{use.body_tokens:,} tokens of loaded child skill text (separate)"
                )
            if use.unmeasured:
                fields.append(
                    f"loaded-body attribution unavailable for {use.unmeasured} use(s)"
                )
        else:
            fields.append(
                "0 observed · earlier usage unknown"
                if since is not None
                else "usage unknown"
            )
        lines.append(f"{name}: " + " · ".join(fields))

    conflicts = {name for name, count in counts.items() if count > 1}
    if cost is not None:
        conflicts.update(cost.conflicts)
    overrides = [
        name
        for row in rows
        if row.get("overrides_builtin") and (name := _safe_name(row.get("name")))
    ]
    blocked_names = [
        name
        for row in blocked
        if isinstance(row, Mapping) and (name := _safe_name(row.get("name")))
    ]
    disabled_names = {
        name
        for row in available
        if isinstance(row, Mapping)
        and (name := _safe_name(row.get("name")))
        and (
            row.get("builtin_disabled")
            or row.get("enabled") is False
            or (
                row.get("disable_model_invocation")
                and row.get("user_invocable") is False
            )
        )
    }
    disabled_names.update(
        name for item in disabled_builtins if (name := _safe_name(item))
    )
    lines.append("")
    for label, values in (
        ("Conflicting names", sorted(conflicts)),
        ("User skills overriding built-ins", overrides),
        ("Blocked skills", blocked_names),
        ("Unavailable or disabled skills", sorted(disabled_names)),
    ):
        if values:
            lines.append(
                f"{label}: {', '.join(values[:_MAX_SKILLS])} · review in Library ▸ Skills"
            )
    if plugin_components is None:
        lines.append("Plugin inventory unavailable · review in Plugins")
    else:
        unavailable_components = {
            name
            for row in plugin_components
            if row.get("plugin_selected")
            and not row.get("plugin_available")
            and (name := _safe_name(row.get("name")))
        }
        disabled_components = {
            name
            for row in plugin_components
            if row.get("plugin_selected") is False
            and (name := _safe_name(row.get("name")))
        }
        for label, names in (
            ("Plugin components with unmet requirements", unavailable_components),
            ("Disabled plugin components", disabled_components),
        ):
            if names:
                listed = ", ".join(sorted(names)[:_MAX_SKILLS])
                lines.append(f"{label}: {listed} · review in Plugins")
    if mcp_enabled is False:
        lines.append("MCP is disabled · review in Settings ▸ Tools & MCP")
    if not catalog:
        lines.append("Current skill catalog unavailable.")
    elif any(name.casefold() not in counts for name in advertised):
        lines.append(
            "The request advertised skills absent from the current catalog; costs are historical."
        )
    lines.append(
        "Manage skills in Library ▸ Skills and connectors in Settings ▸ Tools & MCP."
    )
    return "\n".join(lines)


def prepared_agent_skill_observer(
    session: Any,
    scope: SkillReportScope,
    schemas: Sequence[ToolSchema],
    skill_names: Mapping[str, str],
    *,
    native: bool,
    protocol: str = "",
    conflicts: Sequence[str] = (),
) -> Callable[[PreparedProviderRequest], None]:
    """Bind the admitted schema identities to the first prepared model request."""
    from tldw_chatbook.Agents.native_tools import schemas_to_openai_tools

    from .console_prepared_request import thaw_json

    def observe(request: PreparedProviderRequest) -> None:
        observed_weights = schema_cost_weights(schemas, skill_names)
        native_rows = schemas_to_openai_tools(list(schemas)) if native else []
        fenced_tokens = None
        if thaw_json(request.semantic.tools) != native_rows:
            observed_weights = None
        elif not native:
            # Use the exact protocol rendered by the live request owner.
            if not schemas:
                fenced_tokens = 0
            elif protocol and any(
                isinstance(row.get("content"), str) and protocol in row["content"]
                for row in request.semantic.system
            ):
                fenced_tokens = (len(protocol) + 3) // 4
        record_skill_cost(
            session,
            scope,
            request.accounting,
            native=native,
            schema_weights=observed_weights,
            fenced_tokens=fenced_tokens,
            prepared=True,
            conflicts=conflicts,
        )

    return observe
