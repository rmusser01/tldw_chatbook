# Workspace Assistant Defaults Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give each explicit Console workspace a default agent persona with per-persona tool policy rules and per-workspace permission profiles, unified with tldw_server's Workspace Assistant Defaults contract.

**Architecture:** A reference-backed `assistant_defaults` JSON column on `workspace_records` (WorkspaceDB v3) names a persona from the JSON persona store and a named profile in `mcp_permissions.json` (the dormant `profiles` dict goes live). Persona policy rules evaluate deny-by-default and only narrow. Turn context carries the two posture components; composition and the registry dispatch choke point enforce them after all existing gates.

**Tech Stack:** Python ≥3.11, Textual 8.x, SQLite (WAL, per-thread held connections), pydantic schemas in `tldw_api/` mirroring tldw_server, pytest (file-based `tmp_path` fixtures, no `:memory:`).

**Spec:** `Docs/superpowers/specs/2026-08-29-workspace-assistant-defaults-design.md`

## Global Constraints

- Narrowing-only: no persona rule, profile, or defaults field may re-enable anything a config gate, binding access rule, ephemeral restriction, kill switch, rug-pull hash floor, or high-risk floor disabled. Order: gates → binding → kill switch → profile grants → persona floor → call caps.
- `MCP/permission_store.py` `SCHEMA_VERSION` stays **1** (bumping triggers the corrupt-file `.bak` policy and destroys user permissions). Shape changes are additive keys under `profiles`.
- WorkspaceDB moves v2 → **v3** (`_CURRENT_SCHEMA_VERSION = 3`), with a runner SQL file kept aligned in `DB/migrations/`.
- `ws-` profile-id prefix is reserved for auto-created workspace profiles.
- `persona_memory_mode` `read_write` saves require an explicit confirmation step.
- Server parity: field names, enum values, and reason codes mirror `tldw_server` `origin/dev` (`WorkspaceAssistantDefaults`, `WorkspaceEffectiveAssistantDefault`, `PersonaPolicyRule`). Rule kinds: `"mcp_tool" | "skill"` (locally `mcp_tool` covers every non-skill catalog tool).
- All SQL parameterized; permission-store writes atomic (existing tmp+rename); no full-suite test sweeps — targeted runs only, per repo policy.
- **Commits are pathspec-scoped** (`git commit -m "..." -- <paths>`): this branch carries an unrelated staged deletion (`backlog/tasks/task-19610 ...`) that must never ride along.
- ADR is filed before implementation (Task 1). Backlog CLI always with `--plain`.

---

### Task 1: File ADR-079 and the Backlog task

**Files:**
- Create: `backlog/decisions/079-workspace-assistant-defaults.md`
- Create via CLI: one Backlog task (note the assigned `task-<id>` for later tasks)

**Interfaces:**
- Consumes: the approved spec.
- Produces: ADR path + task id referenced by Tasks 2-12 ("ADR-079", "TASK-<assigned>").

- [ ] **Step 1: Verify the next free ADR number**

Run: `ls backlog/decisions/ | sort | tail -8`
Expected: `076-library-lifecycle-progressive-disclosure.md`, `076-server-offloaded-scheduled-agent-tasks.md` (the known duplicate; TASK-19610 renumbers it to 077), `078-...`. If `079-*` already exists, use the next free number and adjust every reference in this plan accordingly.

- [ ] **Step 2: Write the ADR**

Create `backlog/decisions/079-workspace-assistant-defaults.md` with the repo's ADR format (Status: Accepted, Date: 2026-08-29, Context/Decision/Consequences). Decision content, condensed from the spec: adopt the server's workspace `assistant_defaults` contract (reference-backed, stored-vs-effective, four-tier precedence, session independence); persona-local policy rules narrow only; named permission profiles referenced by id with per-key inheritance from `default`; `tool_policy_profile_id` accepted locally ahead of the server (its Tool Administration PRD is draft; our substrate exists); convenience auto-create as a local extension (persona + `ws-<id>` profile created then referenced, non-fatal on failure, backfill skips archived); `read_write` memory mode gated by explicit confirmation; ADR-069 binding authority and all floors unchanged. Link the spec path.

- [ ] **Step 3: File the Backlog task**

```bash
backlog task create "Workspace assistant defaults — personas, policy rules, permission profiles" \
  -d "Per-workspace default agent persona with narrowing-only tool policy rules and named permission profiles, unified with tldw_server's Workspace Assistant Defaults contract." \
  --ac "Workspace assistant_defaults stored/read with server-shaped validation and effective resolution with reason codes" \
  -l console,personas,workspaces --priority high
```

Record the assigned id. Leave status To Do until Task 2 starts (`backlog task edit <id> -a @robert -s "In Progress" --plain`).

- [ ] **Step 4: Commit (pathspec-scoped)**

```bash
git add backlog/decisions/079-workspace-assistant-defaults.md
git commit -m "docs(backlog): file ADR-079 and task for workspace assistant defaults" -- backlog/decisions/079-workspace-assistant-defaults.md
```

---

### Task 2: Persona policy rules — schema mirror and store validation

**Files:**
- Modify: `tldw_chatbook/tldw_api/character_persona_schemas.py` (after `LocalPersonaProfileCreate` ~L556 and `LocalPersonaProfileUpdate` ~L575)
- Modify: `tldw_chatbook/Character_Chat/local_character_persona_service.py` (record normalization)
- Test: `Tests/Persona/test_persona_policy_rules.py` (create dir if missing; check `Tests/` for an existing Persona dir name first and follow it)

**Interfaces:**
- Produces: `PersonaPolicyRule` (pydantic, `extra="forbid"`, fields `rule_kind: Literal["mcp_tool","skill"]`, `rule_name: str` min 1 max 512, `allowed: bool = True`, `require_confirmation: bool = False`, `max_calls_per_turn: int | None` ge 1); optional `policy_rules: list[PersonaPolicyRule] | None = None` on `LocalPersonaProfileCreate`/`LocalPersonaProfileUpdate`; `normalize_policy_rules(value) -> list[dict]` in the persona service module (tolerant: drops malformed entries with a warning, returns []).

- [ ] **Step 1: Write the failing schema parity test**

```python
"""Persona policy rules mirror the tldw_server PersonaPolicyRule contract."""
import pytest
from tldw_chatbook.tldw_api.character_persona_schemas import PersonaPolicyRule


def test_rule_shape_matches_server_contract():
    rule = PersonaPolicyRule.model_validate(
        {"rule_kind": "mcp_tool", "rule_name": "fs_write", "allowed": False}
    )
    assert rule.rule_kind == "mcp_tool"
    assert rule.require_confirmation is False
    assert rule.max_calls_per_turn is None


def test_rejects_unknown_kind_and_extras():
    with pytest.raises(Exception):
        PersonaPolicyRule.model_validate({"rule_kind": "syscall", "rule_name": "x"})
    with pytest.raises(Exception):
        PersonaPolicyRule.model_validate(
            {"rule_kind": "skill", "rule_name": "x", "grant": True}
        )


def test_caps_minimum_is_one():
    with pytest.raises(Exception):
        PersonaPolicyRule.model_validate(
            {"rule_kind": "mcp_tool", "rule_name": "x", "max_calls_per_turn": 0}
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest Tests/Persona/test_persona_policy_rules.py -v`
Expected: FAIL — `ImportError`/`cannot import name 'PersonaPolicyRule'`.

- [ ] **Step 3: Add the schema**

In `character_persona_schemas.py`:

```python
PersonaPolicyRuleKind = Literal["mcp_tool", "skill"]


class PersonaPolicyRule(BaseModel):
    """Persona-local tool policy rule — mirrors tldw_server PersonaPolicyRule.

    Narrowing-only at runtime: ``allowed=False`` removes a tool from the
    advertised set, ``require_confirmation=True`` floors it to "ask",
    ``max_calls_per_turn`` caps invocations per run. No rule can widen.
    """

    model_config = ConfigDict(extra="forbid")

    rule_kind: PersonaPolicyRuleKind
    rule_name: str = Field(..., min_length=1, max_length=512)
    allowed: bool = True
    require_confirmation: bool = False
    max_calls_per_turn: int | None = Field(default=None, ge=1)
```

Add `policy_rules: list[PersonaPolicyRule] | None = None` to `LocalPersonaProfileCreate` and `LocalPersonaProfileUpdate` (follow each model's existing field style; `Update` mirrors create fields as nullable-or-optional per its current convention).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest Tests/Persona/test_persona_policy_rules.py -v` — Expected: PASS.

- [ ] **Step 5: Tolerant normalization on read**

In `local_character_persona_service.py` add a module function:

```python
def normalize_policy_rules(value: Any) -> list[dict[str, Any]]:
    """Return validated policy-rule dicts; malformed entries drop with a warning."""
    if not isinstance(value, list):
        return []
    rules: list[dict[str, Any]] = []
    for entry in value:
        try:
            from ..tldw_api.character_persona_schemas import PersonaPolicyRule

            rules.append(
                PersonaPolicyRule.model_validate(entry).model_dump(mode="json")
            )
        except Exception:
            logger.warning("Dropping malformed persona policy rule: {!r}", entry)
    return rules
```

Add `logger` import per file conventions (loguru is already the house logger; if the module lacks it, `from loguru import logger`). Call it in `_persona_profile_view` so views always expose a clean `policy_rules` list, and in `update_persona_profile` after `changes` are applied (`record["policy_rules"] = normalize_policy_rules(record.get("policy_rules"))`) so hand-edited JSON self-heals on next save.

Extend the test file:

```python
def test_normalize_drops_malformed_rules():
    from tldw_chatbook.Character_Chat.local_character_persona_service import (
        normalize_policy_rules,
    )

    cleaned = normalize_policy_rules(
        [{"rule_kind": "mcp_tool", "rule_name": "ok"}, {"rule_kind": "bogus"}, "junk"]
    )
    assert cleaned == [
        {"rule_kind": "mcp_tool", "rule_name": "ok", "allowed": True,
         "require_confirmation": False, "max_calls_per_turn": None}
    ]
```

Run: `pytest Tests/Persona/test_persona_policy_rules.py -v` — PASS.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(personas): policy rules schema mirror with tolerant normalization" \
  -- tldw_chatbook/tldw_api/character_persona_schemas.py \
     tldw_chatbook/Character_Chat/local_character_persona_service.py \
     Tests/Persona/test_persona_policy_rules.py
```

---

### Task 3: Persona policy evaluator (pure module)

**Files:**
- Create: `tldw_chatbook/Agents/persona_policy.py`
- Test: `Tests/Agents/test_persona_policy.py` (follow the existing `Tests/Agents/` layout if present, else `Tests/Persona/`)

**Interfaces:**
- Consumes: `PersonaPolicyRule` schema from Task 2.
- Produces:
  - `PersonaToolPolicy` (frozen dataclass; `rules: tuple[dict, ...]`, `kinds: frozenset[str]`)
  - `parse_persona_policy(record: Mapping) -> PersonaToolPolicy` (tolerant of missing/invalid)
  - `parse_persona_policy_from_rules(rules: Iterable[Mapping] | None) -> PersonaToolPolicy`
  - `ToolPolicyVerdict` (frozen dataclass: `advertised: bool`, `requires_confirmation: bool`, `max_calls_per_turn: int | None`)
  - `evaluate_tool_policy(policy: PersonaToolPolicy, *, rule_kind: str, tool_name: str) -> ToolPolicyVerdict`
  - `persona_floor_state(state: EffectiveToolState, policy: PersonaToolPolicy, tool_name: str) -> EffectiveToolState` — floors `allow`→`ask` (origin `"persona_policy"`) when the verdict requires confirmation; other states pass through.

- [ ] **Step 1: Write the failing tests (including the never-widen property)**

```python
import copy
from dataclasses import replace

from hypothesis import given, settings
from hypothesis import strategies as st

from tldw_chatbook.Agents.persona_policy import (
    PersonaToolPolicy,
    evaluate_tool_policy,
    parse_persona_policy,
    parse_persona_policy_from_rules,
    persona_floor_state,
)
from tldw_chatbook.MCP.permission_store import EffectiveToolState

RULES = st.lists(
    st.fixed_dictionaries(
        {
            "rule_kind": st.sampled_from(["mcp_tool", "skill"]),
            "rule_name": st.sampled_from(["fs_write", "web_search", "fs_*", "web_*", "x*"]),
            "allowed": st.booleans(),
            "require_confirmation": st.booleans(),
            "max_calls_per_turn": st.one_of(st.none(), st.integers(min_value=1, max_value=9)),
        }
    ),
    max_size=6,
)
NAMES = st.sampled_from(["fs_write", "fs_read", "web_search", "web_fetch", "unrelated"])


def baseline(name):
    return evaluate_tool_policy(PersonaToolPolicy(), rule_kind="mcp_tool", tool_name=name)


@given(rules=RULES, name=NAMES)
@settings(max_examples=300)
def test_rules_never_widen(rules, name):
    verdict = evaluate_tool_policy(
        parse_persona_policy_from_rules(rules), rule_kind="mcp_tool", tool_name=name
    )
    base = baseline(name)
    assert verdict.advertised <= base.advertised
    assert verdict.requires_confirmation >= base.requires_confirmation


def test_no_rules_is_identity_posture():
    verdict = evaluate_tool_policy(
        parse_persona_policy({}), rule_kind="mcp_tool", tool_name="fs_write"
    )
    assert (verdict.advertised, verdict.requires_confirmation) == (True, False)


def test_deny_by_default_when_kind_rules_present():
    policy = parse_persona_policy_from_rules(
        [{"rule_kind": "mcp_tool", "rule_name": "fs_read", "allowed": True}]
    )
    unlisted = evaluate_tool_policy(policy, rule_kind="mcp_tool", tool_name="fs_write")
    assert unlisted.advertised is False  # kinds with rules deny unlisted tools


def test_explicit_denial_wins_and_confirmation_ors():
    policy = parse_persona_policy_from_rules(
        [
            {"rule_kind": "mcp_tool", "rule_name": "web_*", "allowed": True,
             "require_confirmation": True, "max_calls_per_turn": 4},
            {"rule_kind": "mcp_tool", "rule_name": "web_search", "allowed": False},
        ]
    )
    verdict = evaluate_tool_policy(policy, rule_kind="mcp_tool", tool_name="web_search")
    assert verdict.advertised is False
    other = evaluate_tool_policy(policy, rule_kind="mcp_tool", tool_name="web_fetch")
    assert other.advertised and other.requires_confirmation and other.max_calls_per_turn == 4


def test_bounded_wildcard_is_prefix_only():
    policy = parse_persona_policy_from_rules(
        [{"rule_kind": "mcp_tool", "rule_name": "fs_*", "allowed": False}]
    )
    assert not evaluate_tool_policy(policy, rule_kind="mcp_tool", tool_name="fs_list").advertised
    assert evaluate_tool_policy(policy, rule_kind="mcp_tool", tool_name="git_status").advertised


def test_skill_rules_do_not_affect_mcp_tools():
    policy = parse_persona_policy_from_rules(
        [{"rule_kind": "skill", "rule_name": "deep-research", "allowed": False}]
    )
    assert evaluate_tool_policy(policy, rule_kind="mcp_tool", tool_name="fs_read").advertised


def test_floor_state_only_lowers_allow():
    policy = parse_persona_policy_from_rules(
        [{"rule_kind": "mcp_tool", "rule_name": "web_*", "require_confirmation": True}]
    )
    allowed = EffectiveToolState(state="allow", origin="tool_override")
    floored = persona_floor_state(allowed, policy, "web_search")
    assert (floored.state, floored.origin) == ("ask", "persona_policy")
    # deny/ask pass through untouched; non-matching tool untouched
    assert persona_floor_state(
        EffectiveToolState(state="deny", origin="tool_override"), policy, "web_search"
    ).state == "deny"
    assert persona_floor_state(allowed, policy, "fs_read") is allowed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Agents/test_persona_policy.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `Agents/persona_policy.py`**

```python
"""Persona-local tool policy evaluation — narrowing-only.

Mirrors the server's persona policy semantics: deny-by-default when rules
exist for a kind, bounded (prefix-only) wildcards, explicit deny precedence,
confirmation floors, and per-run call caps. No rule can widen access; callers
layer this after every gate and floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from loguru import logger

from tldw_chatbook.MCP.permission_store import EffectiveToolState


@dataclass(frozen=True)
class PersonaToolPolicy:
    rules: tuple[dict, ...] = ()
    kinds: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ToolPolicyVerdict:
    advertised: bool
    requires_confirmation: bool
    max_calls_per_turn: int | None


def parse_persona_policy_from_rules(rules: Iterable[Mapping] | None) -> PersonaToolPolicy:
    cleaned: list[dict] = []
    for entry in rules or ():
        if not isinstance(entry, Mapping):
            continue
        try:
            from tldw_chatbook.tldw_api.character_persona_schemas import PersonaPolicyRule

            cleaned.append(
                PersonaPolicyRule.model_validate(dict(entry)).model_dump(mode="json")
            )
        except Exception:
            logger.warning("Dropping malformed persona policy rule: {!r}", entry)
    return PersonaToolPolicy(
        rules=tuple(cleaned), kinds=frozenset(r["rule_kind"] for r in cleaned)
    )


def parse_persona_policy(record: Mapping) -> PersonaToolPolicy:
    return parse_persona_policy_from_rules(
        record.get("policy_rules") if isinstance(record, Mapping) else None
    )


def _matches(rule_name: str, tool_name: str) -> bool:
    if rule_name.endswith("*"):
        return tool_name.startswith(rule_name[:-1])
    return rule_name == tool_name


def evaluate_tool_policy(
    policy: PersonaToolPolicy, *, rule_kind: str, tool_name: str
) -> ToolPolicyVerdict:
    if rule_kind not in policy.kinds:
        return ToolPolicyVerdict(True, False, None)
    matched = [r for r in policy.rules if r["rule_kind"] == rule_kind and _matches(r["rule_name"], tool_name)]
    if not matched:
        return ToolPolicyVerdict(False, False, None)
    if any(r.get("allowed") is False for r in matched):
        # A denied tool still reports requires_confirmation=True so downstream
        # refusal copy stays informative even when unadvertised.
        return ToolPolicyVerdict(False, True, None)
    caps = [r["max_calls_per_turn"] for r in matched if r.get("max_calls_per_turn")]
    return ToolPolicyVerdict(
        advertised=True,
        requires_confirmation=any(r.get("require_confirmation") for r in matched),
        max_calls_per_turn=min(caps) if caps else None,
    )


def persona_floor_state(
    state: EffectiveToolState, policy: PersonaToolPolicy, tool_name: str
) -> EffectiveToolState:
    verdict = evaluate_tool_policy(policy, rule_kind="mcp_tool", tool_name=tool_name)
    if verdict.requires_confirmation and state.state == "allow":
        return EffectiveToolState(state="ask", origin="persona_policy")
    return state
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/Agents/test_persona_policy.py -v` — PASS (including the property test).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/persona_policy.py Tests/Agents/test_persona_policy.py
git commit -m "feat(agents): narrowing-only persona policy evaluator" -- tldw_chatbook/Agents/persona_policy.py Tests/Agents/test_persona_policy.py
```

---

### Task 4: WorkspaceDB v3 + registry assistant_defaults + effective resolver

**Files:**
- Modify: `tldw_chatbook/DB/Workspace_DB.py` (schema, backfill flag)
- Create: `tldw_chatbook/DB/migrations/workspaces_v2_to_v3_assistant_defaults.sql`
- Modify: `tldw_chatbook/Workspaces/models.py` (`WorkspaceAssistantDefaults`, record field)
- Modify: `tldw_chatbook/Workspaces/registry_service.py` (mapper, create kwarg, set/clear)
- Create: `tldw_chatbook/Workspaces/assistant_defaults.py` (effective resolver)
- Test: `Tests/Workspaces/test_workspace_assistant_defaults.py`

**Interfaces:**
- Produces:
  - `WorkspaceAssistantDefaults` frozen dataclass in `models.py`: `assistant_kind: str = "persona"`, `assistant_id: str = ""`, `persona_memory_mode: str = "read_only"`, `voice: None = None`, `style: None = None`, `tool_policy_profile_id: str | None = None`; `__post_init__` validates kind ∈ `("persona",)`, mode ∈ `("read_only", "read_write")`, non-empty `assistant_id`.
  - `WorkspaceDB.is_agent_backfill_complete() -> bool`, `WorkspaceDB.mark_agent_backfill_complete() -> None`.
  - Registry: `create_workspace(..., assistant_defaults: WorkspaceAssistantDefaults | None = None)`; `set_assistant_defaults(workspace_id, defaults: WorkspaceAssistantDefaults, *, confirm_read_write: bool = False) -> WorkspaceRecord`; `clear_assistant_defaults(workspace_id) -> WorkspaceRecord`. Read-write without `confirm_read_write=True` raises `WorkspaceRegistryServiceError`.
  - `Workspaces/assistant_defaults.py`: `DEGRADED_REASONS` tuple (server codes verbatim), `WorkspaceEffectiveAssistantDefault` frozen dataclass (`status`, `source`, `assistant_kind`, `assistant_id`, `label`, `persona_memory_mode`, `degraded_reason`), `resolve_effective_assistant_default(defaults: WorkspaceAssistantDefaults | None, persona_lookup: Callable[[str], Mapping | None]) -> WorkspaceEffectiveAssistantDefault`.

- [ ] **Step 1: Write the failing tests**

```python
"""WorkspaceDB v3 + assistant_defaults roundtrip + effective resolution."""
import json
import sqlite3
from pathlib import Path

import pytest

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.assistant_defaults import (
    WorkspaceEffectiveAssistantDefault,
    resolve_effective_assistant_default,
)
from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults
from tldw_chatbook.Workspaces.registry_service import (
    LocalWorkspaceRegistryService,
    WorkspaceRegistryServiceError,
)


def build_registry(tmp_path: Path) -> LocalWorkspaceRegistryService:
    return LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="client-1")
    )


def test_v2_database_migrates_preserving_rows(tmp_path):
    legacy = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(legacy)
    conn.executescript(
        """
        CREATE TABLE schema_version (version INTEGER NOT NULL);
        INSERT INTO schema_version (version) VALUES (2);
        CREATE TABLE workspace_records (
            workspace_id TEXT PRIMARY KEY, name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '', authority TEXT NOT NULL,
            sync_status TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 0,
            archived INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        INSERT INTO workspace_records VALUES
            ('w-1', 'Research', '', 'local_only', 'not_configured', 0, 0, 't1', 't1');
        """
    )
    conn.commit()
    conn.close()
    db = WorkspaceDB(legacy, client_id="client-1")
    assert db.get_schema_version() == 3
    cols = {row[1] for row in db.connection().execute("PRAGMA table_info(workspace_records)")}
    assert "assistant_defaults" in cols
    row = db.connection().execute(
        "SELECT name FROM workspace_records WHERE workspace_id = 'w-1'"
    ).fetchone()
    assert row[0] == "Research"
    assert db.is_agent_backfill_complete() is False
    db.mark_agent_backfill_complete()
    assert WorkspaceDB(legacy, client_id="client-1").is_agent_backfill_complete() is True


def test_defaults_roundtrip_and_validation(tmp_path):
    registry = build_registry(tmp_path)
    record = registry.create_workspace(workspace_id="w-9", name="Lit Review")
    assert record.assistant_defaults is None
    defaults = WorkspaceAssistantDefaults(
        assistant_id="local-persona-abc", tool_policy_profile_id="ws-w-9"
    )
    updated = registry.set_assistant_defaults("w-9", defaults)
    assert updated.assistant_defaults == defaults
    assert registry.get_workspace("w-9").assistant_defaults == defaults
    cleared = registry.clear_assistant_defaults("w-9")
    assert cleared.assistant_defaults is None


def test_read_write_requires_confirmation(tmp_path):
    registry = build_registry(tmp_path)
    registry.create_workspace(workspace_id="w-2", name="W2")
    defaults = WorkspaceAssistantDefaults(
        assistant_id="p1", persona_memory_mode="read_write"
    )
    with pytest.raises(WorkspaceRegistryServiceError):
        registry.set_assistant_defaults("w-2", defaults)
    registry.set_assistant_defaults("w-2", defaults, confirm_read_write=True)


def test_malformed_stored_json_degrades_to_none(tmp_path):
    registry = build_registry(tmp_path)
    registry.create_workspace(workspace_id="w-3", name="W3")
    with registry.db.transaction() as conn:
        conn.execute(
            "UPDATE workspace_records SET assistant_defaults = ? WHERE workspace_id = 'w-3'",
            ("{not json",),
        )
    assert registry.get_workspace("w-3").assistant_defaults is None


def test_effective_resolution_reason_codes():
    none = resolve_effective_assistant_default(None, lambda _id: {})
    assert (none.status, none.degraded_reason) == ("none", None)
    deleted = resolve_effective_assistant_default(
        WorkspaceAssistantDefaults(assistant_id="gone"), lambda _id: None
    )
    assert (deleted.status, deleted.degraded_reason) == ("unavailable", "persona_deleted")
    ok = resolve_effective_assistant_default(
        WorkspaceAssistantDefaults(assistant_id="p"),
        lambda _id: {"id": "p", "name": "Lit Agent"},
    )
    assert (ok.status, ok.label, ok.source) == ("available", "Lit Agent", "workspace")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Workspaces/test_workspace_assistant_defaults.py -v`
Expected: FAIL — no `assistant_defaults` attribute, no resolver module.

- [ ] **Step 3: Implement the migration**

In `Workspace_DB.py`: set `_CURRENT_SCHEMA_VERSION = 3`. After the v2 block in `_initialize_schema` (which ends with the v2 version stamp), add:

```python
        needs_v3 = version < 3
        if needs_v3:
            with self.transaction() as write_conn:
                columns = {
                    row[1]
                    for row in write_conn.execute(
                        "PRAGMA table_info(workspace_records)"
                    )
                }
                if "assistant_defaults" not in columns:
                    write_conn.execute(
                        "ALTER TABLE workspace_records ADD COLUMN assistant_defaults TEXT"
                    )
                write_conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS workspace_agent_backfill (
                        key TEXT PRIMARY KEY,
                        completed_at TEXT NOT NULL
                    )
                    """
                )
                write_conn.execute(
                    "INSERT OR IGNORE INTO schema_version (version) VALUES (3)"
                )
```

(Keep the `needs_v2`/`needs_v3` gating shaped like the existing `needs_v2` check; fresh databases seed version 1 via the executescript and run both gates.) Add the flag methods:

```python
    def is_agent_backfill_complete(self) -> bool:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM workspace_agent_backfill WHERE key = 'assistant_defaults'"
            ).fetchone()
        return row is not None

    def mark_agent_backfill_complete(self) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO workspace_agent_backfill (key, completed_at)
                VALUES ('assistant_defaults', ?)
                """,
                (utc_now_iso(),),
            )
```

(`utc_now_iso` import from `..Workspaces.models` or inline `datetime.now(timezone.utc).isoformat()` matching the file's existing imports.) Create `DB/migrations/workspaces_v2_to_v3_assistant_defaults.sql` with the same three statements and a header comment mirroring the v1→v2 runner file's style.

- [ ] **Step 4: Implement model + registry**

`models.py`:

```python
WORKSPACE_ASSISTANT_KINDS = ("persona",)
WORKSPACE_PERSONA_MEMORY_MODES = ("read_only", "read_write")


@dataclass(frozen=True)
class WorkspaceAssistantDefaults:
    """Reference-backed default assistant for a workspace (server contract shape)."""

    assistant_kind: str = "persona"
    assistant_id: str = ""
    persona_memory_mode: str = "read_only"
    voice: None = None
    style: None = None
    tool_policy_profile_id: str | None = None

    def __post_init__(self) -> None:
        if self.assistant_kind not in WORKSPACE_ASSISTANT_KINDS:
            raise ValueError(f"unsupported assistant_kind: {self.assistant_kind!r}")
        if self.persona_memory_mode not in WORKSPACE_PERSONA_MEMORY_MODES:
            raise ValueError(
                f"invalid persona_memory_mode: {self.persona_memory_mode!r}"
            )
        if not _required_text(self.assistant_id, "assistant_id"):
            raise ValueError("assistant_id must be a non-empty string")
        if self.voice is not None or self.style is not None:
            raise ValueError("voice/style are reserved and must be null")
```

Add `assistant_defaults: WorkspaceAssistantDefaults | None = None` to `WorkspaceRecord` (after `archived`, before timestamps, so keyword construction order stays intact).

`registry_service.py`: add JSON guards mirroring `_metadata_to_json`/`_metadata_from_json`:

```python
def _assistant_defaults_to_json(defaults: WorkspaceAssistantDefaults | None) -> str | None:
    if defaults is None:
        return None
    try:
        return json.dumps(
            {
                "assistant_kind": defaults.assistant_kind,
                "assistant_id": defaults.assistant_id,
                "persona_memory_mode": defaults.persona_memory_mode,
                "voice": None,
                "style": None,
                "tool_policy_profile_id": defaults.tool_policy_profile_id,
            },
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise WorkspaceRegistryServiceError(f"assistant_defaults not serializable: {exc}")


def _assistant_defaults_from_json(value: Any) -> WorkspaceAssistantDefaults | None:
    if not value:
        return None
    try:
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError("not a dict")
        return WorkspaceAssistantDefaults(
            assistant_kind=str(payload.get("assistant_kind") or "persona"),
            assistant_id=str(payload.get("assistant_id") or ""),
            persona_memory_mode=str(payload.get("persona_memory_mode") or "read_only"),
            tool_policy_profile_id=payload.get("tool_policy_profile_id"),
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Ignoring malformed workspace assistant_defaults ({})", exc)
        return None
```

Wire into `_workspace_from_row` (`assistant_defaults=_assistant_defaults_from_json(row["assistant_defaults"])`), add the column + param to `create_workspace`'s INSERT, and add the two mutators (single `transaction()` UPDATE of `assistant_defaults` + `updated_at`, re-read and return; validate `read_write` confirmation and workspace existence first; malformed JSON planted via raw SQL degrades to `None` on read because the guard is tolerant).

- [ ] **Step 5: Implement `Workspaces/assistant_defaults.py`**

```python
"""Effective workspace assistant default resolution (server reason-code contract)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .models import WorkspaceAssistantDefaults

DEGRADED_REASONS = (
    "persona_deleted",
    "persona_unavailable",
    "persona_feature_disabled",
    "permission_denied",
    "invalid_default",
    "unsupported_assistant_kind",
)


@dataclass(frozen=True)
class WorkspaceEffectiveAssistantDefault:
    status: str  # "available" | "unavailable" | "none"
    source: str  # "workspace" | "none"
    assistant_kind: str | None = None
    assistant_id: str | None = None
    label: str | None = None
    persona_memory_mode: str | None = None
    degraded_reason: str | None = None


_NONE = WorkspaceEffectiveAssistantDefault(status="none", source="none")


def resolve_effective_assistant_default(
    defaults: WorkspaceAssistantDefaults | None,
    persona_lookup: Callable[[str], Mapping | None],
) -> WorkspaceEffectiveAssistantDefault:
    if defaults is None:
        return _NONE
    if defaults.assistant_kind != "persona":
        return WorkspaceEffectiveAssistantDefault(
            "unavailable", "workspace", degraded_reason="unsupported_assistant_kind"
        )
    record = persona_lookup(defaults.assistant_id)
    if record is None or record.get("deleted"):
        return WorkspaceEffectiveAssistantDefault(
            "unavailable", "workspace", degraded_reason="persona_deleted"
        )
    if not isinstance(record, Mapping) or not str(record.get("id") or ""):
        return WorkspaceEffectiveAssistantDefault(
            "unavailable", "workspace", degraded_reason="persona_unavailable"
        )
    return WorkspaceEffectiveAssistantDefault(
        status="available",
        source="workspace",
        assistant_kind="persona",
        assistant_id=defaults.assistant_id,
        label=str(record.get("name") or "") or None,
        persona_memory_mode=defaults.persona_memory_mode,
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest Tests/Workspaces/test_workspace_assistant_defaults.py Tests/Workspaces/test_workspace_registry_service.py -v`
Expected: PASS (new file green; existing registry suite still green — `SELECT *` reads now include the new column, and the mapper is the only construction path).

- [ ] **Step 7: Commit**

```bash
git commit -m "feat(workspaces): assistant_defaults storage, migration v3, effective resolver" \
  -- tldw_chatbook/DB/Workspace_DB.py \
     tldw_chatbook/DB/migrations/workspaces_v2_to_v3_assistant_defaults.sql \
     tldw_chatbook/Workspaces/models.py \
     tldw_chatbook/Workspaces/registry_service.py \
     tldw_chatbook/Workspaces/assistant_defaults.py \
     Tests/Workspaces/test_workspace_assistant_defaults.py
```

---

### Task 5: Named permission profiles in the permission store

**Files:**
- Modify: `tldw_chatbook/MCP/permission_store.py`
- Test: `Tests/MCP/test_permission_store.py` (extend; keep existing tests untouched-green)

**Interfaces:**
- Produces:
  - Every mutator (`set_global_default`, `set_server_default`, `set_tool_state`, `mark_config_changed`) gains keyword `profile_id: str = _DEFAULT_PROFILE_ID`; behavior at `"default"` is byte-identical to today.
  - `ensure_profile(profile_id: str) -> None` — creates the named profile seeded `{"servers": {}}` only (no `global_default` key: absence means inherit).
  - `list_profiles() -> list[str]`.
  - Resolvers (`resolve_effective_state`, `resolve_effective_state_by_key`, `resolve_builtin_state`) gain keyword `profile_id: str = _DEFAULT_PROFILE_ID`; lookups walk **level-by-level across the chain** `[named, default]`: tool override in named → tool override in default → server default in named → server default in default → global_default in named → global_default in default → `DEFAULT_GLOBAL`.
  - `_normalize_payload_shape` coerces every profile's `servers` to a dict.

- [ ] **Step 1: Write the failing tests**

Add to `Tests/MCP/test_permission_store.py` (follow its existing tmp_path store fixture style):

```python
def test_named_profile_survives_load_and_normalizes(store, tmp_path):
    store.ensure_profile("ws-w-1")
    store.save({**store.load()})
    reloaded = MCPPermissionStore(tmp_path / "mcp_permissions.json").load()
    assert "ws-w-1" in reloaded["profiles"]
    # hand-edited named profile with null servers coerces on load
    payload = reloaded
    payload["profiles"]["ws-w-1"]["servers"] = None
    store.save(payload)
    assert isinstance(store.load()["profiles"]["ws-w-1"]["servers"], dict)


def test_mutators_write_only_the_named_profile(store):
    store.ensure_profile("ws-w-1")
    store.set_tool_state("local:__local__", "fs_write", "deny", profile_id="ws-w-1")
    payload = store.load()
    named = payload["profiles"]["ws-w-1"]["servers"]["local:__local__"]["tools"]["fs_write"]
    assert named["state"] == "deny"
    assert "local:__local__" not in payload["profiles"]["default"]["servers"]


def test_resolver_inherits_level_by_level(store):
    store.set_tool_state("local:__local__", "fs_read", "allow", definition_hash=None)
    store.set_server_default("local:__local__", "ask", profile_id="ws-w-1")
    payload = store.load()
    # named server default beats default-profile tool override (per-level chain)
    state = resolve_effective_state_by_key(payload, "local:__local__", "fs_read", profile_id="ws-w-1")
    assert state.state == "ask"
    # key absent from named falls through to default-profile tool override
    state = resolve_effective_state_by_key(payload, "local:__local__", "fs_read")
    assert state.state == "allow"


def test_unknown_profile_id_inherits_everything(store):
    payload = store.load()
    state = resolve_effective_state_by_key(
        payload, "local:__local__", "fs_read", profile_id="ws-never-created"
    )
    assert state.state == DEFAULT_GLOBAL  # fresh workspace behaves like today
```

Adjust fixture usage to the file's actual store fixture; if `set_tool_state("allow")` demands a hash for this server key, use a `HASH_FREE_SERVER_KEYS` member or pass a dummy hash, mirroring existing tests.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/MCP/test_permission_store.py -v`
Expected: new tests FAIL (`TypeError: unexpected keyword 'profile_id'`), existing tests PASS.

- [ ] **Step 3: Implement**

- `_profile(payload, profile_id=_DEFAULT_PROFILE_ID)`: for named profiles seed only `{"servers": {}}` (no `global_default`); for `"default"` keep today's seeding exactly.
- `_normalize_payload_shape`: after the existing default-profile coercion, loop `for profile in profiles.values(): profile["servers"] = _as_mapping(profile.get("servers"))` (guard each value is a mapping first, as the default branch does).
- Mutators: thread `profile_id` into `_profile(...)` calls only.
- New chain helper:

```python
def _profile_chain(payload: dict[str, Any], profile_id: str) -> list[dict[str, Any]]:
    profiles = _as_mapping(payload.get("profiles"))
    chain: list[dict[str, Any]] = []
    if profile_id != _DEFAULT_PROFILE_ID:
        named = _as_mapping(profiles.get(profile_id))
        if named:
            chain.append(named)
    chain.append(_as_mapping(profiles.get(_DEFAULT_PROFILE_ID)))
    return chain
```

- Each resolver: replace the single `profile = ...get(_DEFAULT_PROFILE_ID)` walk with a level-by-level loop over `_profile_chain(payload, profile_id)` — tool override first across the chain, then server default across the chain, then `global_default` across the chain, final fallback `DEFAULT_GLOBAL`. Keep rug-pull, high-risk floor, and collapse rules applied to the resolved result exactly as today (they run after the walk, unchanged).
- `ensure_profile` / `list_profiles`:

```python
    def ensure_profile(self, profile_id: str) -> None:
        if profile_id == _DEFAULT_PROFILE_ID or not profile_id:
            return
        payload = self.load()
        profiles = payload.setdefault("profiles", {})
        profiles.setdefault(profile_id, {"servers": {}})
        self.save(payload)

    def list_profiles(self) -> list[str]:
        return sorted(_as_mapping(self.load().get("profiles")).keys())
```

- [ ] **Step 4: Run the full permission-store test file**

Run: `pytest Tests/MCP/test_permission_store.py Tests/MCP/test_permission_resolution.py -v`
Expected: PASS — new tests green, every pre-existing test byte-identical behavior at `"default"`.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(mcp): named permission profiles with per-key inheritance" \
  -- tldw_chatbook/MCP/permission_store.py Tests/MCP/test_permission_store.py
```

---

### Task 6: Control-plane service and provider plumbing

**Files:**
- Modify: `tldw_chatbook/MCP/unified_control_plane_service.py`
- Modify: `tldw_chatbook/Agents/mcp_tool_provider.py`
- Test: `Tests/MCP/test_control_plane_permissions.py` (extend) and `Tests/Agents/test_mcp_provider_profile.py` (new)

**Interfaces:**
- Produces:
  - Service: `effective_tool_states(tools, *, profile_id: str = "default")`, `gate_tool_test(tool, *, profile_id: str = "default")`, `gate_tool_test_for_profile(tool, profile_id: str)`, `set_tool_state(..., *, profile_id: str = "default")`, `set_server_default(..., *, profile_id: str = "default")`, `set_global_default(..., *, profile_id: str = "default")` — all defaulting to today's behavior.
  - `MCPToolProvider.__init__` gains `profile_id_provider: Callable[[], str] | None = None`; `compose_catalog` and the `_apply_verdict` persist path pass `profile_id=self._profile_id()` where `self._profile_id = profile_id_provider or (lambda: "default")`.
- Explicit non-goals (document in code comments): `Agents/builtin_tool_gate.py` and `MCP/local_server_tools.py` keep resolving against the default profile in V1 — they serve global surfaces outside the Console run path.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/MCP/test_control_plane_permissions.py (extend)
def test_gate_tool_test_for_profile_respects_named_profile(service, store):
    store.ensure_profile("ws-w-1")
    store.set_tool_state("local:__local__", "fs_write", "deny", profile_id="ws-w-1")
    hub = HubTool(server_key="local:__local__", name="fs_write")  # match existing hub fixtures
    assert service.gate_tool_test_for_profile(hub, "ws-w-1").state == "deny"
    assert service.gate_tool_test(hub).state != "deny"
```

Build the service/store fixtures the way the existing tests in that file do (grep for how `UnifiedMCPControlPlaneService` is constructed there and reuse). The provider test asserts `compose_catalog` drops a tool denied only in the named profile when `profile_id_provider` returns that id, and keeps it when it returns `"default"` — mirror the compose-catalog test setup from `Tests/Agents/` if one exists, otherwise construct `MCPToolProvider` with a stub `service` object exposing `effective_tool_states(tools, *, profile_id)` and the other methods `compose_catalog` touches (`local_external_catalog`, builtin inventory path), following the constructor requirements at `mcp_tool_provider.py:172-180`.

- [ ] **Step 2: Run to verify failure**, then **Step 3: implement**

In `unified_control_plane_service.py`, thread the keyword through the five methods to the store/resolver calls (the funnel is narrow — each method makes exactly one store call). Add the alias:

```python
    def gate_tool_test_for_profile(self, tool: HubTool, profile_id: str) -> EffectiveToolState:
        return self.gate_tool_test(tool, profile_id=profile_id)
```

In `mcp_tool_provider.py`: store the provider callable in `__init__`, and in `compose_catalog` replace `self._service.effective_tool_states(hub_tools)` with `self._service.effective_tool_states(hub_tools, profile_id=self._profile_id())`; in `_apply_verdict`'s always-allow persist path add `profile_id=self._profile_id()`.

- [ ] **Step 4: Run both test files** — PASS (existing suites untouched-green).

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(mcp): profile-aware service gates and provider catalog" \
  -- tldw_chatbook/MCP/unified_control_plane_service.py \
     tldw_chatbook/Agents/mcp_tool_provider.py \
     Tests/MCP/test_control_plane_permissions.py \
     Tests/Agents/test_mcp_provider_profile.py
```

---

### Task 7: Turn context, composition filter, and run call caps

**Files:**
- Modify: `tldw_chatbook/Chat/console_turn_context.py` (fields + capture)
- Modify: `tldw_chatbook/UI/Console_Modules/session.py` (`_build_console_turn_execution_context` ~L1656)
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (`_compose_local_provider` resolve_state closure ~L5545)
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py` (`_compose_run_registry_and_allowed` ~L2487)
- Create: `tldw_chatbook/Agents/run_tool_policy.py`
- Modify: `tldw_chatbook/Agents/tool_catalog.py` (`invoke_by_name` choke point ~L1208)
- Test: `Tests/Agents/test_run_tool_policy.py`, `Tests/Chat/test_turn_context_posture.py`

**Interfaces:**
- Produces:
  - `ConsoleTurnExecutionContext.persona_policy_rules: tuple[Mapping[str, Any], ...] = ()` and `.tool_policy_profile_id: str = "default"` (+ `capture` kwargs of the same names; frozen like the other mappings).
  - `Agents/run_tool_policy.py`: `PERSONA_POLICY_CALL_CAP_REFUSAL = "persona_policy_call_cap_reached: {name}"`; `RunToolPolicy(caps: Mapping[str, int])` with `check(run_id: str, name: str) -> tuple[bool, str | None]` (per-`(run_id, name)` counter; second+ call past the cap returns `(False, refusal_message)`).
  - `ToolCatalogRegistry.set_run_tool_policy(policy: RunToolPolicy | None)`; `invoke_by_name` refuses capped tools before dispatch with the pinned message in the same error-`ToolResult` shape that function already uses for unknown tools (grep the function body and mirror it exactly).

- [ ] **Step 1: Failing tests**

```python
# Tests/Agents/test_run_tool_policy.py
from tldw_chatbook.Agents.run_tool_policy import (
    PERSONA_POLICY_CALL_CAP_REFUSAL,
    RunToolPolicy,
)


def test_cap_allows_up_to_limit_then_refuses_persistently():
    policy = RunToolPolicy({"web_search": 2})
    assert policy.check("run-1", "web_search") == (True, None)
    assert policy.check("run-1", "web_search") == (True, None)
    ok, refusal = policy.check("run-1", "web_search")
    assert ok is False and refusal == PERSONA_POLICY_CALL_CAP_REFUSAL.format(name="web_search")
    assert policy.check("run-1", "web_search")[0] is False  # stays refused
    assert policy.check("run-2", "web_search")[0] is True  # per-run counters
    assert policy.check("run-1", "fs_read") == (True, None)  # uncapped untouched
```

`Tests/Chat/test_turn_context_posture.py`: build a `ConsoleTurnExecutionContext` via `capture` with the two new kwargs and assert the frozen values; then assert `_compose_run_registry_and_allowed`'s filter — construct the smallest harness the existing bridge tests use (look for an existing `Tests/Chat/` test covering `_compose_run_registry_and_allowed` and extend its fixture): with `persona_policy_rules=[{"rule_kind": "mcp_tool", "rule_name": "fs_*", "allowed": False}]` in turn context, `fs_write` is absent from the returned `allowed_tools` while `web_search` remains; with `rule_kind: "skill"` rules, only skill-provider tool names are filtered.

- [ ] **Step 2: Verify failure.**

- [ ] **Step 3: Implement**

`run_tool_policy.py`:

```python
"""Per-run tool call caps from persona policy rules."""

from __future__ import annotations

from typing import Mapping

PERSONA_POLICY_CALL_CAP_REFUSAL = "persona_policy_call_cap_reached: {name}"


class RunToolPolicy:
    """Counts invocations per (run_id, tool name); refuses past the cap."""

    def __init__(self, caps: Mapping[str, int]) -> None:
        self._caps = dict(caps)
        self._counts: dict[tuple[str, str], int] = {}

    def check(self, run_id: str, name: str) -> tuple[bool, str | None]:
        cap = self._caps.get(name)
        if cap is None:
            return True, None
        key = (run_id, name)
        count = self._counts.get(key, 0)
        if count >= cap:
            return False, PERSONA_POLICY_CALL_CAP_REFUSAL.format(name=name)
        self._counts[key] = count + 1
        return True, None
```

Turn-context fields + capture kwargs (follow the existing `_freeze`/`MappingProxyType` pattern; store rules as a frozen tuple of mappings). In `_build_console_turn_execution_context` after the workspace id resolution (~L1683): resolve `tool_policy_profile_id` from the workspace's `assistant_defaults` (guarded `getattr(self.app_instance, "workspace_registry_service", None)` → `get_workspace(workspace_id)` → defaults → `tool_policy_profile_id or "default"`; absent/Default/global → `"default"`), and `persona_policy_rules` from the session's assistant identity (store session record → `assistant_kind == "persona"` → `assistant_id` → `getattr(..., "local_character_persona_service", None)` → `get_persona_profile(assistant_id)` → its `policy_rules`; every failure degrades to `()`).

In `_compose_local_provider`, wrap the resolve state:

```python
        profile_id = (
            turn_context.tool_policy_profile_id if turn_context is not None else "default"
        )
        persona_policy = parse_persona_policy_from_rules(
            turn_context.persona_policy_rules if turn_context is not None else None
        )
        ...
        resolve_state=(
            lambda hub: persona_floor_state(
                service.gate_tool_test_for_profile(hub, profile_id), persona_policy, hub.name
            )
        ),
```

(`from tldw_chatbook.Agents.persona_policy import parse_persona_policy_from_rules, persona_floor_state` at module top.) In `_compose_run_registry_and_allowed`, after the allowed lists are assembled: parse the policy from turn context (thread turn context in if not already available — the function is called from `build_console_first_request_plan` which holds it), split names by source (the skills list is local to the function), and intersect:

```python
        filtered: list[str] = []
        for name in allowed_tools:
            kind = "skill" if name in skill_names else "mcp_tool"
            if evaluate_tool_policy(persona_policy, rule_kind=kind, tool_name=name).advertised:
                filtered.append(name)
        allowed_tools = tuple(filtered)
```

Build a `RunToolPolicy` from the verdict caps (`evaluate_tool_policy(...).max_calls_per_turn` per cataloged name; `None` → no cap) and call `registry.set_run_tool_policy(policy)`. In `tool_catalog.py`, store the policy (`self._run_tool_policy = None` in `__init__`/`reset_catalog_cache` semantics preserved) and at the head of `invoke_by_name` (after name resolution, before dispatch):

```python
        if self._run_tool_policy is not None:
            allowed, refusal = self._run_tool_policy.check(current_run_id(), name)
            if not allowed:
                # same error-ToolResult construction the unknown-tool path uses
                ...
```

Grep `invoke_by_name`'s unknown-tool branch and reuse its exact `ToolResult` error construction for the refusal body. Source the run id the way `local_tool_provider.py` does (grep `current_run_id` there and import from the same module — `invoke_by_name` may not import it today); if the run id is unavailable at that layer, key the counters on the policy instance's own per-run reset instead, preserving per-run semantics.

- [ ] **Step 4: Run tests**

Run: `pytest Tests/Agents/test_run_tool_policy.py Tests/Chat/test_turn_context_posture.py Tests/Chat/test_console_agent_bridge.py -v` (last file: use whatever existing bridge/composition test file exists — locate with `grep -rl "_compose_run_registry_and_allowed" Tests/`).
Expected: PASS, existing composition tests green (no rules → identity posture).

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(console): posture turn-context, advertising filter, run call caps" \
  -- tldw_chatbook/Chat/console_turn_context.py \
     tldw_chatbook/UI/Console_Modules/session.py \
     tldw_chatbook/Chat/console_chat_controller.py \
     tldw_chatbook/Chat/console_agent_bridge.py \
     tldw_chatbook/Agents/run_tool_policy.py \
     tldw_chatbook/Agents/tool_catalog.py \
     Tests/Agents/test_run_tool_policy.py \
     Tests/Chat/test_turn_context_posture.py
```

---

### Task 8: Workspace agent provisioner, create hook, and startup backfill

**Files:**
- Create: `tldw_chatbook/Workspaces/agent_provisioning.py`
- Modify: `tldw_chatbook/Workspaces/registry_service.py` (provisioner hook on `create_workspace`)
- Modify: `tldw_chatbook/app.py` (wiring after `_wire_character_persona_services()`)
- Test: `Tests/Workspaces/test_agent_provisioning.py`

**Interfaces:**
- Produces:
  - `WorkspaceAgentProvisioner(persona_service, permission_store)` with `provision(workspace: WorkspaceRecord) -> WorkspaceAssistantDefaults | None` — creates persona `f"{workspace.name} Agent"` via `persona_service.create_persona_profile(...)`, calls `permission_store.ensure_profile(f"ws-{workspace.workspace_id}")`, returns defaults referencing both; **never raises** (logs and returns `None` on any failure).
  - `run_workspace_agent_backfill(*, registry, provisioner) -> int` — iterates explicit non-archived non-Default workspaces with null `assistant_defaults`, provisions each, marks completion via `registry.db.mark_agent_backfill_complete()`; returns the count provisioned; idempotent.
  - `LocalWorkspaceRegistryService.__init__` gains keyword-only `agent_provisioner: Callable[[WorkspaceRecord], WorkspaceAssistantDefaults | None] | None = None`; `create_workspace` invokes it after the INSERT and persists the returned defaults (failure → `assistant_defaults` stays NULL + warning).
  - `app.py` gains `_wire_workspace_agent_provisioning()` called right after `self._wire_character_persona_services()` (call site ~L5775): constructs the provisioner from `self.local_character_persona_service` and the unified service's `permission_store` (guarded `getattr`; skip wiring when either is unavailable) and runs the backfill.

- [ ] **Step 1: Failing tests**

```python
"""Provisioner: convenience auto-create is reference-backed and non-fatal."""
from pathlib import Path

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.MCP.permission_store import MCPPermissionStore
from tldw_chatbook.Workspaces.agent_provisioning import (
    WorkspaceAgentProvisioner,
    run_workspace_agent_backfill,
)
from tldw_chatbook.Workspaces.models import DEFAULT_WORKSPACE_ID
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService


class StubPersonaService:
    def __init__(self):
        self.created = []

    def create_persona_profile(self, payload):
        payload = dict(payload)
        payload.setdefault("id", f"local-persona-{len(self.created) + 1}")
        self.created.append(payload)
        return payload


def build(tmp_path: Path, personas=None):
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")
    provisioner = (
        WorkspaceAgentProvisioner(personas, store) if personas is not None else None
    )
    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "ws.sqlite", client_id="c1"),
        agent_provisioner=provisioner.provision if provisioner is not None else None,
    )
    return registry, store


def test_provision_creates_persona_profile_and_reference(tmp_path):
    personas = StubPersonaService()
    registry, store = build(tmp_path, personas)
    record = registry.create_workspace(workspace_id="w-2", name="Research")
    assert personas.created and personas.created[0]["name"] == "Research Agent"
    assert record.assistant_defaults is not None
    assert record.assistant_defaults.tool_policy_profile_id == "ws-w-2"
    assert "ws-w-2" in store.list_profiles()


def test_provision_failure_is_non_fatal(tmp_path):
    class Broken:
        def create_persona_profile(self, payload):
            raise RuntimeError("boom")

    registry, _store = build(tmp_path, Broken())
    record = registry.create_workspace(workspace_id="w-3", name="W3")
    assert record.assistant_defaults is None


def test_backfill_skips_archived_and_default_and_is_idempotent(tmp_path):
    registry, store = build(tmp_path)
    personas = StubPersonaService()
    provisioner = WorkspaceAgentProvisioner(personas, store)
    registry.ensure_default_workspace()
    registry.create_workspace(workspace_id="w-4", name="Keep")
    registry.create_workspace(workspace_id="w-5", name="Skip")
    registry.archive_workspace("w-5")
    first = run_workspace_agent_backfill(registry=registry, provisioner=provisioner)
    assert first == 1
    assert registry.get_workspace("w-4").assistant_defaults is not None
    assert registry.get_workspace("w-5").assistant_defaults is None
    assert run_workspace_agent_backfill(registry=registry, provisioner=provisioner) == 0
```

Note: the backfill test builds its registry without the constructor hook (pass `build(tmp_path)` with no personas) and passes the provisioner to `run_workspace_agent_backfill` explicitly — the hook covers creation-time provisioning, the backfill covers pre-existing workspaces.

- [ ] **Step 2: Verify failure.**

- [ ] **Step 3: Implement `agent_provisioning.py`**

```python
"""Convenience auto-create of a workspace's default agent persona + profile."""

from __future__ import annotations

from loguru import logger

from .models import WorkspaceAssistantDefaults, WorkspaceRecord

SEED_SYSTEM_PROMPT = (
    "You are the default assistant for the \"{name}\" workspace. "
    "Help the user with work in this workspace; be direct and grounded in "
    "workspace sources when they are provided."
)


class WorkspaceAgentProvisioner:
    def __init__(self, persona_service, permission_store) -> None:
        self._personas = persona_service
        self._permissions = permission_store

    def provision(self, workspace: WorkspaceRecord) -> WorkspaceAssistantDefaults | None:
        try:
            record = self._personas.create_persona_profile(
                {
                    "name": f"{workspace.name} Agent",
                    "description": f"Default agent persona for workspace {workspace.name}.",
                    "system_prompt": SEED_SYSTEM_PROMPT.format(name=workspace.name),
                    "mode": "session_scoped",
                    "is_active": True,
                }
            )
            profile_id = f"ws-{workspace.workspace_id}"
            self._permissions.ensure_profile(profile_id)
            return WorkspaceAssistantDefaults(
                assistant_id=str(record["id"]), tool_policy_profile_id=profile_id
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Workspace agent provisioning failed for {}", workspace.workspace_id
            )
            return None


def run_workspace_agent_backfill(*, registry, provisioner: WorkspaceAgentProvisioner) -> int:
    if registry.db.is_agent_backfill_complete():
        return 0
    count = 0
    for record in registry.list_workspaces():
        if record.archived or record.workspace_id == DEFAULT_WORKSPACE_ID:
            continue
        if record.assistant_defaults is not None:
            continue
        defaults = provisioner.provision(record)
        if defaults is not None:
            registry.set_assistant_defaults(
                record.workspace_id, defaults, confirm_read_write=True
            )
            count += 1
    registry.db.mark_agent_backfill_complete()
    return count
```

Wire the hook into `create_workspace` (invoke after the successful INSERT + re-read; persist via the same UPDATE path `set_assistant_defaults` uses, or simply include the returned defaults in the INSERT by provisioning before insert — choose the after-insert form so a provisioning failure cannot roll back workspace creation) and add `_wire_workspace_agent_provisioning` to `app.py` as specified. Import `DEFAULT_WORKSPACE_ID` in `agent_provisioning.py`.

- [ ] **Step 4: Run tests** — `pytest Tests/Workspaces/test_agent_provisioning.py Tests/Workspaces/test_workspace_assistant_defaults.py -v` — PASS.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(workspaces): agent provisioner, create hook, startup backfill" \
  -- tldw_chatbook/Workspaces/agent_provisioning.py \
     tldw_chatbook/Workspaces/registry_service.py \
     tldw_chatbook/app.py \
     Tests/Workspaces/test_agent_provisioning.py
```

---

### Task 9: Session startup application (Console)

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_models.py` (`ConsoleSessionSettings`)
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (`new_session` ~L3808)
- Modify: `tldw_chatbook/UI/Console_Modules/session.py` (`_create_native_console_session_from_active_context` ~L1594; new helper)
- Test: `Tests/Chat/test_workspace_default_session.py`

**Interfaces:**
- Produces:
  - `ConsoleSessionSettings.persona_memory_mode: str | None = None` (persisted with settings; no store migration).
  - `new_session(..., assistant_kind: str | None = None, assistant_id: str | None = None, assistant_label: str | None = None)` forwarding to `store.create_session` (which already accepts the assistant fields).
  - `ConsoleSessionMixin._workspace_default_for_new_session() -> tuple[str, str, str, str] | None` returning `(assistant_id, label, system_prompt, memory_mode)` when the active workspace is explicit and its effective default is `available`, else `None`.

- [ ] **Step 1: Failing tests**

```python
"""Workspace default persona applies to NEW sessions only; precedence + independence."""
import pytest


def test_new_session_inherits_workspace_default_persona(store_with_default, controller):
    session = controller.new_session()
    assert session.assistant_kind == "persona"
    assert session.assistant_id == "local-persona-1"
    assert session.settings.persona_memory_mode == "read_only"
    assert "Lit Agent" in session.settings.system_prompt


def test_explicit_settings_win_over_workspace_default(controller_with_explicit):
    session = controller_with_explicit.new_session()  # caller passed persona settings
    assert session.assistant_id == "local-persona-explicit"


def test_plain_workspace_gets_no_persona(plain_controller):
    session = plain_controller.new_session()
    assert session.assistant_kind in (None, "generic")


def test_existing_sessions_independent_of_later_default_edits(controller, registry):
    session = controller.new_session()
    before = (session.assistant_kind, session.assistant_id, session.settings.system_prompt)
    registry.set_assistant_defaults(
        "w-1",
        WorkspaceAssistantDefaults(assistant_id="other"),
        confirm_read_write=True,
    )
    reloaded = controller.store.get_session(session.id)
    after = (reloaded.assistant_kind, reloaded.assistant_id, reloaded.settings.system_prompt)
    assert before == after
```

Build the harness the way existing console controller/store tests in `Tests/Chat/` do (grep for `new_session` / `create_session` tests and mirror their fixture construction — they typically instantiate `ConsoleChatStore` with a tmp ChaChaNotes DB and a controller with stubbed app services). The workspace-default stub: `workspace_registry_service` returning a workspace record whose `assistant_defaults` references a persona record in a stub `local_character_persona_service`.

- [ ] **Step 2: Verify failure.**

- [ ] **Step 3: Implement**

- Add the settings field (follow the dataclass's existing default-`None` optional fields).
- `_workspace_default_for_new_session`:

```python
    def _workspace_default_for_new_session(self):
        registry = getattr(self.app_instance, "workspace_registry_service", None)
        personas = getattr(self.app_instance, "local_character_persona_service", None)
        if registry is None or personas is None:
            return None
        workspace_id = (
            self._ensure_console_chat_store().workspace_context.active_workspace_id
        )
        if not workspace_id or workspace_id in (
            CONSOLE_GLOBAL_WORKSPACE_ID,
            DEFAULT_WORKSPACE_ID,
        ):
            return None
        workspace = registry.get_workspace(workspace_id)
        if workspace is None or workspace.archived:
            return None
        effective = resolve_effective_assistant_default(
            workspace.assistant_defaults,
            lambda pid: _safe_persona_lookup(personas, pid),
        )
        if effective.status != "available":
            return None
        record = personas.get_persona_profile(effective.assistant_id)
        prompt = build_persona_agent_system_prompt(record)
        return (
            effective.assistant_id,
            effective.label or "Workspace Agent",
            prompt,
            effective.persona_memory_mode or "read_only",
        )
```

with two small module helpers: `_safe_persona_lookup` (try/except → `None`) and

```python
def build_persona_agent_system_prompt(record: Mapping) -> str:
    """Compose a persona record into a Console system prompt (preview seam parity)."""
    from tldw_chatbook.Character_Chat.Character_Chat_Lib import compose_character_card_text

    return (
        compose_character_card_text(
            name=str(record.get("name") or "Workspace Agent"),
            system_prompt=str(record.get("system_prompt") or ""),
            personality=str(record.get("personality") or ""),
            description=str(record.get("description") or ""),
            user_name="User",
        )
        or "Stay in character."
    )
```

- In `_create_native_console_session_from_active_context`, when no explicit persona settings were supplied (the plain new-tab path passes defaults): resolve the tuple and pass `assistant_kind="persona"`, `assistant_id`, `assistant_label`, and `replace(settings, system_prompt=prompt, persona_memory_mode=mode)` into `new_session`. Handoff/character paths do not route through this function and stay untouched (their precedence above the workspace default). `new_session` forwards the assistant kwargs to `store.create_session`; the store already fills `workspace_id` from the active context.

- [ ] **Step 4: Run tests** — `pytest Tests/Chat/test_workspace_default_session.py -v` plus the existing session-creation tests (`grep -rl "new_session" Tests/Chat/ | head -3`), targeted. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(console): workspace default persona on new session creation" \
  -- tldw_chatbook/Chat/console_chat_models.py \
     tldw_chatbook/Chat/console_chat_controller.py \
     tldw_chatbook/UI/Console_Modules/session.py \
     Tests/Chat/test_workspace_default_session.py
```

---

### Task 10: Settings → Workspaces "Default assistant" surface

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py` (render ~L14991 region + handlers ~L18639 region)
- Test: `Tests/UI/test_settings_workspace_assistant_defaults.py`

**Interfaces:**
- Produces: `_render_workspace_default_assistant(registry, workspace_id)` generator yielding: section header Static `"Default assistant"` (`classes="destination-section"`); effective-status Static `id="settings-workspace-assistant-status"`; persona `OptionList` `id="settings-workspace-persona-picker"`; memory-mode Button `id="settings-workspace-memory-toggle"` (two-press confirm: first press re-renders as `"Confirm read_write?"`, second within the selection applies — status line explains); profile `OptionList` `id="settings-workspace-profile-picker"`; clear Button `id="settings-workspace-assistant-clear"`. Posture preview: `Static` lines `id="settings-workspace-posture-preview"` built by a new pure helper `compose_posture_preview(persona_rules, store_payload, profile_id, tool_names) -> list[str]` living in `Workspaces/assistant_defaults.py` (per tool: `available | ask | denied | capped (<n>)` plus the deciding layer name; display-only).

- [ ] **Step 1: Failing test** — follow the harness style of an existing settings-pane UI test (grep `Tests/UI/` for a settings workspaces or settings agents test; mount the region or call the render generator + handlers with a stub `app_instance`). Assert: the section renders for an explicit workspace with the effective label; selecting a persona + pressing apply calls `registry.set_assistant_defaults` with confirm gating on `read_write`; clear calls `clear_assistant_defaults`; degraded persona shows `persona_deleted` copy.

- [ ] **Step 2: Verify failure.**

- [ ] **Step 3: Implement** — follow the `_render_workspace_change_review` shape exactly (header Static + availability Statics + compact Buttons with stashed attributes instead of parsed ids; handlers as `@on(Button.Pressed, "#settings-workspace-memory-toggle")` etc., each ending with `self._refresh_settings_workspaces_pane()` and `_set_settings_workspaces_result(...)` feedback). Persona list source: `getattr(self.app_instance, "local_character_persona_service", None)` profile listing; profile list: the permission store's `list_profiles()` via the unified service's `permission_store` property (guarded). Tool names for the preview: the unified service's hub tool list (guarded; preview degrades to a "tool catalog unavailable" line rather than an error). Default workspace and archived cards render a locked note instead of the picker (matches the existing special-casing at L14952/L14959).

- [ ] **Step 4: Run tests** — `pytest Tests/UI/test_settings_workspace_assistant_defaults.py -v` plus `grep -rl "settings-workspace" Tests/UI/` targeted files. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(settings): workspace default assistant section with posture preview" \
  -- tldw_chatbook/UI/Screens/settings_screen.py \
     tldw_chatbook/Workspaces/assistant_defaults.py \
     Tests/UI/test_settings_workspace_assistant_defaults.py
```

---

### Task 11: Personas workbench rules editor, switcher label, import display

**Files:**
- Create: `tldw_chatbook/Widgets/Persona_Widgets/personas_policy_rules_editor.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py` (read-only section)
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py` (push data to both)
- Modify: `tldw_chatbook/Widgets/Console/console_workspace_switcher_modal.py` (persona label suffix)
- Modify: `tldw_chatbook/Persona_Visual/importer.py` + `Widgets/Persona_Widgets/personas_persona_visual_pack_widget.py` (review display)
- Test: `Tests/UI/test_persona_policy_rules_editor.py`, extend the importer test file

**Interfaces:**
- Produces:
  - `PersonasPolicyRulesEditor(Vertical)`: `show_rules(rules: list[dict])`, `clear_rules()`, posts `PersonaPolicyRulesChanged(rules: list[dict])` message; list + mini-form following `settings_agents_panel.py` (ListView `#personas-policy-rules-list`, inputs `#personas-policy-kind`, `#personas-policy-name`, `#personas-policy-allowed`, `#personas-policy-confirm`, `#personas-policy-caps`, New/Save/Delete buttons, status line). Kind input validated to `mcp_tool|skill`; caps input parsed as `int ≥ 1` or blank.
  - Inspector `show_policy_rules(rules)` renders a read-only summary Static (hidden until selection, kind-gated to personas like `_CONSOLE_ACTION_APPLICABLE_KINDS`).
  - Switcher entries append `" · {persona_label}"` when the workspace's effective default is available (guarded registry lookup; silent omit otherwise).
  - Import review: `inspect_persona_visual_draft` output includes a `policy_rule_count` line when the carried persona record has rules; the pack widget's notice Static displays `"Carries {n} narrowing-only tool policy rule(s) — review before publishing."`

- [ ] **Step 1: Failing tests** — editor CRUD roundtrip (add rule → `PersonaPolicyRulesChanged` carries the validated dict; malformed kind rejected with status message); inspector section hidden without selection; switcher label formatting for a workspace with/without defaults; importer review counts rules from a crafted draft persona record (mirror the existing importer test fixtures).

- [ ] **Step 2: Verify failure.**

- [ ] **Step 3: Implement** per the Interfaces block, following each file's established compose/handler patterns (the explorer notes above give the exact anchors: inspector `compose` L164-268, editor pattern `settings_agents_panel.py` L56-111, switcher modal entries, importer `inspect_persona_visual_draft`).

- [ ] **Step 4: Run tests** — targeted: `pytest Tests/UI/test_persona_policy_rules_editor.py` + the importer/personas tests that exist (`grep -rl "persona_visual\|importer" Tests/ | head -5`).

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(personas): policy rules editor, switcher label, import review display" \
  -- tldw_chatbook/Widgets/Persona_Widgets/personas_policy_rules_editor.py \
     tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py \
     tldw_chatbook/UI/Screens/personas_screen.py \
     tldw_chatbook/Widgets/Console/console_workspace_switcher_modal.py \
     tldw_chatbook/Persona_Visual/importer.py \
     tldw_chatbook/Widgets/Persona_Widgets/personas_persona_visual_pack_widget.py \
     Tests/UI/test_persona_policy_rules_editor.py
```

---

### Task 12: Docs, backlog hygiene, and verification

**Files:**
- Modify: `AGENTS.md` (Special Systems — new paragraph), the Backlog task (via CLI), `Docs/superpowers/specs/2026-08-29-workspace-assistant-defaults-design.md` (status line → Implemented)

- [ ] **Step 1: AGENTS.md note** — add under "Special Systems" (after the Model Catalog section):

```markdown
### Workspace Assistant Defaults
- Explicit workspaces carry reference-backed `assistant_defaults` (persona + permission profile); Default/global stay unset.
- Persona policy rules narrow only (deny-by-default advertising, ask floors, per-run call caps); profiles inherit unset keys from `default`; all existing gates/floors apply first.
- Governance: `backlog/decisions/079-workspace-assistant-defaults.md` and `Docs/superpowers/specs/2026-08-29-workspace-assistant-defaults-design.md`.
```

- [ ] **Step 2: Targeted verification sweep** — run every test file created/extended by Tasks 2-11 in one `pytest` invocation, plus `ruff check` on the touched packages if the repo lints that way (`grep -n ruff pyproject.toml` to confirm). Fix any failures.

- [ ] **Step 3: Backlog closeout** — check every AC checkbox, add the `## Implementation Notes` section (approach, decisions, files), then `backlog task edit <id> -s Done --plain`. Lessons check: add a `lessons-*` entry only if a genuine new trap surfaced; do not invent one.

- [ ] **Step 4: Commit**

```bash
git commit -m "docs: workspace assistant defaults closeout" -- AGENTS.md Docs/superpowers/specs/2026-08-29-workspace-assistant-defaults-design.md
```

---

## Self-Review (completed during planning)

- **Spec coverage:** data model (Tasks 4, 2), evaluator (3), profiles (5), service/provider plumbing (6), turn context + composition + caps (7), lifecycle/auto-create/backfill (8), session mechanics + precedence + independence (9), settings surface + preview (10), personas workbench + switcher + import trust display (11), ADR/docs/hygiene (1, 12). Stored-vs-effective split: Task 4 resolver, surfaced in Tasks 9/10. `persona_memory_mode` confirmation: Task 4 (registry gate) + Task 10 (UI two-press confirm). Reserved-null fields: Task 4 model `__post_init__`.
- **Type consistency:** `WorkspaceAssistantDefaults` (Task 4) consumed by Tasks 8-10; `PersonaToolPolicy`/`evaluate_tool_policy`/`persona_floor_state` (Task 3) consumed by Tasks 6-7, 10; `ensure_profile`/`list_profiles`/profile-keyword resolvers (Task 5) consumed by Tasks 6, 8, 10; `RunToolPolicy` (Task 7) wired at the registry choke point in the same task.
- **Known deliberate deferrals:** `builtin_tool_gate`/`local_server_tools` stay default-profile (documented in Task 6); session-approval set stays `(server_key, tool_name)`-keyed (spec: unchanged); server sync, scope rules, voice/style reserved (spec Future Stages).
