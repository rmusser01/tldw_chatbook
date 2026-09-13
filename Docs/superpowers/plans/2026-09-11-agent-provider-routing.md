# Agent Provider Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a Console master agent spawn sub-agents onto a specific provider+model (including `custom-ep:` registry endpoints) or a user-configured sub-agent default, with children owning their own sampling/API params.

**Architecture:** A new pure resolver (`Agents/agent_routing.py`) resolves provider → model → base_url → params for every spawn through a four-level routing order and a six-layer params stack, validated through the existing readiness seams. `AgentService.spawn()` calls it before fleet reservation; failures come back as `SpawnAdmissionRefusal` tool errors. The resolved target is snapshotted onto the run row (schema v16) so resume/continuation is stable.

**Tech Stack:** Python 3.11+, Textual 8.x, SQLite (idempotent ALTER migrations), pytest.

**Spec:** `Docs/superpowers/specs/2026-09-11-agent-provider-routing-design.md`
**ADR:** `backlog/decisions/147-agent-provider-routing.md` (amends ADR-146)
**Task:** TASK-32477

## Global Constraints

- Python ≥ 3.11; type hints on public APIs; Google-style docstrings.
- All SQL parameterized; schema changes bump `_CURRENT_SCHEMA_VERSION` and add idempotent ALTERs (`DB/AgentRuns_DB.py` pattern).
- `Agents/agent_service.py` stays the ONLY impure Agents module — new logic goes in pure modules.
- New `[agents]` config keys ship COMMENTED-OUT in `config.py`; defaults live in `Agents/agent_routing.py` and are read via `_setting` (from `Agents/run_log.py`).
- Spawn tool args carry identity only: `provider`/`model` strings, NEVER URLs or params.
- No silent cross-provider fallback: every routing failure is a loud tool error naming the failing level.
- Frozen dataclasses; params represented as `tuple[tuple[str, object], ...]` (sorted pairs) outside config-parse boundaries.
- Tests: real SQLite in-memory for DB tests; run targeted pytest only — NEVER the full suite unless the user explicitly opts in (AGENTS.md testing rule).
- `timeout` command is unavailable in this environment.

---

### Task 1: `Chat/sampling_params.py` — shared param keys, validator, merge helpers

**Files:**
- Create: `tldw_chatbook/Chat/sampling_params.py`
- Modify: `tldw_chatbook/Chat/console_session_settings.py:1561-1640` (five private merge helpers move out; call sites keep working via import aliases)
- Test: `Tests/Chat/test_sampling_params.py` (new)

**Interfaces:**
- Consumes: nothing (new leaf module; stdlib only).
- Produces:
  - `KNOWN_SAMPLING_PARAM_KEYS: frozenset[str]`
  - `validate_sampling_params(params: Mapping[str, object]) -> list[str]`
  - `params_to_tuple(params: Mapping[str, object]) -> tuple[tuple[str, object], ...]` (sorted)
  - `params_to_dict(pairs: tuple[tuple[str, object], ...] | Iterable...) -> dict`
  - `float_setting_from_sources(sources, key, default)`, `optional_float_setting_from_sources(sources, key)`, `optional_int_setting_from_sources(sources, key)`, `optional_string_setting_from_sources(sources, key)`, `bool_setting_from_sources(sources, key, default)` — moved VERBATIM from `console_session_settings.py:1561-1640`.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Chat/test_sampling_params.py
from tldw_chatbook.Chat.sampling_params import (
    KNOWN_SAMPLING_PARAM_KEYS,
    params_to_dict,
    params_to_tuple,
    validate_sampling_params,
)

def test_known_keys_accepted():
    assert validate_sampling_params(
        {"temperature": 0.2, "top_k": 40, "reasoning_effort": "low"}
    ) == []

def test_unknown_key_rejected_as_typo_guard():
    errors = validate_sampling_params({"temprature": 0.2})
    assert len(errors) == 1 and "unknown" in errors[0].lower()

def test_bool_rejected_for_numeric_key():
    # bool IS an int subclass; without an explicit guard `seed: true`
    # would pass an int check.
    assert validate_sampling_params({"seed": True}) != []

def test_wrong_type_rejected():
    assert validate_sampling_params({"temperature": "hot"}) != []
    assert validate_sampling_params({"max_tokens": 1.5}) != []

def test_streaming_is_not_a_sampling_param():
    assert "streaming" not in KNOWN_SAMPLING_PARAM_KEYS

def test_params_tuple_round_trip_sorted():
    pairs = params_to_tuple({"top_p": 0.9, "temperature": 0.2})
    assert pairs == (("temperature", 0.2), ("top_p", 0.9))
    assert params_to_dict(pairs) == {"temperature": 0.2, "top_p": 0.9}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Chat/test_sampling_params.py -v`
Expected: FAIL (`ModuleNotFoundError: tldw_chatbook.Chat.sampling_params`)

- [ ] **Step 3: Implement `sampling_params.py` and move the helpers**

```python
# tldw_chatbook/Chat/sampling_params.py
"""Shared sampling-param keys, validation, and layer-merge helpers (ADR-147).

Single source of truth for the param names a preset, a registry entry, or
the agent resolver may carry. Transport-level keys (``streaming``) are
deliberately excluded: child runs keep the run loop's streaming policy.
"""
from __future__ import annotations
from collections.abc import Iterable, Mapping
from typing import Any

KNOWN_SAMPLING_PARAM_KEYS: frozenset[str] = frozenset({
    "temperature", "top_p", "min_p", "top_k", "max_tokens", "seed",
    "presence_penalty", "frequency_penalty",
    "reasoning_effort", "reasoning_summary", "verbosity",
    "thinking_effort", "thinking_budget_tokens",
})
_FLOAT_KEYS = frozenset({
    "temperature", "top_p", "min_p", "presence_penalty", "frequency_penalty",
})
_INT_KEYS = frozenset({"top_k", "max_tokens", "seed", "thinking_budget_tokens"})
_STRING_KEYS = frozenset({
    "reasoning_effort", "reasoning_summary", "verbosity", "thinking_effort",
})

def validate_sampling_params(params: Mapping[str, Any]) -> list[str]:
    """Return validation errors for ``params``; empty list means valid."""
    errors: list[str] = []
    for key, value in params.items():
        if key not in KNOWN_SAMPLING_PARAM_KEYS:
            errors.append(f"unknown sampling param '{key}'")
            continue
        if isinstance(value, bool):
            errors.append(f"'{key}' must not be a boolean")
        elif key in _FLOAT_KEYS and not isinstance(value, (int, float)):
            errors.append(f"'{key}' must be a number")
        elif key in _INT_KEYS and not isinstance(value, int):
            errors.append(f"'{key}' must be an integer")
        elif key in _STRING_KEYS and not isinstance(value, str):
            errors.append(f"'{key}' must be a string")
    return errors

def params_to_tuple(params: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(params.items()))

def params_to_dict(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    return dict(pairs)
```

Then MOVE the five `_*_setting_from_sources` helpers from
`console_session_settings.py:1561-1640` into this module under public names
(drop the leading underscore), and in `console_session_settings.py` replace
the moved defs with:

```python
from tldw_chatbook.Chat.sampling_params import (
    bool_setting_from_sources as _bool_setting_from_sources,
    float_setting_from_sources as _float_setting_from_sources,
    optional_float_setting_from_sources as _optional_float_setting_from_sources,
    optional_int_setting_from_sources as _optional_int_setting_from_sources,
    optional_string_setting_from_sources as _optional_string_setting_from_sources,
)
```

- [ ] **Step 4: Run new tests + the byte-identical regression**

Run: `pytest Tests/Chat/test_sampling_params.py Tests/Chat/test_console_session_settings.py Tests/Chat/test_console_settings_defaults.py -v`
Expected: all PASS — the two existing suites green prove the move changed
no session-default behavior.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/sampling_params.py tldw_chatbook/Chat/console_session_settings.py Tests/Chat/test_sampling_params.py
git commit -m "feat: shared sampling-param keys/validator + merge helpers (TASK-32477)"
```

---

### Task 2: Registry entry `params` (amends ADR-146)

**Files:**
- Modify: `tldw_chatbook/Chat/custom_endpoint_registry.py` (`CustomEndpointEntry`, `_parse_models` area:353, `load_custom_endpoints`:109, `build_entry_mutation`:207, `validate_entry`:237)
- Test: `Tests/Chat/test_custom_endpoint_registry.py` (exists — extend)

**Interfaces:**
- Consumes: Task 1's `validate_sampling_params`, `params_to_tuple`, `params_to_dict`.
- Produces: `CustomEndpointEntry.params: tuple[tuple[str, object], ...]` (default `()`), consumed by the resolver (Task 5) and the endpoint modal (Task 9).

- [ ] **Step 1: Write the failing tests**

```python
# appended to Tests/Chat/test_custom_endpoint_registry.py
def test_entry_loads_params_table():
    config = {"custom_endpoints": {"qwen-local": {
        "display_name": "Qwen Local", "family": "llama_cpp",
        "base_url": "http://127.0.0.1:8080",
        "params": {"temperature": 0.2, "top_k": 40},
    }}}
    entries = load_custom_endpoints(config)
    assert entries["qwen-local"].params == (("temperature", 0.2), ("top_k", 40))

def test_entry_without_params_unchanged():
    config = {"custom_endpoints": {"plain": {
        "display_name": "Plain", "family": "ollama",
        "base_url": "http://127.0.0.1:11434",
    }}}
    assert load_custom_endpoints(config)["plain"].params == ()

def test_invalid_param_key_drops_entry_with_warning(caplog):
    config = {"custom_endpoints": {"bad": {
        "display_name": "Bad", "family": "ollama",
        "base_url": "http://127.0.0.1:11434",
        "params": {"temprature": 0.2},
    }}}
    with caplog.at_level(logging.WARNING):
        entries = load_custom_endpoints(config)
    assert "bad" not in entries
    assert any("bad" in record.getMessage() for record in caplog.records)

def test_validate_entry_reports_param_errors():
    errors = validate_entry(
        "Name", "ollama", "http://127.0.0.1:11434",
        params={"seed": "not-an-int"},
    )
    assert any("seed" in error for error in errors)

def test_entry_mutation_round_trips_params():
    entry = CustomEndpointEntry(
        slug="qwen-local", display_name="Qwen Local", family="llama_cpp",
        base_url="http://127.0.0.1:8080",
        params=(("temperature", 0.2),),
    )
    mutation = build_entry_mutation(entry)
    assert mutation["custom_endpoints.qwen-local"]["params"] == {
        "temperature": 0.2,
    }
    reloaded = load_custom_endpoints(
        {"custom_endpoints": {
            "qwen-local": mutation["custom_endpoints.qwen-local"],
        }}
    )
    assert reloaded["qwen-local"].params == (("temperature", 0.2),)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Chat/test_custom_endpoint_registry.py -k params -v`
Expected: FAIL (`TypeError: load_custom_endpoints...` / assertion on missing `.params`)

- [ ] **Step 3: Implement**

- Add field to `CustomEndpointEntry`: `params: tuple[tuple[str, object], ...] = ()` (append after `created_from` so positional constructors keep working).
- Add `_parse_params(value: object) -> tuple[tuple[str, object], ...]`: non-Mapping → `()`; Mapping → validate via `validate_sampling_params`; any error → raise `ValueError` so `load_custom_endpoints`' existing invalid-entry drop logs the slug and skips.
- `load_custom_endpoints`: `params=_parse_params(raw.get("params"))`.
- `build_entry_mutation`: include `"params": params_to_dict(entry.params)` only when non-empty.
- `validate_entry(display_name, family, base_url, *, params: Mapping | None = None)`: extend the signature and append `validate_sampling_params(params or {})` errors.

- [ ] **Step 4: Run the registry suite**

Run: `pytest Tests/Chat/test_custom_endpoint_registry.py -v`
Expected: PASS (new + all pre-existing)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/custom_endpoint_registry.py Tests/Chat/test_custom_endpoint_registry.py
git commit -m "feat: registry entry params table (ADR-147 amends ADR-146, TASK-32477)"
```

---

### Task 3: `AgentDefinition` gains `provider` + `params`

**Files:**
- Modify: `tldw_chatbook/Agents/agent_models.py` (`AgentDefinition`:358-376, `validate_agent_definition`:379-404, `definition_fingerprint`:407-421, definition dict:417, `definition_from_row`:424-434)
- Test: `Tests/Agents/test_agent_models.py` (exists — extend)

**Interfaces:**
- Consumes: Task 1's `validate_sampling_params`, `params_to_dict`, `params_to_tuple`.
- Produces: `AgentDefinition.provider: str`, `AgentDefinition.params: tuple[tuple[str, object], ...]` — consumed by the resolver (Task 5), spawn integration (Task 6), schema builder (Task 7), DB layer (Task 4), settings panel (Task 9).

- [ ] **Step 1: Write the failing tests**

```python
# appended to Tests/Agents/test_agent_models.py
def test_definition_provider_and_params_default_empty():
    defn = AgentDefinition(name="reader", instructions="Read files.")
    assert defn.provider == "" and defn.params == ()
    assert validate_agent_definition(defn) == []

def test_definition_rejects_unknown_provider():
    defn = AgentDefinition(
        name="reader", instructions="Read files.", provider="not-a-provider"
    )
    assert any("provider" in e for e in validate_agent_definition(defn))

def test_definition_accepts_custom_ep_slug_form():
    defn = AgentDefinition(
        name="reader", instructions="Read files.", provider="custom-ep:qwen-local"
    )
    assert validate_agent_definition(defn) == []

def test_definition_rejects_bad_custom_ep_slug():
    defn = AgentDefinition(
        name="reader", instructions="Read files.", provider="custom-ep:BAD SLUG"
    )
    assert validate_agent_definition(defn) != []

def test_definition_rejects_unknown_param_key():
    defn = AgentDefinition(
        name="reader", instructions="Read files.",
        params=(("temprature", 0.2),),
    )
    assert any("temprature" in e for e in validate_agent_definition(defn))

def test_fingerprint_legacy_shape_unchanged_for_model_only_preset():
    # provider/params enter the fingerprint ONLY when set, so a legacy
    # model-only preset keeps its pre-ADR-147 fingerprint (the audit
    # identity persisted on existing run rows stays comparable).
    defn = AgentDefinition(name="reader", instructions="Read files.", model="m1")
    import hashlib, json
    legacy = hashlib.sha256(json.dumps({
        "instructions": defn.instructions,
        "tool_allowlist": sorted(defn.tool_allowlist),
        "model": defn.model,
    }, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    assert definition_fingerprint(defn) == legacy

def test_fingerprint_changes_with_provider():
    base = AgentDefinition(name="reader", instructions="Read files.", model="m1")
    routed = AgentDefinition(
        name="reader", instructions="Read files.", model="m1", provider="ollama"
    )
    assert definition_fingerprint(base) != definition_fingerprint(routed)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Agents/test_agent_models.py -k "provider or params or fingerprint" -v`
Expected: FAIL (`TypeError: unexpected keyword argument 'provider'`)

- [ ] **Step 3: Implement**

Append two fields AFTER `enabled` (positional-constructor safety):

```python
    provider: str = ""
    params: tuple[tuple[str, object], ...] = ()
```

Update the class docstring: replace the "``model`` overrides the parent's
model on the SAME provider endpoint" sentence with the ADR-147 semantics
(model on the preset's `provider`; `model` without `provider` keeps the
legacy same-endpoint behavior; `params` are the role-owned top layer of the
child's sampling stack).

In `validate_agent_definition`, append:

```python
    if defn.provider:
        if defn.provider.startswith("custom-ep:"):
            # Lazy: keeps Agents/ -> Chat/ edges out of module import time.
            from tldw_chatbook.Chat.custom_endpoint_registry import (
                SLUG_PATTERN,
                split_custom_endpoint_id,
            )
            slug = split_custom_endpoint_id(defn.provider)
            if slug is None or not SLUG_PATTERN.fullmatch(slug):
                errors.append("provider custom-ep id has an invalid slug")
        else:
            from tldw_chatbook.Chat.console_provider_support import (
                supported_console_provider_readiness_keys,
            )
            from tldw_chatbook.Chat.provider_readiness import provider_config_key
            if provider_config_key(defn.provider) not in set(
                supported_console_provider_readiness_keys()
            ):
                errors.append(
                    f"provider '{defn.provider}' is not a known provider id"
                )
    errors.extend(validate_sampling_params(params_to_dict(defn.params)))
```

In `definition_fingerprint`, keep legacy payloads byte-identical unless the
new fields are set:

```python
    payload_dict = {
        "instructions": defn.instructions,
        "tool_allowlist": sorted(defn.tool_allowlist),
        "model": defn.model,
    }
    if defn.provider:
        payload_dict["provider"] = defn.provider
    if defn.params:
        payload_dict["params"] = [list(pair) for pair in defn.params]
    payload = json.dumps(payload_dict, sort_keys=True)
```

In the definition dict (was :417) add `"provider": defn.provider` and
`"params": params_to_dict(defn.params)`. In `definition_from_row` add
`provider=row.get("provider", "")` and
`params=params_to_tuple(row.get("params", {}))` — the DB layer hands over
`params` already JSON-decoded (same contract as `tool_allowlist`).

- [ ] **Step 4: Run the module suite**

Run: `pytest Tests/Agents/test_agent_models.py -v`
Expected: PASS (new + all pre-existing)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/agent_models.py Tests/Agents/test_agent_models.py
git commit -m "feat: AgentDefinition provider + params with legacy-stable fingerprint (TASK-32477)"
```

---

### Task 4: AgentRuns DB schema v16 — preset routing columns + run snapshot

**Files:**
- Modify: `tldw_chatbook/DB/AgentRuns_DB.py` (`_CURRENT_SCHEMA_VERSION`:57, `agent_definitions` DDL:281, idempotent ALTERs:385-424, `create_agent_definition`:1142, `update_agent_definition`:1178, definition row decode, `create_run`:1061-1141)
- Create: `tldw_chatbook/DB/migrations/agent_runs_v15_to_v16_agent_routing.sql` (repo keeps per-version SQL files, e.g. `agent_runs_v14_to_v15_runtime_owner.sql`)
- Test: `Tests/DB/test_agent_runs_db.py` (exists — extend)

**Interfaces:**
- Consumes: Task 3's `AgentDefinition.provider/.params`.
- Produces: `create_run(..., resolved_provider=None, resolved_model=None, resolved_base_url=None, resolved_params_json=None)`; definition rows carry `provider` + `params` (JSON-decoded) — consumed by Task 6 (snapshot write) and Task 8 (snapshot read).

- [ ] **Step 1: Write the failing tests**

```python
# appended to Tests/DB/test_agent_runs_db.py
def test_schema_v16_definition_columns(db):
    cols = {row[1] for row in db._conn.execute(
        "PRAGMA table_info(agent_definitions)").fetchall()}
    assert {"provider", "params_json"} <= cols

def test_schema_v16_run_snapshot_columns(db):
    cols = {row[1] for row in db._conn.execute(
        "PRAGMA table_info(agent_runs)").fetchall()}
    assert {"resolved_provider", "resolved_model",
            "resolved_base_url", "resolved_params_json"} <= cols

def test_definition_round_trip_with_routing(db):
    defn = AgentDefinition(
        name="implementer", instructions="Implement the task.",
        provider="custom-ep:qwen-local", model="qwen3.8-27b",
        params=(("temperature", 0.2),),
    )
    defn_id = db.create_agent_definition(defn)
    loaded = db.get_agent_definition(defn_id)  # use the existing getter name
    assert loaded.provider == "custom-ep:qwen-local"
    assert loaded.params == (("temperature", 0.2),)

def test_definition_legacy_defaults(db):
    defn = AgentDefinition(name="reader", instructions="Read files.")
    db.create_agent_definition(defn)
    loaded = db.get_agent_definition_by_name("reader")  # existing getter name
    assert loaded.provider == "" and loaded.params == ()

def test_run_snapshot_persists(db, conversation):
    run_id = db.create_run(
        conversation_id=conversation, status="running",
        resolved_provider="custom-ep:qwen-local", resolved_model="qwen3.8-27b",
        resolved_base_url="http://127.0.0.1:8080",
        resolved_params_json='{"temperature": 0.2}',
    )  # shape to the existing create_run signature; kwargs are new
    row = db.get_run(run_id)
    assert row["resolved_provider"] == "custom-ep:qwen-local"
    assert row["resolved_params_json"] == '{"temperature": 0.2}'

def test_run_snapshot_defaults_null(db, conversation):
    run_id = db.create_run(conversation_id=conversation, status="running")
    row = db.get_run(run_id)
    assert row["resolved_provider"] is None
```

(Fixture names `db` / `conversation` and the getter names `get_agent_definition`
/ `get_run` are placeholders ONLY in the sense of "match the file's existing
fixtures/accessors" — open `Tests/DB/test_agent_runs_db.py` and reuse its
fixtures and the DB class's real method names; do not invent new ones.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/DB/test_agent_runs_db.py -k "v16 or routing or snapshot" -v`
Expected: FAIL (columns missing / unexpected kwargs)

- [ ] **Step 3: Implement**

- `_CURRENT_SCHEMA_VERSION = 16`.
- Fresh-table DDL for `agent_definitions` gains
  `provider TEXT NOT NULL DEFAULT ''` and `params_json TEXT NOT NULL DEFAULT '{}'`.
- In `_initialize_schema`, after the existing ALTER block (line ~424):

```python
            # v15->v16 (ADR-147, TASK-32477): preset routing fields on
            # agent_definitions; resolved-target snapshot on agent_runs.
            # Same idempotent-ALTER mechanism as every column above.
            definition_columns = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(agent_definitions)"
                ).fetchall()
            }
            if "provider" not in definition_columns:
                conn.execute(
                    "ALTER TABLE agent_definitions ADD COLUMN provider "
                    "TEXT NOT NULL DEFAULT ''"
                )
            if "params_json" not in definition_columns:
                conn.execute(
                    "ALTER TABLE agent_definitions ADD COLUMN params_json "
                    "TEXT NOT NULL DEFAULT '{}'"
                )
            for column in (
                "resolved_provider", "resolved_model",
                "resolved_base_url", "resolved_params_json",
            ):
                if column not in existing_columns:
                    conn.execute(
                        f"ALTER TABLE agent_runs ADD COLUMN {column} TEXT"
                    )
```

- `DB/migrations/agent_runs_v15_to_v16_agent_routing.sql` carries the same
  five ALTERs (matching the repo's per-version SQL file pattern).
- `create_agent_definition` / `update_agent_definition`: INSERT/UPDATE gain
  `provider` and `params_json` (`json.dumps(params_to_dict(defn.params))`).
- The definition-row decoder gains `provider=row["provider"]` and
  `params=json.loads(row["params_json"])` so `definition_from_row` receives
  them decoded (same contract as `tool_allowlist`).
- `create_run`: add four keyword-only args defaulting `None`; extend the
  INSERT column list (was :1121-1134) unconditionally — post-migration the
  columns always exist.

- [ ] **Step 4: Run the DB suite**

Run: `pytest Tests/DB/test_agent_runs_db.py Tests/Agents/test_agent_runs_db_connection_reuse.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/DB/AgentRuns_DB.py tldw_chatbook/DB/migrations/agent_runs_v15_to_v16_agent_routing.sql Tests/DB/test_agent_runs_db.py
git commit -m "feat: AgentRuns schema v16 — preset routing + resolved-target snapshot (TASK-32477)"
```

---

### Task 5: `Agents/agent_routing.py` — the pure resolver + six-layer params

**Files:**
- Create: `tldw_chatbook/Agents/agent_routing.py`
- Modify: `tldw_chatbook/Chat/console_session_settings.py` (`build_default_console_session_settings`:509 gains an optional keyword-only `extra_sources`)
- Test: `Tests/Agents/test_agent_routing.py` (new)

**Interfaces:**
- Consumes: Task 1 (`Chat/sampling_params.py`), Task 2 (`CustomEndpointEntry.params`), Task 3 (`AgentDefinition.provider/.params`), `load_custom_endpoints` / `entry_for` / `split_custom_endpoint_id` (`Chat/custom_endpoint_registry.py`), `supported_console_provider_readiness_keys` (`Chat/console_provider_support.py:311`), `provider_config_key` / `get_provider_readiness` (`Chat/provider_readiness.py:344,476`), `_setting` (`Agents/run_log.py`).
- Produces (consumed by Tasks 6-9):
  - `AgentsRoutingConfig` + `load_agents_routing_config() -> AgentsRoutingConfig`
  - `RoutingError(Exception)` with `.code: str` and `.level: str`
  - `SpawnTarget(provider: str, model: str, base_url: str | None, params: tuple[tuple[str, object], ...], source: str)`
  - `resolve_spawn_target(app_config, *, parent_provider, parent_model, preset=None, override_provider="", override_model="", routing, readiness=None) -> SpawnTarget`
  - `resolve_child_params(app_config, provider, model, *, preset_params=()) -> tuple[tuple[str, object], ...]`
  - `allowlist_matches(allowlist, provider, model) -> bool` (public: the settings panel's stale-entry check reuses it)

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Agents/test_agent_routing.py
from tldw_chatbook.Agents.agent_models import AgentDefinition
from tldw_chatbook.Agents.agent_routing import (
    AgentsRoutingConfig, RoutingError, allowlist_matches,
    load_agents_routing_config, resolve_child_params, resolve_spawn_target,
)

READY = lambda cfg, provider: None                      # readiness fake: ready
NOT_READY = lambda cfg, provider: "no API key"          # readiness fake: blocked
CFG_OFF = AgentsRoutingConfig()                         # everything default
CFG_ON = AgentsRoutingConfig(
    spawn_override_enabled=True,
    spawn_override_allowlist=("llama_cpp", "custom-ep:qwen-local/qwen3.8-*"),
)
PRESET = AgentDefinition(
    name="implementer", instructions="Implement.",
    provider="custom-ep:qwen-local", model="qwen3.8-27b",
    params=(("temperature", 0.2),),
)
APP_CFG = {
    "custom_endpoints": {"qwen-local": {
        "display_name": "Qwen Local", "family": "llama_cpp",
        "base_url": "http://127.0.0.1:8080",
        "params": {"top_k": 40},
    }},
    "chat_defaults": {"temperature": 0.9},
    "api_settings": {"llama_cpp": {"model": "llama-3-8b"}},
}

def test_inherit_parent_when_nothing_configured():
    t = resolve_spawn_target(APP_CFG, parent_provider="moonshot",
        parent_model="kimi-k2", routing=CFG_OFF, readiness=READY)
    assert (t.provider, t.model, t.source) == ("moonshot", "kimi-k2", "inherit")
    assert t.base_url is None

def test_preset_routes_across_providers():
    t = resolve_spawn_target(APP_CFG, parent_provider="moonshot",
        parent_model="kimi-k2", preset=PRESET, routing=CFG_OFF, readiness=READY)
    assert t.provider == "custom-ep:qwen-local" and t.model == "qwen3.8-27b"
    assert t.base_url == "http://127.0.0.1:8080" and t.source == "preset"

def test_preset_model_only_keeps_parent_endpoint():
    legacy = AgentDefinition(name="r", instructions="i", model="qwen3.8-27b")
    t = resolve_spawn_target(APP_CFG, parent_provider="llama_cpp",
        parent_model="llama-3-8b", preset=legacy, routing=CFG_OFF, readiness=READY)
    assert (t.provider, t.model) == ("llama_cpp", "qwen3.8-27b")

def test_subagent_default_used_when_no_preset_or_override():
    cfg = AgentsRoutingConfig(subagent_default_provider="llama_cpp",
                              subagent_default_model="qwen3.8-27b")
    t = resolve_spawn_target(APP_CFG, parent_provider="moonshot",
        parent_model="kimi-k2", routing=cfg, readiness=READY)
    assert (t.provider, t.model, t.source) == ("llama_cpp", "qwen3.8-27b", "default")

def test_override_refused_when_disabled():
    try:
        resolve_spawn_target(APP_CFG, parent_provider="m", parent_model="k",
            override_provider="llama_cpp", routing=CFG_OFF, readiness=READY)
        assert False, "expected RoutingError"
    except RoutingError as e:
        assert e.code == "override_disabled" and e.level == "override"

def test_override_provider_not_allowlisted():
    try:
        resolve_spawn_target(APP_CFG, parent_provider="m", parent_model="k",
            override_provider="openai", routing=CFG_ON, readiness=READY)
        assert False
    except RoutingError as e:
        assert e.code == "provider_not_allowlisted"

def test_final_provider_guard_refuses_model_only_onto_paid_default():
    cfg = AgentsRoutingConfig(spawn_override_enabled=True,
        spawn_override_allowlist=("llama_cpp",),
        subagent_default_provider="moonshot", subagent_default_model="kimi-k2")
    try:
        resolve_spawn_target(APP_CFG, parent_provider="anthropic",
            parent_model="claude", override_model="kimi-k2-thinking",
            routing=cfg, readiness=READY)
        assert False
    except RoutingError as e:
        assert e.code == "provider_not_allowlisted"

def test_final_provider_guard_allows_model_only_on_parent_provider():
    t = resolve_spawn_target(APP_CFG, parent_provider="llama_cpp",
        parent_model="llama-3-8b", override_model="qwen3.8-27b",
        routing=CFG_ON, readiness=READY)
    assert (t.provider, t.model) == ("llama_cpp", "qwen3.8-27b")

def test_allowlist_glob_matching():
    al = ("llama_cpp/qwen3.8-*", "custom-ep:box")
    assert allowlist_matches(al, "llama_cpp", "Qwen3.8-27B")      # case-insensitive
    assert not allowlist_matches(al, "llama_cpp", "llama-3-8b")
    assert allowlist_matches(al, "custom-ep:box", "anything")
    assert not allowlist_matches(al, "custom-ep:other", "anything")

def test_unknown_endpoint_slug():
    bad = AgentDefinition(name="r", instructions="i", provider="custom-ep:gone")
    try:
        resolve_spawn_target(APP_CFG, parent_provider="m", parent_model="k",
            preset=bad, routing=CFG_OFF, readiness=READY)
        assert False
    except RoutingError as e:
        assert e.code == "unknown_endpoint_slug" and e.level == "preset"

def test_unknown_provider():
    bad = AgentDefinition(name="r", instructions="i", provider="not-a-provider")
    try:
        resolve_spawn_target(APP_CFG, parent_provider="m", parent_model="k",
            preset=bad, routing=CFG_OFF, readiness=READY)
        assert False
    except RoutingError as e:
        assert e.code == "unknown_provider"

def test_provider_not_ready():
    try:
        resolve_spawn_target(APP_CFG, parent_provider="m", parent_model="k",
            preset=PRESET, routing=CFG_OFF, readiness=NOT_READY)
        assert False
    except RoutingError as e:
        assert e.code == "provider_not_ready" and "no API key" in str(e)

def test_no_model_resolved_only_when_routed_and_unconfigured():
    preset = AgentDefinition(name="r", instructions="i", provider="custom-ep:qwen-local")
    try:
        resolve_spawn_target(APP_CFG, parent_provider="m", parent_model="k",
            preset=preset, routing=CFG_OFF, readiness=READY)
        assert False
    except RoutingError as e:
        assert e.code == "no_model_resolved"
    # inherit path with blank parent model never raises:
    t = resolve_spawn_target(APP_CFG, parent_provider="llama_cpp",
        parent_model="", routing=CFG_OFF, readiness=READY)
    assert t.model == ""

def test_builtin_provider_configured_model_fills_blank():
    preset = AgentDefinition(name="r", instructions="i", provider="llama_cpp")
    t = resolve_spawn_target(APP_CFG, parent_provider="m", parent_model="k",
        preset=preset, routing=CFG_OFF, readiness=READY)
    assert t.model == "llama-3-8b"   # from api_settings.llama_cpp.model

def test_params_precedence_preset_over_entry_over_chat_defaults():
    t = resolve_spawn_target(APP_CFG, parent_provider="moonshot",
        parent_model="kimi-k2", preset=PRESET, routing=CFG_OFF, readiness=READY)
    merged = dict(t.params)
    assert merged["temperature"] == 0.2    # preset beats chat_defaults 0.9
    assert merged["top_k"] == 40           # entry params present
    assert "streaming" not in merged

def test_params_never_come_from_parent():
    # parent params simply are not an input; resolved params come from the
    # child's own provider stack.
    t = resolve_spawn_target(APP_CFG, parent_provider="moonshot",
        parent_model="kimi-k2", routing=CFG_OFF, readiness=READY)
    assert dict(t.params)["temperature"] == 0.9   # chat_defaults, not a parent value
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Agents/test_agent_routing.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3a: Add `extra_sources` to `build_default_console_session_settings`**

In `console_session_settings.py:509`:

```python
def build_default_console_session_settings(
    app_config, provider=None, model=None, *, extra_sources=()
):
```

and change the source tuple (was :538) to:

```python
    default_sources = (model_profile, saved_defaults, *extra_sources, chat_defaults, provider_settings)
```

Each `extra_sources` entry is a Mapping of param name → value slotting
between saved defaults and chat_defaults. Default `()` keeps behavior
byte-identical.

- [ ] **Step 3b: Implement `Agents/agent_routing.py`**

```python
"""Pure spawn-target routing for sub-agents (ADR-147, TASK-32477).

Resolution order (each level fills only blanks left above): ad-hoc spawn
args (gated) -> preset fields -> [agents] sub-agent default -> inherit
parent. Params are NEVER inherited from the parent; they resolve through
the six-layer stack in resolve_child_params.
"""
from __future__ import annotations

import fnmatch
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from tldw_chatbook.Agents.agent_models import AgentDefinition
from tldw_chatbook.Chat.sampling_params import (
    KNOWN_SAMPLING_PARAM_KEYS, params_to_dict, params_to_tuple,
)

logger = logging.getLogger(__name__)

DEFAULT_SUBAGENT_DEFAULT_PROVIDER = ""
DEFAULT_SUBAGENT_DEFAULT_MODEL = ""
DEFAULT_SPAWN_OVERRIDE_ENABLED = False
DEFAULT_SPAWN_OVERRIDE_ALLOWLIST: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentsRoutingConfig:
    subagent_default_provider: str = DEFAULT_SUBAGENT_DEFAULT_PROVIDER
    subagent_default_model: str = DEFAULT_SUBAGENT_DEFAULT_MODEL
    spawn_override_enabled: bool = DEFAULT_SPAWN_OVERRIDE_ENABLED
    spawn_override_allowlist: tuple[str, ...] = DEFAULT_SPAWN_OVERRIDE_ALLOWLIST


def load_agents_routing_config() -> AgentsRoutingConfig:
    """Read the [agents] routing keys via the same _setting accessor the
    other agent keys use (Agents/run_log.py)."""
    from tldw_chatbook.Agents.run_log import _setting

    raw = _setting("spawn_override_allowlist", DEFAULT_SPAWN_OVERRIDE_ALLOWLIST) or ()
    if isinstance(raw, str):
        raw = (raw,)
    return AgentsRoutingConfig(
        subagent_default_provider=str(
            _setting("subagent_default_provider", "") or "").strip(),
        subagent_default_model=str(
            _setting("subagent_default_model", "") or "").strip(),
        spawn_override_enabled=bool(_setting("spawn_override_enabled", False)),
        spawn_override_allowlist=tuple(
            str(entry).strip() for entry in raw if str(entry).strip()),
    )


class RoutingError(Exception):
    """A refused spawn routing. ``code`` is machine-readable; ``level``
    names the failing resolution level (override / preset / default)."""

    def __init__(self, code: str, message: str, *, level: str) -> None:
        super().__init__(message)
        self.code = code
        self.level = level


@dataclass(frozen=True)
class SpawnTarget:
    provider: str
    model: str
    base_url: str | None
    params: tuple[tuple[str, object], ...]
    source: str  # "override" | "preset" | "default" | "inherit"


def allowlist_matches(allowlist: tuple[str, ...], provider: str, model: str) -> bool:
    """Bare 'provider' entries match any model; 'provider/glob' entries
    additionally fnmatch the model (case-insensitive)."""
    for entry in allowlist:
        entry_provider, sep, glob = entry.partition("/")
        if entry_provider != provider:
            continue
        if not sep:
            return True
        if model and fnmatch.fnmatchcase(model.lower(), glob.lower()):
            return True
    return False


def _allowlist_covers_provider(allowlist: tuple[str, ...], provider: str) -> bool:
    return any(entry.partition("/")[0] == provider for entry in allowlist)


def _configured_model_for(app_config: Mapping[str, Any], provider: str) -> str:
    """The provider's configured/default model (mirrors the console
    selection builder's configured_model lookup); '' when none. Registry
    entries carry no default model by design (ADR-146)."""
    from tldw_chatbook.Chat.custom_endpoint_registry import split_custom_endpoint_id
    from tldw_chatbook.Chat.provider_readiness import provider_config_key

    if split_custom_endpoint_id(provider):
        return ""
    api_settings = app_config.get("api_settings")
    if not isinstance(api_settings, Mapping):
        return ""
    section = api_settings.get(provider_config_key(provider))
    if not isinstance(section, Mapping):
        return ""
    for key in ("model", "api_model", "default_model"):
        value = str(section.get(key) or "").strip()
        if value:
            return value
    return ""


def _default_readiness(app_config: Mapping[str, Any], provider: str) -> str | None:
    """None when the provider may be sent to, else the human-readable block.
    Wraps get_provider_readiness (Chat/provider_readiness.py:476) — the same
    readiness the Console send path projects, custom-ep aware per ADR-146."""
    from tldw_chatbook.Chat.provider_readiness import get_provider_readiness

    readiness = get_provider_readiness(app_config, provider)
    return None if readiness.ready else readiness.summary


def resolve_child_params(
    app_config: Mapping[str, Any],
    provider: str,
    model: str,
    *,
    preset_params: tuple[tuple[str, object], ...] = (),
) -> tuple[tuple[str, object], ...]:
    """Six-layer stack, highest first: preset params -> per-model profile ->
    console.provider_defaults -> registry entry params -> chat_defaults ->
    api_settings scalars -> function fallbacks. Layers 2-6 are merged by
    build_default_console_session_settings; entry params ride its
    extra_sources seam; preset params overlay last."""
    from tldw_chatbook.Chat.console_session_settings import (
        build_default_console_session_settings,
    )
    from tldw_chatbook.Chat.custom_endpoint_registry import (
        entry_for, split_custom_endpoint_id,
    )

    entry_params: dict[str, object] = {}
    if split_custom_endpoint_id(provider):
        entry = entry_for(app_config, provider)
        if entry is not None:
            entry_params = params_to_dict(entry.params)
    extra = (entry_params,) if entry_params else ()
    settings = build_default_console_session_settings(
        app_config, provider, model or None, extra_sources=extra)
    merged = {
        key: getattr(settings, key)
        for key in KNOWN_SAMPLING_PARAM_KEYS
        if getattr(settings, key, None) is not None
    }
    merged.update(params_to_dict(preset_params))
    return params_to_tuple(merged)


def resolve_spawn_target(
    app_config: Mapping[str, Any],
    *,
    parent_provider: str,
    parent_model: str,
    preset: AgentDefinition | None = None,
    override_provider: str = "",
    override_model: str = "",
    routing: AgentsRoutingConfig,
    readiness: Callable[[Mapping[str, Any], str], str | None] | None = None,
) -> SpawnTarget:
    """Resolve where a spawned child runs. Raises RoutingError on refusal."""
    from tldw_chatbook.Chat.console_provider_support import (
        supported_console_provider_readiness_keys,
    )
    from tldw_chatbook.Chat.custom_endpoint_registry import (
        entry_for, split_custom_endpoint_id,
    )
    from tldw_chatbook.Chat.provider_readiness import provider_config_key

    override_provider = (override_provider or "").strip()
    override_model = (override_model or "").strip()
    if (override_provider or override_model) and not routing.spawn_override_enabled:
        raise RoutingError(
            "override_disabled",
            "ad-hoc provider/model args are disabled "
            "([agents] spawn_override_enabled = false)",
            level="override")
    if override_provider and not _allowlist_covers_provider(
            routing.spawn_override_allowlist, override_provider):
        raise RoutingError(
            "provider_not_allowlisted",
            f"provider '{override_provider}' is not in spawn_override_allowlist",
            level="override")

    if override_provider:
        provider, source = override_provider, "override"
    elif preset is not None and preset.provider:
        provider, source = preset.provider, "preset"
    elif routing.subagent_default_provider:
        provider, source = routing.subagent_default_provider, "default"
    else:
        provider, source = parent_provider, "inherit"

    model = (
        override_model
        or (preset.model if preset is not None else "")
        or routing.subagent_default_model
    )
    if not model and provider == parent_provider:
        model = parent_model
    if not model and source != "inherit":
        model = _configured_model_for(app_config, provider)
        if not model:
            raise RoutingError(
                "no_model_resolved",
                f"target provider '{provider}' has no configured/default "
                "model; set one or name a model",
                level=source)

    if (override_provider or override_model) and provider != parent_provider \
            and not allowlist_matches(
                routing.spawn_override_allowlist, provider, model):
        raise RoutingError(
            "provider_not_allowlisted",
            f"final target '{provider}/{model}' matches no allowlist entry",
            level="override")

    base_url: str | None = None
    if split_custom_endpoint_id(provider):
        entry = entry_for(app_config, provider)
        if entry is None:
            raise RoutingError(
                "unknown_endpoint_slug",
                f"registry has no endpoint '{provider}'",
                level=source)
        base_url = entry.base_url
    elif provider_config_key(provider) not in set(
            supported_console_provider_readiness_keys()):
        raise RoutingError(
            "unknown_provider",
            f"'{provider}' is not a known provider id",
            level=source)

    check = readiness if readiness is not None else _default_readiness
    blocked = check(app_config, provider)
    if blocked is not None:
        raise RoutingError("provider_not_ready", blocked, level=source)

    params = resolve_child_params(
        app_config, provider, model,
        preset_params=preset.params if preset is not None else ())
    return SpawnTarget(
        provider=provider, model=model, base_url=base_url,
        params=params, source=source)
```

(If `ProviderReadiness`'s attribute names differ from `.ready`/`.summary`,
adapt `_default_readiness` to the real type at `provider_readiness.py:476` —
unit tests always inject `readiness=`, so they never depend on it.)

- [ ] **Step 4: Run tests**

Run: `pytest Tests/Agents/test_agent_routing.py Tests/Chat/test_console_session_settings.py Tests/Chat/test_console_settings_defaults.py -v`
Expected: PASS (existing suites prove `extra_sources=()` changed nothing)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/agent_routing.py tldw_chatbook/Chat/console_session_settings.py Tests/Agents/test_agent_routing.py
git commit -m "feat: pure spawn-target resolver with six-layer params (TASK-32477)"
```

---

### Task 6: Spawn integration — resolver hook, child config, snapshot

**Files:**
- Modify: `tldw_chatbook/Agents/agent_models.py` (`AgentConfig`:449-533 gains two kw-only fields)
- Modify: `tldw_chatbook/Agents/agent_runtime.py:1450-1499` (spawn dispatch parses `provider`/`model` args)
- Modify: `tldw_chatbook/Agents/agent_service.py` (spawn closure:3115; child config:3300-3314; child kwargs:3330-3356; `_run_one`:2368; `call_model` send:1753)
- Test: `Tests/Agents/test_agent_routing_integration.py` (new; reuse the AgentService construction fixtures from `Tests/Agents/test_fleet_runtime.py`)

**Interfaces:**
- Consumes: Task 4 (`create_run` resolved_* kwargs), Task 5 (`resolve_spawn_target`, `load_agents_routing_config`, `RoutingError`, `SpawnTarget`).
- Produces: `AgentConfig.base_url: str | None`, `AgentConfig.sampling_params: tuple[tuple[str, object], ...]`; spawn rows carry resolved_* snapshots (read by Task 8).

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Agents/test_agent_routing_integration.py
# Build AgentService exactly as Tests/Agents/test_fleet_runtime.py does,
# with a capturing stub for self.chat_call.

def test_routed_preset_child_uses_own_provider_and_params(service, ...):
    # preset 'implementer' -> custom-ep:qwen-local/qwen3.8-27b params temp 0.2
    result = service.run_turn(..., messages=[{"role":"user","content":"go"}])
    child_calls = [c for c in service.chat_call.calls
                   if c["api_endpoint"] == "custom-ep:qwen-local"]
    assert child_calls, "child never called the routed provider"
    assert child_calls[0]["model"] == "qwen3.8-27b"
    assert child_calls[0]["temp"] == 0.2
    assert child_calls[0]["api_base_url"] == "http://127.0.0.1:8080"

def test_override_refusal_returns_tool_error_and_spawns_nothing(service, ...):
    # spawn_override_enabled=false; primary's stubbed model emits a
    # spawn_subagent tool call with provider=llama_cpp
    ...
    assert refusal visible in tool results and "[override_disabled]" in it
    assert no child run row was created

def test_refusal_consumes_no_fleet_slot(service, ...):
    # after the refused spawn, a subsequent legal spawn still fits the budget
    ...

def test_run_row_carries_resolved_snapshot(service, db, ...):
    ...
    row = db.get_run(child_run_id)
    assert row["resolved_provider"] == "custom-ep:qwen-local"
    assert json.loads(row["resolved_params_json"])["temperature"] == 0.2

def test_plain_spawn_snapshot_matches_parent(service, db, ...):
    # no routing configured: child inherits provider/model and the snapshot
    # still records them (source=inherit)
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Agents/test_agent_routing_integration.py -v`
Expected: FAIL (resolver never called; AgentConfig has no such fields)

- [ ] **Step 3: Implement**

a. `AgentConfig` (`agent_models.py`, after `reasoning_replay`):

```python
    base_url: str | None = field(default=None, kw_only=True)
    sampling_params: tuple[tuple[str, object], ...] = field(default=(), kw_only=True)
```

b. `agent_service.py` module level:

```python
#: AgentConfig.sampling_params keys -> chat_api_call kwarg names
#: (signature at Chat/Chat_Functions.py:899-941).
_CHAT_CALL_PARAM_MAP = {
    "temperature": "temp", "top_p": "topp", "min_p": "minp", "top_k": "topk",
    "max_tokens": "max_tokens", "seed": "seed",
    "presence_penalty": "presence_penalty", "frequency_penalty": "frequency_penalty",
    "reasoning_effort": "reasoning_effort", "reasoning_summary": "reasoning_summary",
    "verbosity": "verbosity", "thinking_effort": "thinking_effort",
    "thinking_budget_tokens": "thinking_budget_tokens",
}
```

c. `call_model` (`agent_service.py`), immediately before `resp = self.chat_call(...)`
(was :1753):

```python
            for param_key, param_value in config.sampling_params:
                call_kwargs[_CHAT_CALL_PARAM_MAP[param_key]] = param_value
            if config.base_url:
                call_kwargs["api_base_url"] = config.base_url
```

d. `spawn` closure signature (was :3115) and resolver hook immediately after
the `resolved` definition lookup completes (before the budget work at
~:3229):

```python
        def spawn(
            spawn_task: str,
            *,
            allowed_tools: tuple[str, ...] | None = None,
            agent: str | None = None,
            inline: bool = False,
            provider: str = "",
            model: str = "",
        ) -> ToolResult:
```

```python
            # ADR-147: resolve WHERE the child runs before touching budget
            # or fleet capacity. A refusal is an admission refusal: no child
            # run, no slot consumed, error back to the supervisor model.
            try:
                target = resolve_spawn_target(
                    self._app_config,
                    parent_provider=api_endpoint,
                    parent_model=config.model,
                    preset=resolved,
                    override_provider=provider,
                    override_model=model,
                    routing=load_agents_routing_config(),
                )
            except RoutingError as err:
                return SpawnAdmissionRefusal(
                    ok=False, error=f"[{err.code}] {err}"
                )
```

(`self._app_config`: use the app-config reference the service already holds
for readiness/gateway wiring; if the service holds none, thread it from the
constructor's call site in `Chat/console_chat_controller.py`, which builds
the `ConsoleProviderSelection`.)

e. Child construction: replace `child_model = config.model` /
`if resolved.model: child_model = resolved.model` (was :3282-3291) with the
resolved target (the preset `model`-only legacy path is INSIDE the resolver,
so delete the inline override here), and extend `AgentConfig`/`child_kwargs`:

```python
            child_model = target.model or config.model
            child_config = AgentConfig(
                model=child_model,
                ...,  # unchanged fields as today
                base_url=target.base_url,
                sampling_params=target.params,
            )
            child_kwargs = dict(
                ...,
                api_endpoint=target.provider,
                resolved_provider=target.provider,
                resolved_model=child_model,
                resolved_base_url=target.base_url,
                resolved_params_json=(
                    json.dumps(params_to_dict(target.params)) if target.params else None
                ),
            )
```

f. `_run_one` (was :2368): add four keyword-only args
(`resolved_provider=None, resolved_model=None, resolved_base_url=None,
resolved_params_json=None`) and forward them to its `create_run(...)` call.

g. `agent_runtime.py` spawn dispatch (was :1458-1491):

```python
                        override_provider = str(call.args.get("provider") or "").strip()
                        override_model = str(call.args.get("model") or "").strip()
```

and pass both into `deps.spawn(...)` on the named and unnamed branches alike.

- [ ] **Step 4: Run integration + neighboring suites**

Run: `pytest Tests/Agents/test_agent_routing_integration.py Tests/Agents/test_fleet_runtime.py Tests/Agents/test_agent_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/agent_models.py tldw_chatbook/Agents/agent_runtime.py tldw_chatbook/Agents/agent_service.py Tests/Agents/test_agent_routing_integration.py
git commit -m "feat: spawn routing integration — resolver hook, child config, snapshot (TASK-32477)"
```

---

### Task 6B: `_StreamingModelAdapter` honors per-call routing kwargs (PLAN AMENDMENT — added during execution, ledger R9)

**Why this task exists:** Task 6's review confirmed with file:line evidence that the
production send path discarded everything Task 6 emits: `_StreamingModelAdapter.chat_call`
(`tldw_chatbook/Chat/console_agent_bridge.py:1983`, class at :1798, wired at :3951)
accepts `api_endpoint` but never references it, drops `api_base_url` and all 13 sampling
kwargs into `**_ignored` (:1992), and always streams via `self._resolution` — the parent's
fixed `ConsoleProviderResolution` (:2064-2066). Without this task the entire routing
feature is inert in production: children record a resolved_* snapshot while actually
streaming from the parent's provider. The capturing-stub tests in Task 6 could not see
this; the brief's known-unknown trace found it.

**Files:**
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py` (`_StreamingModelAdapter.chat_call`:1983-2177; resolution construction sites :3173/:3397/:3704 show how `ConsoleProviderResolution` is built)
- Test: `Tests/Chat/test_console_agent_bridge.py` (exists — extend; mirror its adapter/gateway fixture patterns)

**Interfaces:**
- Consumes: Task 6's per-call kwargs at `chat_call`: `api_endpoint` (resolved provider id, possibly `custom-ep:<slug>`), `model`, `api_base_url`, and the 13 `_CHAT_CALL_PARAM_MAP` kwargs (`temp`, `topp`, `minp`, `topk`, `max_tokens`, `seed`, `presence_penalty`, `frequency_penalty`, `reasoning_effort`, `reasoning_summary`, `verbosity`, `thinking_effort`, `thinking_budget_tokens`).
- Produces: when a call carries routing kwargs that differ from the parent resolution, the bytes stream from the CHILD's provider/model/base_url with the child's sampling params; when no routing kwargs are present (plain parent/child-inherit calls), behavior is byte-identical to today.

- [ ] **Step 1: Write the failing tests**

Build the adapter with a capturing fake gateway (mirror the existing fixtures in
`Tests/Chat/test_console_agent_bridge.py`). Cases:

```python
def test_per_call_routing_kwargs_re_resolve_and_forward():
    # adapter built with parent resolution moonshot/kimi-k2;
    # chat_call(api_endpoint="custom-ep:qwen-local", model="qwen3.8-27b",
    #           api_base_url="http://127.0.0.1:8080", temp=0.2, topk=40, ...)
    # => gateway.stream_chat receives a resolution for the custom-ep child
    #    (provider identity + base_url + model), and temp/topk reach the
    #    gateway call — assert on the captured call, not on internals.

def test_no_routing_kwargs_uses_parent_resolution_unchanged():
    # chat_call() with only today's kwargs (messages_payload, model, tools)
    # => stream_chat receives self._resolution (identity assert) and no
    #    sampling kwargs appear that were not already forwarded before.

def test_per_call_sampling_only_reroutes_params_not_provider():
    # chat_call(temp=0.2) with api_endpoint == parent provider (or unset but
    # model == parent model): provider resolution stays the parent's,
    # params forward. (This is the inherit-with-own-params case.)

def test_unknown_per_call_provider_is_a_loud_error_not_silent_parent_fallback():
    # api_endpoint="not-a-provider" => error surfaced to the run loop,
    # NOT a silent stream from the parent.
```

(Names/shape are indicative — match the existing test file's real fixtures and
assertion style. The contract being pinned is the four behaviors, not the names.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Chat/test_console_agent_bridge.py -k routing -v`
Expected: FAIL (kwargs swallowed today)

- [ ] **Step 3: Implement**

- `_StreamingModelAdapter.chat_call`: stop dropping the routing kwargs. Detect
  "routed call": `api_endpoint` present AND (`api_endpoint`, `model or resolution
  model`, `api_base_url`) differs from `self._resolution`'s identity — OR any of
  the 13 sampling kwargs present.
- For a routed call, build a per-call `ConsoleProviderResolution` reusing the SAME
  builder the bridge uses for the parent (`resolve_console_provider_identity` /
  the gateway's resolution seam at :3173/:3397/:3704 — locate by symbol, these line
  numbers are approximate), passing the per-call provider id, model, and base_url.
  The builder must handle `custom-ep:<slug>` ids (the gateway is custom-ep aware per
  ADR-146; the Task 5 `_default_readiness` adaptation at agent_routing.py:119-153
  shows the exact `entry_for` → `family_execution_key` → identity chain to mirror).
- NEVER mutate `self._resolution` — the adapter is shared across concurrent
  children; build the per-call resolution as a local. If construction is expensive,
  cache keyed by `(api_endpoint, model, api_base_url)` in an instance dict —
  measure-keeping-simple first: only add the cache if construction touches I/O.
- Forward the 13 sampling kwargs (those actually provided) into
  `gateway.stream_chat(...)` — first READ what `stream_chat` does with **kwargs:
  trace until the values reach `chat_api_call` (Chat/Chat_Functions.py:899+) whose
  kwarg names Task 6's `_CHAT_CALL_PARAM_MAP` already matches. If `stream_chat`
  filters kwargs, extend ITS forwarding minimally rather than renaming here.
- Unknown/unready per-call provider: surface a loud error through the adapter's
  existing error channel (the same one stream failures use) — never fall back to
  the parent resolution silently.
- Usage-accounting labels (:2160/:2166) must reflect the per-call model when routed.

- [ ] **Step 4: Run bridge + agent suites**

Run: `python -m pytest Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_console_agent_bridge_local.py Tests/Agents/test_agent_routing_integration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_agent_bridge.py Tests/Chat/test_console_agent_bridge.py
git commit -m "feat: streaming adapter honors per-call routing kwargs (TASK-32477)"
```

**Out of scope (documented, not done):** the headless path (`_default_chat_call` →
`chat_api_call`) dispatches per-call kwargs natively but raises
`ValueError("Unsupported API endpoint")` on `custom-ep:` ids (Chat_Functions.py:1018) —
headless custom-ep routing is a documented limitation for this PR (console is the
routing surface per ADR-147); a follow-up may teach `chat_api_call` registry
resolution. Builtin-provider per-call routing on the headless path already works.

---

### Task 7: Spawn schema gating + master visibility

**Files:**
- Modify: `tldw_chatbook/Agents/tool_catalog.py` (`SPAWN_TOOL_SCHEMA`:66, `build_spawn_schema`:86-125)
- Modify: `tldw_chatbook/Agents/agent_service.py` (schema build call site :2514)
- Test: `Tests/Agents/test_tool_catalog.py` (exists — extend)

**Interfaces:**
- Consumes: Task 5's `AgentsRoutingConfig`, `allowlist_matches`; Task 3's `AgentDefinition.provider/.model`.
- Produces: `build_spawn_schema(definitions, *, override_enabled=False, override_targets: Sequence[tuple[str, tuple[str, ...]]] = ()) -> ToolSchema` — same signature consumed by the Task 6 call site.

- [ ] **Step 1: Write the failing tests**

```python
def test_spawn_schema_omits_override_args_when_disabled():
    schema = build_spawn_schema([PRESET], override_enabled=False)
    assert "provider" not in schema.parameters["properties"]
    assert "model" not in schema.parameters["properties"]

def test_spawn_schema_offers_override_args_when_enabled():
    schema = build_spawn_schema(
        [PRESET], override_enabled=True,
        override_targets=(("custom-ep:qwen-local", ("qwen3.8-27b",)),),
    )
    props = schema.parameters["properties"]
    assert props["provider"]["type"] == "string"
    assert props["model"]["type"] == "string"
    assert "custom-ep:qwen-local" in props["provider"]["description"]
    assert "qwen3.8-27b" in props["provider"]["description"]

def test_roster_lines_carry_routing():
    schema = build_spawn_schema([PRESET])
    desc = schema.parameters["properties"]["agent"]["description"]
    assert "implementer" in desc and "custom-ep:qwen-local" in desc

def test_roster_lines_omit_routing_when_unrouted():
    plain = AgentDefinition(name="reader", instructions="Read.")
    schema = build_spawn_schema([plain])
    desc = schema.parameters["properties"]["agent"]["description"]
    assert "runs on" not in desc

def test_identity_schema_unchanged_with_no_definitions():
    assert build_spawn_schema([]) is SPAWN_TOOL_SCHEMA
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Agents -k spawn_schema -v`
Expected: FAIL (`TypeError: unexpected keyword argument 'override_enabled'`)

- [ ] **Step 3: Implement**

Extend `build_spawn_schema` (`tool_catalog.py:86`):

```python
def build_spawn_schema(
    definitions: Sequence[AgentDefinition],
    *,
    override_enabled: bool = False,
    override_targets: Sequence[tuple[str, tuple[str, ...]]] = (),
) -> ToolSchema:
```

- Roster lines gain routing when a definition sets `provider` or `model`:
  `f"- {d.name} — {d.description} (runs on {d.provider or 'parent'} / {d.model or 'default model'})"`
- When `override_enabled`: add `provider` / `model` optional string
  properties; the `provider` description enumerates `override_targets` as
  `provider (models: a, b)` lines; the `model` description notes globs are
  NOT expanded here — the master picks from the enumerated models.
- Call site (`agent_service.py:2514`): build `override_targets` from
  `load_agents_routing_config()` + app config — for each allowlisted
  provider, its configured model (`api_settings` section) or, for
  `custom-ep:` entries, `CustomEndpointEntry.models` — and pass
  `override_enabled`. Keep the `SPAWN_TOOL_SCHEMA` identity return when no
  definitions AND override disabled (byte-identical pre-ADR-147 payloads).

- [ ] **Step 4: Run tests**

Run: `pytest Tests/Agents -k "spawn_schema or first_request_schema" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/tool_catalog.py tldw_chatbook/Agents/agent_service.py Tests/Agents/
git commit -m "feat: spawn schema gating + allowlist/roster visibility (TASK-32477)"
```

---

### Task 8: Resume / continuation reuses the persisted snapshot

**Files:**
- Modify: `tldw_chatbook/DB/AgentRuns_DB.py` (add `get_run_resolved_target`)
- Modify: `tldw_chatbook/Agents/agent_service.py` (the continuation branch that re-resolves a retained child's definition — search `resumed_from_run_id` and the fleet retained-transcript continuation)
- Test: `Tests/Agents/test_fleet_continuation.py` (exists — extend)

**Interfaces:**
- Consumes: Task 4 snapshot columns; Task 6's spawn wiring.
- Produces: `AgentRunsDB.get_run_resolved_target(run_id: str) -> dict | None` returning `{"provider", "model", "base_url", "params_json"}` or None for legacy/never-routed rows.

- [ ] **Step 1: Write the failing tests**

```python
# appended to Tests/Agents/test_fleet_continuation.py
def test_continuation_reuses_snapshot_after_preset_edit(service, db, ...):
    # 1. preset 'implementer' routes to custom-ep:qwen-local; run + finish a child
    # 2. EDIT the preset to provider='ollama'
    # 3. continue the retained child (send_to_agent)
    # assert: the continued child's chat_call still went to
    # custom-ep:qwen-local (snapshot), not ollama (live re-resolution)
    ...

def test_continuation_legacy_row_reroutes_live(service, db, ...):
    # simulate a pre-v16 row: UPDATE agent_runs SET resolved_provider = NULL
    # assert: continuation resolves through resolve_spawn_target as today
    ...

def test_get_run_resolved_target_none_for_plain_rows(db, conversation):
    run_id = db.create_run(conversation_id=conversation, status="running")
    assert db.get_run_resolved_target(run_id) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Agents/test_fleet_continuation.py -k snapshot -v`
Expected: FAIL (`AttributeError: get_run_resolved_target`)

- [ ] **Step 3: Implement**

a. `AgentRuns_DB.py`:

```python
    def get_run_resolved_target(self, run_id: str) -> dict | None:
        """The v16 resolved-target snapshot for ``run_id``; None for legacy
        rows (NULL snapshot) so callers fall back to live re-resolution."""
        row = self._conn.execute(  # match the file's connection accessor
            "SELECT resolved_provider, resolved_model, resolved_base_url,"
            " resolved_params_json FROM agent_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return {
            "provider": row[0], "model": row[1],
            "base_url": row[2], "params_json": row[3],
        }
```

b. In the continuation spawn path (the branch that re-resolves the retained
child's `AgentDefinition` by name): first `get_run_resolved_target(prior_run_id)`;
on a hit, build the child's `api_endpoint` / `AgentConfig.model` /
`base_url` / `sampling_params` from the snapshot and skip
`resolve_spawn_target`; on None, fall through to the Task 6 resolver path.

- [ ] **Step 4: Run tests**

Run: `pytest Tests/Agents/test_fleet_continuation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/DB/AgentRuns_DB.py tldw_chatbook/Agents/agent_service.py Tests/Agents/test_fleet_continuation.py
git commit -m "feat: continuation reuses persisted resolved-target snapshot (TASK-32477)"
```

---

### Task 9: Settings UI — preset routing, defaults, override policy, Test routing

**Files:**
- Modify: `tldw_chatbook/Widgets/settings_agents_panel.py` (`compose`:57, `_load_bulk_reader_preset`:178, `_save`:221)
- Modify: `tldw_chatbook/Widgets/Console/console_endpoint_template_modal.py` (params section)
- Test: `Tests/Widgets/test_settings_agents_panel_routing.py` (new), `Tests/Widgets/test_console_endpoint_template_params.py` (new)

**Interfaces:**
- Consumes: Task 3 fields, Task 4 DB accessors, Task 5's `resolve_spawn_target` / `load_agents_routing_config` / `allowlist_matches`, Task 2 entry `params`, `save_settings_to_cli_config` (the writer the registry module documents), the Console provider-option builder (`console_session_settings.py`, registry-aware options :402/:457).

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Widgets/test_settings_agents_panel_routing.py — Textual run_test pilot
async def test_panel_saves_preset_routing(pilot_env):
    # set provider Select to custom-ep:qwen-local, model Input, params
    # TextArea "temperature = 0.2"; save; reload definition from DB
    assert loaded.provider == "custom-ep:qwen-local"
    assert loaded.params == (("temperature", 0.2),)

async def test_panel_rejects_bad_param_key_on_save(pilot_env):
    # params "temprature = 0.2" -> save blocked, error naming the key shown
    ...

async def test_panel_flags_stale_allowlist_slug(pilot_env):
    # allowlist contains custom-ep:deleted-slug -> a warning line names it
    # AND the config value is left untouched
    ...

async def test_test_routing_reports_readiness(pilot_env, monkeypatch):
    # one ready preset, one preset targeting a deleted slug -> report lists
    # the first as ready (provider/model) and the second with
    # [unknown_endpoint_slug]
    ...

async def test_allowlist_model_glob_entries_round_trip(pilot_env):
    # save "llama_cpp/qwen3.8-*" entries; config reload preserves them
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Widgets/test_settings_agents_panel_routing.py -v`
Expected: FAIL (controls do not exist)

- [ ] **Step 3: Implement**

Agents panel (`settings_agents_panel.py`):
- Preset section: provider `Select` (options from the Console provider-option
  builder so custom-ep entries appear by `display_name`), model `Input`,
  params `TextArea` (one `key = value` per line, parsed with
  `toml`-free splitting on `=`; validated via `validate_sampling_params`;
  save blocked with the error shown when invalid).
- Defaults section: sub-agent default provider `Select` (plus an
  "(inherit parent)" empty option) and model `Input`.
- Override section: `spawn_override_enabled` `Checkbox`, allowlist
  `TextArea` (one entry per line; each line must parse as a known provider
  id or `provider/glob`; deleted `custom-ep:` slugs produce a flagged
  warning `Static`, never a silent config rewrite).
- "Test routing" `Button`: for each enabled preset and for the configured
  default, call `resolve_spawn_target(self.app_config, ...)`; render one
  line each — `preset name -> provider / model — ready` or
  `preset name -> [code] message`. No child is spawned; the resolver is pure.
- `_save` writes presets via `update_agent_definition` and the four
  `[agents]` keys via `save_settings_to_cli_config("agents", {...})`.

Endpoint modal (`console_endpoint_template_modal.py`): optional params
`TextArea` (same `key = value` grammar and validator), threaded into
`build_entry_mutation`; empty means no `params` table.

- [ ] **Step 4: Run tests**

Run: `pytest Tests/Widgets/test_settings_agents_panel_routing.py Tests/Widgets/test_console_endpoint_template_params.py Tests/UI/test_settings_agents_category.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/settings_agents_panel.py tldw_chatbook/Widgets/Console/console_endpoint_template_modal.py Tests/Widgets/test_settings_agents_panel_routing.py Tests/Widgets/test_console_endpoint_template_params.py
git commit -m "feat: settings UI for agent routing + endpoint params (TASK-32477)"
```

---

### Task 10: Config template + user guide

**Files:**
- Modify: `tldw_chatbook/config.py` (after the `[agents]` `autowake_enabled` block, :3088)
- Modify: `Docs/User_Guide/console/agent-runs-and-tools.md`

- [ ] **Step 1: Insert the commented template block**

```toml
# --- Sub-agent routing (ADR-147) ---
# Default provider/model for spawned sub-agents when neither the spawn call
# nor the named agent preset routes them. Empty = inherit the parent's.
# subagent_default_provider = ""
# subagent_default_model = ""
#
# Let the supervisor model pass ad-hoc provider/model args to
# spawn_subagent. Off by default: routing then comes only from presets and
# the default above. Ad-hoc args never carry URLs or sampling params.
# spawn_override_enabled = false
#
# Ad-hoc targets the supervisor may pick. One entry per list item: a
# provider id ("llama_cpp", "custom-ep:qwen-local") or "provider/model-glob"
# ("llama_cpp/qwen3.8-*"). Presets are user-authored and never gated.
# spawn_override_allowlist = []
```

- [ ] **Step 2: Add the user-guide section** — new section in
`agent-runs-and-tools.md`: "Routing sub-agents to other providers/models"
covering: the four-level resolution order; preset provider/model/params
fields; the `[agents]` keys; the allowlist glob form; the six-layer params
stack; the plain-spawn params behavior change; a "route by task shape"
tiering guide (fast workhorse for recon/mechanical edits; mid-tier for
routine delegation; deep reasoning only for hard, well-scoped tasks; an
intent-strong model for ambiguous judgment work); the resume snapshot rule;
and the "Test routing" button.

- [ ] **Step 3: Verify the config template parses**

Run: `python -c "import toml; toml.load('tldw_chatbook/config.py'.replace('config.py',''))"` — NO; instead: `python -c "import tldw_chatbook.config"` and eyeball the rendered template section. (The template lives inside a Python string in `config.py`; import proves no syntax break.)

- [ ] **Step 4: Commit**

```bash
git add tldw_chatbook/config.py Docs/User_Guide/console/agent-runs-and-tools.md
git commit -m "docs: agents routing config template + user guide (TASK-32477)"
```

---

### Task 11: Live verification (per `backlog/docs/lessons-live-verification.md`)

**Files:** none (evidence only; record findings in the task's Implementation Notes)

- [ ] **Step 1:** Start a real local endpoint (llama.cpp or ollama) serving a small model; register it as `custom-ep:qwen-local` with `params = { temperature = 0.2 }`.
- [ ] **Step 2:** In Settings → Agents, create preset `implementer` (provider `custom-ep:qwen-local`, model the served id); set the sub-agent default to the same; enable `spawn_override_enabled` with allowlist `["custom-ep:qwen-local/qwen*"]`; run **Test routing** — expect "ready".
- [ ] **Step 3:** In Console with a cheap cloud model, ask the master to delegate a two-step task to `implementer`; then ask it to spawn ad-hoc onto `openai` — expect a `[provider_not_allowlisted]` tool error visible in the transcript.
- [ ] **Step 4:** Inspect the child's run row: `resolved_provider = custom-ep:qwen-local`, `resolved_params_json` has `temperature: 0.2`; the run log's request shows the endpoint's params, not the master's.
- [ ] **Step 5:** Edit the preset to another provider; continue the finished child — it must stay on `custom-ep:qwen-local` (snapshot semantics).
- [ ] **Step 6:** Record pass/fail + screenshots/log excerpts in TASK-32477 Implementation Notes; then `backlog task edit 32477 -s Done` only after all ACs check.

---

## Self-Review Notes (already applied)

- Spec coverage: every spec section maps to a task — resolver/order (T5),
  params stack (T1/T5), registry params (T2), AgentDefinition (T3), DB v16
  (T4), spawn integration + security policy (T6), schema gating/visibility
  (T7), resume snapshot (T8), settings UI + Test routing (T9), config/docs
  (T10), live verification (T11).
- Type consistency: `params` is `tuple[tuple[str, object], ...]` across
  Tasks 1-6, 8, 9; `params_json` is the JSON object form at the DB
  boundary; `SpawnTarget` fields match Task 6's consumption.
- Deferred: preset `fallback_models` (TASK-32508), thinking ceiling,
  provider-scoped role matrices — all recorded as spec non-goals.
