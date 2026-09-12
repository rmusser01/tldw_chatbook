# Console Custom Endpoint Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Named, user-created custom endpoints (templated off any existing provider) selectable as Console providers, persisted in a `[custom_endpoints.<slug>]` config registry.

**Architecture:** A pure registry module owns entry data, validation, and config mutations; `custom-ep:<slug>` provider ids resolve onto an existing execution family (llama_cpp / openai_compatible / ollama) at the seams that already hold `app_config` (readiness, defaults, options, gateway, restart survival). The Conversation Settings modal gains a template-based creation flow; F9 Settings gains management. Session `base_url` remains the endpoint carrier, so the gateway's existing `selected_endpoint` seam pins the entry URL with no new wire families.

**Tech Stack:** Python 3.11, Textual 8.x, existing `provider_endpoint_contract` / `console_provider_support` / `console_provider_endpoints` seams, `save_settings_to_cli_config`, pytest + Textual Pilot.

**Spec:** `Docs/superpowers/specs/2026-09-10-console-custom-endpoint-registry-design.md`

## Global Constraints

- Provider id format is exactly `custom-ep:<slug>`; slugs are `^[a-z0-9-]+$`, max 64 chars, immutable after creation.
- Families are exactly `llama_cpp`, `openai_compatible`, `ollama`; no new wire/execution paths.
- The registry table is top-level `[custom_endpoints]` — never nested under `api_settings`.
- Secrets follow existing provider semantics: `api_key_env` (env-var name) preferred, stored `api_key` supported; display uses `safe_endpoint_display` only; never log secret values.
- Seams without `app_config` treat an unresolvable `custom-ep:*` id as family `custom` (generic OpenAI-compatible) — never crash, never drop the selection.
- Built-in `custom` / `custom_2` slots keep working unchanged; conversion is optional and one-way.
- Config writes go through `save_settings_to_cli_config` / `delete_settings_from_cli_config` (atomic mutation + cache reload), off the UI thread (`asyncio.to_thread`) exactly like the modal's Save-as-default handler.
- Only targeted tests run (files touched by each task) unless the user separately approves a full sweep.
- Validate endpoint URLs at boundaries via `validate_url` after family-appropriate normalization (`normalize_llamacpp_base_url` for `llama_cpp`).

---

### Task 1: ADR-146 — Custom endpoint registry

**Files:**
- Create: `backlog/decisions/146-console-custom-endpoint-registry.md`
- Modify: `backlog/tasks/task-32474 - Console-custom-endpoint-registry-named-endpoints-from-templates.md`

**Interfaces:**
- Consumes: the approved spec's Decisions table.
- Produces: ADR path referenced by every later task's commit messages and by the task file.

- [ ] **Step 1: Write the ADR**

Use the repo's ADR shape (see `backlog/decisions/145-console-live-thinking-presentation.md` for tone/sections: Context, Decision, Alternatives, Consequences). Record: top-level `[custom_endpoints.<slug>]` storage outside `api_settings`; `custom-ep:<slug>` ids resolving onto family execution keys with the session `base_url` as endpoint carrier; credential precedence `api_key_env` over stored `api_key`; template scope limited to family + endpoint + model list; built-in slots retained with optional one-way conversion; unresolvable-prefix fallback to family `custom`. Link the spec and TASK-32474.

- [ ] **Step 2: Link it from the task and spec**

Append to TASK-32474's Implementation Plan section: `ADR: backlog/decisions/146-console-custom-endpoint-registry.md`. Append the same link to the spec's "ADR required" bullet.

- [ ] **Step 3: Commit**

```bash
git add backlog/decisions/146-console-custom-endpoint-registry.md "backlog/tasks/task-32474 - Console-custom-endpoint-registry-named-endpoints-from-templates.md" Docs/superpowers/specs/2026-09-10-console-custom-endpoint-registry-design.md
git commit -m "docs: ADR-146 custom endpoint registry"
```

---

### Task 2: Registry module — entries, validation, slugs, config mutations

**Files:**
- Create: `tldw_chatbook/Chat/custom_endpoint_registry.py`
- Test: `Tests/Chat/test_custom_endpoint_registry.py`

**Interfaces:**
- Consumes: `save_settings_to_cli_config(section_values, *, delete_keys=None) -> bool` and `delete_settings_from_cli_config(section, keys) -> bool` from `tldw_chatbook.config`; `normalize_llamacpp_base_url`, `validate_url`.
- Produces (used by Tasks 3-7):

```python
CUSTOM_ENDPOINT_ID_PREFIX = "custom-ep:"
ENDPOINT_FAMILIES = frozenset({"llama_cpp", "openai_compatible", "ollama"})
SLUG_PATTERN = re.compile(r"^[a-z0-9-]{1,64}$")

@dataclass(frozen=True)
class CustomEndpointEntry:
    slug: str
    display_name: str
    family: str                      # one of ENDPOINT_FAMILIES
    base_url: str                    # validated, family-normalized
    api_key_env: str | None = None
    api_key: str | None = None       # stored; display/log never reads this
    models: tuple[str, ...] = ()
    created_from: str | None = None  # template provider id, informational

def split_custom_endpoint_id(provider: str | None) -> str | None:
    """Return the slug when `provider` is `custom-ep:<slug>`, else None."""

def load_custom_endpoints(app_config: Mapping[str, object]) -> dict[str, CustomEndpointEntry]:
    """Return valid entries keyed by slug; drop invalid ones with one
    logger.warning naming the slug (registry-disabled-for-that-entry only)."""

def entry_for(app_config: Mapping[str, object], provider: str | None) -> CustomEndpointEntry | None:
    """split_custom_endpoint_id + load, or None."""

def derive_slug(display_name: str, existing_slugs: Collection[str]) -> str:
    """Lowercase, non-[a-z0-9] -> '-', collapse, trim '-'; suffix -2..-99 on collision."""

def build_entry_mutation(entry: CustomEndpointEntry) -> dict[str, dict[str, object]]:
    """{'custom_endpoints.<slug>': {'display_name': ..., 'family': ...,
    'base_url': ..., 'api_key_env': ..., 'api_key': ..., 'models': [...],
    'created_from': ...}} — None fields omitted."""

def validate_entry(display_name: str, family: str, base_url: str) -> list[str]:
    """User-facing validation errors: blank/oversized name (>80 chars),
    unknown family, invalid URL after family normalization."""

def family_execution_key(family: str) -> str:
    """'llama_cpp' -> 'llama_cpp'; 'ollama' -> 'ollama'; 'openai_compatible' -> 'custom'."""

def family_normalizes_like_llama(family: str) -> bool:
    """True only for 'llama_cpp'."""
```

- [ ] **Step 1: Write the failing tests**

```python
"""Pure custom endpoint registry tests."""
from tldw_chatbook.Chat.custom_endpoint_registry import (
    CUSTOM_ENDPOINT_ID_PREFIX, CustomEndpointEntry, derive_slug,
    entry_for, family_execution_key, load_custom_endpoints,
    split_custom_endpoint_id, validate_entry, build_entry_mutation,
)

def _config_with(slug: str, **overrides) -> dict:
    entry = {"display_name": "GPU box", "family": "llama_cpp",
             "base_url": "http://192.168.1.5:8080"}
    entry.update(overrides)
    return {"custom_endpoints": {slug: entry}}

def test_split_custom_endpoint_id():
    assert split_custom_endpoint_id("custom-ep:llama-gpu") == "llama-gpu"
    assert split_custom_endpoint_id("llama_cpp") is None
    assert split_custom_endpoint_id(None) is None
    assert split_custom_endpoint_id("custom-ep:") is None

def test_load_returns_valid_entries_and_drops_invalid_with_warning(caplog):
    cfg = _config_with("ok")
    cfg["custom_endpoints"]["bad-family"] = {
        "display_name": "X", "family": "groq", "base_url": "http://h:1"}
    entries = load_custom_endpoints(cfg)
    assert set(entries) == {"ok"}
    assert entries["ok"].family == "llama_cpp"

def test_llama_family_normalizes_v1_suffix():
    entries = load_custom_endpoints(_config_with(
        "s", base_url="http://127.0.0.1:8080/v1"))
    assert entries["s"].base_url == "http://127.0.0.1:8080"

def test_derive_slug_collapses_and_uniquifies():
    assert derive_slug("GPU box llama.cpp!", {"gpu-box-llama-cpp"}) == "gpu-box-llama-cpp-2"

def test_entry_for_resolves_provider_id():
    assert entry_for(_config_with("s"), "custom-ep:s").display_name == "GPU box"
    assert entry_for({}, "custom-ep:missing") is None

def test_validate_entry_rejects_bad_family_and_url():
    assert validate_entry("N", "groq", "http://h:1") == ["Unknown endpoint family: groq."]
    assert any("http(s)" in e for e in validate_entry("N", "llama_cpp", "ftp://h"))

def test_mutation_round_trips_and_omits_none():
    entry = CustomEndpointEntry(slug="s", display_name="D", family="ollama",
                                base_url="http://127.0.0.1:11434")
    mutation = build_entry_mutation(entry)
    assert mutation == {"custom_endpoints.s": {
        "display_name": "D", "family": "ollama",
        "base_url": "http://127.0.0.1:11434", "models": []}}
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Chat/test_custom_endpoint_registry.py -q`
Expected: FAIL — `ModuleNotFoundError` / import error.

- [ ] **Step 3: Implement `tldw_chatbook/Chat/custom_endpoint_registry.py`**

Implement the Interfaces block exactly. Rules: `load_custom_endpoints` reads `app_config.get("custom_endpoints")` (non-mapping → `{}`), skips non-mapping sections, validates via `validate_entry` plus slug pattern, and `logger.warning("custom endpoint '%s' ignored: %s", slug, reason)` per drop. `base_url` normalization: llama family → `normalize_llamacpp_base_url`; others → strip trailing `/`. `models` accepts list/tuple of non-blank strings. `validate_entry` URL check: `validate_url(candidate)` where the candidate is the normalized form.

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/Chat/test_custom_endpoint_registry.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/custom_endpoint_registry.py Tests/Chat/test_custom_endpoint_registry.py
git commit -m "feat: custom endpoint registry data module (ADR-146)"
```

---

### Task 3: Provider-settings seam — custom-ep readiness and provider settings

**Files:**
- Modify: `tldw_chatbook/Chat/console_session_settings.py` (`_provider_settings` helper, ~line 1086)
- Modify: `tldw_chatbook/Chat/console_provider_support.py` (fallback identity rule)
- Test: `Tests/Chat/test_console_session_settings.py`

**Interfaces:**
- Consumes: Task 2 (`entry_for`, `family_execution_key`).
- Produces:

```python
# in custom_endpoint_registry.py
def custom_endpoint_provider_settings(
    app_config: Mapping[str, object], provider: str | None
) -> Mapping[str, object] | None:
    """Provider-settings view for a custom-ep id: the entry flattened to
    the provider-settings key aliases — {'api_base_url': entry.base_url,
    'api_url': entry.base_url, 'api_key': entry.api_key or '',
    'api_key_env': entry.api_key_env, 'model': entry.models[0] if any}.
    None when `provider` is not a resolvable custom-ep id."""
```

In `console_provider_support.resolve_console_provider_identity` (~line 216, before the DIRECT key checks): if `split_custom_endpoint_id(raw_provider)` is not None, resolve to the generic fallback identity — readiness/execution keys `custom`, `is_supported=True` — unless a later registry-aware seam overrides (this is the no-`app_config` fallback mandated by the Global Constraints).

- [ ] **Step 1: Write the failing tests** (append to `Tests/Chat/test_console_session_settings.py`)

```python
from tldw_chatbook.Chat.custom_endpoint_registry import (
    custom_endpoint_provider_settings,
)

def _registry_config() -> dict:
    return {
        "custom_endpoints": {
            "gpu": {"display_name": "GPU llama", "family": "llama_cpp",
                    "base_url": "http://192.168.1.5:8080"},
            "paid": {"display_name": "Paid compat", "family": "openai_compatible",
                     "base_url": "https://api.example.com/v1",
                     "api_key_env": "PAID_KEY"},
        }
    }

def test_custom_endpoint_readiness_is_family_readiness():
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(provider="custom-ep:gpu", model="m",
                               base_url="http://192.168.1.5:8080"),
        app_config=_registry_config(), environ={})
    assert readiness.label == "Ready"
    assert readiness.native_send_supported is True

def test_custom_endpoint_keyed_family_requires_key():
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(provider="custom-ep:paid", model="m",
                               base_url="https://api.example.com/v1"),
        app_config=_registry_config(), environ={})
    assert readiness.native_send_supported is False

def test_provider_settings_resolves_custom_endpoint_aliases():
    settings_view = custom_endpoint_provider_settings(_registry_config(), "custom-ep:paid")
    assert settings_view["api_base_url"] == "https://api.example.com/v1"
    assert settings_view["api_key_env"] == "PAID_KEY"

def test_unresolvable_custom_endpoint_falls_back_to_generic_family():
    identity = resolve_console_provider_identity("custom-ep:ghost")
    assert identity.is_supported is True
    assert identity.execution_key == "custom"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Chat/test_console_session_settings.py -k custom_endpoint -q`
Expected: FAIL — readiness treats `custom-ep:gpu` as unknown provider.

- [ ] **Step 3: Implement**

In `build_console_settings_readiness` (console_session_settings.py, ~line 729): before resolving `identity`, call `entry_for(app_config, settings.provider)`; when present, resolve the family identity (`resolve_console_provider_identity(family_execution_key(entry.family))`), use `entry.base_url` as the base URL when the session's is blank, and use `custom_endpoint_provider_settings(...)` in place of `_provider_settings(...)` output. In `_provider_settings` (~line 1086) add the custom-ep branch first. Apply the `resolve_console_provider_identity` fallback edit described in Interfaces. Add `custom_endpoint_provider_settings` to the registry module.

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/Chat/test_console_session_settings.py -q`
Expected: PASS (121 prior + 4 new).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/custom_endpoint_registry.py tldw_chatbook/Chat/console_session_settings.py tldw_chatbook/Chat/console_provider_support.py Tests/Chat/test_console_session_settings.py
git commit -m "feat: custom-ep provider settings and readiness seam"
```

---

### Task 4: Options and session defaults — entries as selectable providers

**Files:**
- Modify: `tldw_chatbook/Chat/console_session_settings.py` (`build_console_provider_options`, `_canonical_chat_provider_id`, `console_session_endpoint_survives_restart`)
- Modify: `tldw_chatbook/Widgets/Console/console_settings_modal.py` (`_provider_select_options`, passes `app_config`)
- Test: `Tests/Chat/test_console_session_settings.py`

**Interfaces:**
- Consumes: Tasks 2-3.
- Produces:

```python
def build_console_provider_options(
    providers_models: Mapping[str, Sequence[str]],
    app_config: Mapping[str, object] | None = None,
) -> list[ConsoleSettingsOption]:
    """Group-ordered built-ins (unchanged) followed by registry entries as a
    final run: value 'custom-ep:<slug>', label '<display_name>', ordered by
    creation (config file order) then display name. `app_config=None` keeps
    the built-in-only result (existing callers unchanged)."""
```

`_canonical_chat_provider_id(provider, app_config)` gains the config param (module-internal; update its three call sites in this file) and returns the entry id unchanged when `entry_for` resolves, else the fallback-`custom` readiness key. `console_session_endpoint_survives_restart` returns `True` immediately for resolvable custom-ep providers.

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_provider_options_append_registry_entries_after_builtins():
    options = build_console_provider_options({"openai": ["m"]}, app_config=_registry_config())
    values = [o.value for o in options]
    assert values.index("custom") < values.index("custom-ep:gpu")
    labels = {o.value: o.label for o in options}
    assert labels["custom-ep:gpu"] == "GPU llama"

def test_effective_configuration_resolves_custom_endpoint():
    effective = resolve_effective_chat_configuration(
        _registry_config(), provider="custom-ep:paid", model=None)
    assert effective.provider == "custom-ep:paid"
    assert effective.base_url == "https://api.example.com/v1"

def test_custom_endpoint_sessions_survive_restart():
    settings = ConsoleSessionSettings(provider="custom-ep:gpu", model="m",
                                      base_url="http://192.168.1.5:8080")
    assert console_session_endpoint_survives_restart(
        settings, app_config=_registry_config(), environ={}) is True
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Chat/test_console_session_settings.py -k "custom_endpoint or registry_entries" -q`
Expected: FAIL (no `app_config` parameter yet).

- [ ] **Step 3: Implement**

Extend `build_console_provider_options` per Interfaces (entries appended after the sorted built-ins). Update `_canonical_chat_provider_id` and its call sites (`resolve_effective_chat_configuration`, `build_canonical_chat_defaults_mutation`, `_default_persist_sections` callers in the modal do not use it directly). Add the early-`True` branch to `console_session_endpoint_survives_restart`. In the modal's `_provider_select_options` (console_settings_modal.py ~line 2219), pass `app_config=self._app_config` and label entry values with their `display_name` (entries already carry display labels from the service; the modal's `provider_display_name` relabel pass must pass registry labels through unchanged — skip relabel when the value starts with `custom-ep:`).

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/Chat/test_console_session_settings.py Tests/UI/test_console_session_settings.py -q`
Expected: PASS (UI file: 194 passing baseline; the 2 pre-existing failures on this branch are out of scope).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_session_settings.py tldw_chatbook/Widgets/Console/console_settings_modal.py Tests/Chat/test_console_session_settings.py
git commit -m "feat: registry entries as Console provider options"
```

---

### Task 5: Gateway — custom-ep sends through the family path, never endpoint-blocked

**Files:**
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py` (provider-settings lookup + endpoint-not-saved guard)
- Test: `Tests/Chat/test_console_provider_gateway.py`

**Interfaces:**
- Consumes: Tasks 2-3 (`entry_for`, `custom_endpoint_provider_settings`, `family_execution_key`).
- Produces: no new public gateway API — behavior change only, asserted through existing resolution tests.

- [ ] **Step 1: Write the failing tests** (append to `Tests/Chat/test_console_provider_gateway.py`, following that file's existing MockTransport resolution-test style)

```python
def test_custom_endpoint_resolves_family_execution_and_entry_url():
    app_config = {
        "custom_endpoints": {"gpu": {
            "display_name": "GPU llama", "family": "llama_cpp",
            "base_url": "http://192.168.1.5:8080"}},
    }
    resolved = _resolve_selection(  # existing helper in this test module for
        # ConsoleProviderSelection — reuse the file's current construction;
        # see test_console_provider_gateway.py's unsaved-endpoint tests
        provider="custom-ep:gpu", model="m",
        base_url="http://192.168.1.5:8080", app_config=app_config)
    assert resolved.execution_provider == "llama_cpp"
    assert "192.168.1.5:8080" in resolved.endpoint
    assert "not saved" not in resolved.visible_copy

def test_custom_endpoint_openai_family_uses_generic_path():
    app_config = {
        "custom_endpoints": {"paid": {
            "display_name": "Paid", "family": "openai_compatible",
            "base_url": "https://api.example.com/v1"}},
    }
    resolved = _resolve_selection(provider="custom-ep:paid", model="m",
                                  base_url="https://api.example.com/v1",
                                  app_config=app_config)
    assert resolved.execution_provider == "custom"
    assert "not saved" not in resolved.visible_copy
```

(If the module has no `_resolve_selection` helper, build the selection exactly as the existing unsaved-endpoint tests at ~lines 928-960 do and assert on the same resolved object fields those tests use.)

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Chat/test_console_provider_gateway.py -k custom_endpoint -q`
Expected: FAIL — custom-ep provider unresolved or endpoint-not-saved copy present.

- [ ] **Step 3: Implement**

In the gateway's provider-resolution path: where provider settings are fetched per key, check `entry_for(app_config, provider)` first and use `custom_endpoint_provider_settings`; where execution/readiness keys are derived, use the family identity via `resolve_console_provider_identity(family_execution_key(entry.family))`; in the endpoint-not-saved guard (the branch producing `unsaved_endpoint_copy`, ~line 1979), skip the guard for resolvable custom-ep providers (their endpoint is config-backed by construction).

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/Chat/test_console_provider_gateway.py -q`
Expected: PASS (292 baseline + 2 new).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_provider_gateway.py Tests/Chat/test_console_provider_gateway.py
git commit -m "feat: gateway routes custom-ep providers through family execution"
```

---

### Task 5.5: Gateway credential wiring for custom-ep entries

> Inserted by controller ruling after Task 5's review: entry-declared credentials
> (api_key_env / api_key) never reach `resolution.api_key` (readiness reads
> `api_settings` under the keyless family key), so a keyed entry whose declared
> credential resolves sends unauthenticated (server 401) while Task 3's UI gate
> says Ready. No other task owns send-time credential wiring.

**Files:**
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py` (resolution api_key on generic and llama paths)
- Test: `Tests/Chat/test_console_provider_gateway.py`

**Interfaces:**
- Consumes: Task 2/3 registry seams (`entry_for`, `custom_endpoint_provider_settings` — note its view exposes `api_key_env`; `resolve_provider_credential` in provider_readiness reads `api_key_env_var`, mind that alias), ADR-146 precedence env ref → stored key → none.
- Produces: no new public API. Behavior: when a custom-ep entry declares a credential and it resolves, `resolution.api_key` carries it on both the generic and llama paths (Authorization header flows; the adapter's `api_key_resolved` stamp must not suppress it); when a declared credential does NOT resolve, the gateway blocks with the existing Missing-key copy (matching Task 3's UI gate contract). Keyless entries and all non-custom-ep providers byte-identical.

- [ ] **Step 1: Write the failing tests** (append to `Tests/Chat/test_console_provider_gateway.py`, same construction style as Task 5's two tests)

```python
def test_custom_endpoint_declared_env_key_flows_to_resolution(monkeypatch):
    monkeypatch.setenv("PAID_KEY", "paid-secret")
    app_config = {
        "custom_endpoints": {"paid": {
            "display_name": "Paid", "family": "openai_compatible",
            "base_url": "https://api.example.com/v1",
            "api_key_env": "PAID_KEY"}},
    }
    resolved = _resolve_custom_ep_selection(  # Task 5's helper/construction
        provider="custom-ep:paid", model="m",
        base_url="https://api.example.com/v1", app_config=app_config)
    assert resolved.api_key == "paid-secret"
    assert resolved.execution_provider == "custom-openai-api"

def test_custom_endpoint_stored_key_flows_to_resolution():
    app_config = {
        "custom_endpoints": {"paid": {
            "display_name": "Paid", "family": "openai_compatible",
            "base_url": "https://api.example.com/v1",
            "api_key": "stored-secret"}},
    }
    resolved = _resolve_custom_ep_selection(
        provider="custom-ep:paid", model="m",
        base_url="https://api.example.com/v1", app_config=app_config)
    assert resolved.api_key == "stored-secret"

def test_custom_endpoint_unresolved_declared_key_blocks_with_missing_key_copy():
    app_config = {
        "custom_endpoints": {"paid": {
            "display_name": "Paid", "family": "openai_compatible",
            "base_url": "https://api.example.com/v1",
            "api_key_env": "PAID_KEY_UNSET"}},
    }
    resolved = _resolve_custom_ep_selection(
        provider="custom-ep:paid", model="m",
        base_url="https://api.example.com/v1", app_config=app_config)
    assert resolved.sendable is False
    assert "API key" in resolved.visible_copy

def test_custom_endpoint_llama_family_declared_key_flows_to_resolution():
    app_config = {
        "custom_endpoints": {"gpu": {
            "display_name": "GPU llama", "family": "llama_cpp",
            "base_url": "http://192.168.1.5:8080",
            "api_key": "llama-secret"}},
    }
    resolved = _resolve_custom_ep_selection(
        provider="custom-ep:gpu", model="m",
        base_url="http://192.168.1.5:8080", app_config=app_config)
    assert resolved.execution_provider == "llama_cpp"
    assert resolved.api_key == "llama-secret"
```

- [ ] **Step 2: Run to verify failure** — `pytest Tests/Chat/test_console_provider_gateway.py -k custom_endpoint -q` in the snapshot env; expected: the three key-flow tests FAIL with `api_key is None`.

- [ ] **Step 3: Implement** — at the resolution site where `custom_entry` is known: resolve the entry credential per ADR-146 precedence (env value of `entry.api_key_env` first, then `entry.api_key`, blank/placeholder rejected) and, when resolved, set it into the resolution's api_key on both paths (override the keyless-family readiness result); when declared but unresolved, block with the existing missing-API-key copy (reuse Task 3's gate semantics). Never log the value.

- [ ] **Step 4: Run to verify pass** — full gateway file (baseline 238 + 4 new = 242) plus `Tests/Chat/test_custom_endpoint_registry.py` (8) — snapshot env per Task 3/4/5 reports.

- [ ] **Step 5: Commit**

```bash
git commit -m "fix: flow declared custom endpoint credentials into gateway resolution" -- tldw_chatbook/Chat/console_provider_gateway.py Tests/Chat/test_console_provider_gateway.py
```

---

### Task 6: Modal — "New endpoint from template…" creation flow

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_endpoint_template_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/console_settings_modal.py` (Base URL row area, compose ~lines 625-634)
- Create: `Tests/Widgets/test_console_endpoint_template_modal.py`
- Test: `Tests/UI/test_console_session_settings.py` (integration)

**Interfaces:**
- Consumes: Tasks 2-4 (`build_entry_mutation`, `derive_slug`, `validate_entry`, `load_custom_endpoints`); `save_settings_to_cli_config`; `probe_models_endpoint`.
- Produces:

```python
class ConsoleEndpointTemplateModal(ModalScreen[str | None]):
    """Create a custom endpoint entry; dismisses with the new 'custom-ep:<slug>'
    provider id (or None on cancel)."""
    def __init__(self, *, app_config: Mapping[str, object],
                 providers_models: Mapping[str, list[str]],
                 template_provider: str | None = None) -> None: ...

    class EndpointCreated(Message):  # carries provider_id: str
        ...
```

Flow inside the template modal: (1) template picker — an `OptionList` of provider options (from `build_console_provider_options(providers_models, app_config)`) plus a leading "OpenAI-compatible (blank)" entry and existing registry entries as "name (duplicate)"; (2) form — display name `Input`, family `Select` (three fixed options, prefilled from template family), base URL `Input` prefilled from the template's configured endpoint or family default (`DEFAULT_LLAMACPP_BASE_URL` for llama, `http://127.0.0.1:11434` for ollama, blank for openai_compatible), models `Input` (comma-separated, prefilled from template's configured models); (3) inline validation via `validate_entry`; (4) Create — `await asyncio.to_thread(save_settings_to_cli_config, build_entry_mutation(entry))`, then `dismiss(provider_id)`.

In `ConsoleSettingsModal`: a `Button("New endpoint…", id="console-settings-endpoint-new")` added to the Base URL row's `Horizontal` (visible only when `self._provider_uses_base_url(provider)` or the provider Select has any registry entries); on `EndpointCreated`, switch the provider Select to the new id, refresh model controls, and trigger the existing Discover worker against the entry URL.

- [ ] **Step 1: Write the failing component tests**

`Tests/Widgets/test_console_endpoint_template_modal.py`, using the harness style of `Tests/UI/test_console_session_settings.py` (`StyledModalHarness`-equivalent minimal app):

```python
@pytest.mark.asyncio
async def test_template_modal_creates_entry_and_dismisses_with_id(tmp_path):
    # app_config fixtures follow Tests/UI/test_console_session_settings.py's
    # tmp-path config pattern (isolate_config style)
    app = _TemplateModalHarness(app_config={})
    async with app.run_test(size=(100, 40)) as pilot:
        modal = ConsoleEndpointTemplateModal(
            app_config=app.app_config,
            providers_models={"llama_cpp": ["model-a"]},
            template_provider="llama_cpp")
        await app.push_screen(modal)
        await pilot.click("#endpoint-template-name")
        await pilot.press(*"GPU box")
        await pilot.click("#endpoint-template-url")
        await pilot.press("ctrl+a", "http://192.168.1.9:8080")
        await pilot.click("#endpoint-template-create")
    assert app.created_provider_id == "custom-ep:gpu-box"
    entry = load_custom_endpoints(app.app_config)["gpu-box"]
    assert entry.family == "llama_cpp"
    assert entry.base_url == "http://192.168.1.9:8080"

@pytest.mark.asyncio
async def test_template_modal_shows_validation_inline(tmp_path):
    # app_config fixtures follow Tests/UI/test_console_session_settings.py's
    # tmp-path config pattern (isolate_config style)
    app = _TemplateModalHarness(app_config={})
    async with app.run_test(size=(100, 40)) as pilot:
        modal = ConsoleEndpointTemplateModal(
            app_config=app.app_config,
            providers_models={"llama_cpp": ["model-a"]},
            template_provider="llama_cpp")
        await app.push_screen(modal)
        await pilot.click("#endpoint-template-name")
        await pilot.press(*"GPU box")
        await pilot.click("#endpoint-template-url")
        await pilot.press("ctrl+a", "ftp://192.168.1.9:8080")
        error = app.screen.query_one("#endpoint-template-error", Static)
        assert "http(s)" in error.renderable
        create = app.screen.query_one("#endpoint-template-create", Button)
        assert create.disabled is True
    assert app.created_provider_id is None
    assert load_custom_endpoints(app.app_config) == {}
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Widgets/test_console_endpoint_template_modal.py -q`
Expected: FAIL — module absent.

- [ ] **Step 3: Implement the modal + wiring** per Interfaces. Reuse `SafeModalDismissMixin`, `ConsoleSettingsInput`, and the error-banner CSS pattern from `console_settings_modal.py`. Escape cancels (dismiss `None`). The Create button stays disabled until `validate_entry(...) == []` (re-evaluated on every `Input.Changed`).

- [ ] **Step 4: Run component tests, then integration**

Run: `pytest Tests/Widgets/test_console_endpoint_template_modal.py -q`
Expected: PASS.
Run: `pytest Tests/UI/test_console_session_settings.py -k endpoint -q`
Expected: PASS for new integration test (add one: open settings modal, click "New endpoint…", create, assert provider Select value switched and readiness Ready).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_endpoint_template_modal.py tldw_chatbook/Widgets/Console/console_settings_modal.py Tests/Widgets/test_console_endpoint_template_modal.py Tests/UI/test_console_session_settings.py
git commit -m "feat: create custom endpoints from templates in Conversation Settings"
```

---

### Task 7: F9 Settings management — list, rename, edit, delete, convert

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_provider_view_model.py` (registry presentation rows)
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py` (Providers & Models category: "Custom endpoints" panel)
- Test: `Tests/UI/test_settings_custom_endpoints.py` (create)

**Interfaces:**
- Consumes: Tasks 2-6 registry module; `ConsoleChatStore.sessions()` + `.settings.provider` (console_chat_store.py:1760-1778) for the reference guard.
- Produces:

```python
# settings_provider_view_model.py
def custom_endpoint_rows(app_config: Mapping[str, object]) -> tuple[SettingsOverviewRow, ...]:
    """One row per entry: name, family label, safe_endpoint_display(base_url),
    model count. Order matches build_console_provider_options entry order."""

def conversations_referencing_endpoint(store, provider_id: str) -> list[str]:
    """Session ids whose settings.provider == provider_id (titles for copy)."""

def detach_and_delete_entry(app_config: Mapping[str, object], store, provider_id: str) -> None:
    """Detach: for each referencing session, replace settings.provider with the
    entry's family execution key, keeping the session's current base_url as
    session-only. Then delete the entry via
    delete_settings_from_cli_config('custom_endpoints', [slug])."""

def convert_slot_to_named_endpoint(app_config: Mapping[str, object], slot_id: str) -> str:
    """Create a registry entry from the custom/custom_2 slot's configured
    endpoint (family 'openai_compatible', created_from=slot_id) and return the
    new 'custom-ep:<slug>' id. The slot's config is left untouched."""
```

Panel actions per entry: **Rename** (display-name `Input` + save via `build_entry_mutation` with the new name, same slug), **Edit** (base URL / api_key_env / models — same validation as creation), **Delete** (blocked with an actionable message listing `conversations_referencing_endpoint` results and offering "Detach references" — sets each referencing session's `settings` to the family provider id with its current `base_url` kept as session-only — then deletes via `delete_settings_from_cli_config("custom_endpoints", [slug])`), and on the `custom` / `custom_2` built-in rows: **Convert to named endpoint** (creates an entry with `family="openai_compatible"`, the slot's configured endpoint, `created_from` = slot id; slot untouched).

- [ ] **Step 1: Write the failing tests**

`Tests/UI/test_settings_custom_endpoints.py`:

```python
def test_custom_endpoint_rows_render_safe_display():
    rows = custom_endpoint_rows(_registry_config())
    row = next(r for r in rows if "GPU llama" in r.label)
    assert "192.168.1.5:8080" in row.value
    assert "api_key" not in row.value

def test_delete_guard_lists_referencing_sessions():
    store = _store_with_session(provider="custom-ep:gpu")  # minimal store stub
    assert conversations_referencing_endpoint(store, "custom-ep:gpu") != []

def test_delete_after_detach_removes_entry(tmp_path):
    app_config = _registry_config()
    store = _store_with_session(provider="custom-ep:gpu")  # minimal store stub
    detach_and_delete_entry(app_config, store, "custom-ep:gpu")  # panel action seam
    assert load_custom_endpoints(load_settings()) == {}
    assert store.session_settings(store.sessions()[0].id).provider == "llama_cpp"

def test_convert_custom_slot_creates_entry_and_keeps_slot(tmp_path):
    app_config = {"api_settings": {"custom": {
        "api_url": "http://127.0.0.1:5000/v1", "model": "my-model"}}}
    provider_id = convert_slot_to_named_endpoint(app_config, "custom")  # panel seam
    assert provider_id == "custom-ep:custom"
    entry = load_custom_endpoints(load_settings())["custom"]
    assert entry.family == "openai_compatible"
    assert entry.base_url == "http://127.0.0.1:5000/v1"
    assert load_settings()["api_settings"]["custom"]["model"] == "my-model"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/UI/test_settings_custom_endpoints.py -q` — expected FAIL (helpers absent).

- [ ] **Step 3: Implement** the view-model helpers and the panel in the Providers & Models category of `settings_screen.py`, following that screen's existing section/row composition and its config-write patterns (threaded writes, one status line, no partial-apply states).

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/UI/test_settings_custom_endpoints.py Tests/UI/test_settings_endpoint_probe.py -q`
Expected: PASS (endpoint-probe file is the adjacent surface; confirms no regression).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/settings_provider_view_model.py tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/test_settings_custom_endpoints.py
git commit -m "feat: manage custom endpoints in F9 Settings with reference guard"
```

---

### Task 8: Docs and task closeout

**Files:**
- Modify: `Docs/User_Guide/settings.md` (Custom endpoints section under Providers & Models)
- Modify: `Docs/User_Guide/console.md` (Conversation Settings: endpoint creation paragraph)
- Modify: `backlog/tasks/task-32474 - Console-custom-endpoint-registry-named-endpoints-from-templates.md`

- [ ] **Step 1: Write user-facing docs** — two short sections: what a custom endpoint is, creating one from a template in Conversation Settings, managing in F9, the convert action, and the delete/detach rule. Match the existing tables/voice in those files.
- [ ] **Step 2: Close TASK-32474** — tick all ACs, add Implementation Notes (approach, files, decisions, verification runs, ADR-146 link), `backlog task edit 32308 -s Done`.
- [ ] **Step 3: Final targeted sweep**

Run: `pytest Tests/Chat/test_custom_endpoint_registry.py Tests/Chat/test_console_session_settings.py Tests/Chat/test_console_provider_gateway.py Tests/Widgets/test_console_endpoint_template_modal.py Tests/UI/test_settings_custom_endpoints.py Tests/UI/test_console_session_settings.py -q`
Expected: PASS except the 2 documented pre-existing failures in `test_console_session_settings.py` (inspector staged-context ordering, unmount-timeout repair — unrelated to this work, fail on the pristine tree).

- [ ] **Step 4: Commit**

```bash
git add Docs/User_Guide/settings.md Docs/User_Guide/console.md "backlog/tasks/task-32474 - Console-custom-endpoint-registry-named-endpoints-from-templates.md"
git commit -m "docs: custom endpoint registry user guide and task closeout"
```
