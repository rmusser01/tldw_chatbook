# TASK-432 — Provider Test evaluates & honestly reports the draft config

- **Date:** 2026-07-22
- **Task:** TASK-432 (RP/character-card UX review). Settings ▸ Providers & Models "Test Provider".
- **Branch base:** origin/dev (tip `89ec674a0`).
- **File:** `tldw_chatbook/UI/Screens/settings_screen.py` (readiness-test report + a new staged-config helper). No changes to `tldw_chatbook/Chat/provider_readiness.py` (its `get_provider_readiness(provider, app_config, *, environ)` already accepts an injectable config).

## Problem

Observed live in Settings ▸ Providers & Models: with an **unsaved draft** endpoint (`:9099`), the Test correctly *exercised* the draft (the live probe hit `:9099`), but the evidence line printed the **saved** value `api_settings.llama_cpp.api_url=http://localhost:8080/completion` — so the proof contradicted what was tested.

Root cause: `_provider_readiness_test_report` reads `provider`/`model` from the draft widgets, and the live probe (`_provider_live_probe_base_url`) uses the draft endpoint widget, but:
- `get_provider_readiness(provider, app_config)` is called on the **saved** `app_config` — so `ready` / `api_key_source` / `env_var` reflect saved config, not the draft.
- The endpoint finding calls `_provider_endpoint_summary(provider)` with **no** `endpoint` argument, so it defaults to `_provider_endpoint_value(provider)` = the saved config value.

## Current state on dev (the review was at older dev `dc196563f`)

- **AC#2 (mouse-clickable):** already satisfied. A later change (task-189) added a real `Button("Test Provider", id="settings-test-provider")` wired via `@on(Button.Pressed, "#settings-test-provider")` → `handle_test_provider` → `action_settings_test_category(allow_text_entry_focus=True)`. Clicking runs the test even when an input has focus.
- **AC#3 (non-hotkey path):** already satisfied by that button. The `t` hotkey's focus-guard (`action_settings_test_category` returns early when a text input is focused unless `allow_text_entry_focus=True`) is intentional — a category hotkey must not fire while the user types `t` in a field — and the binding is listed in the footer.
- **AC#1 (honest evidence):** genuinely broken (root cause above). Per the chosen scope (**full draft-honesty pass**), the fix makes *all* tested values draft-sourced and honestly labeled, not just the endpoint.

## Design

### 1. Draft-overlaid config for the readiness evaluation

The settings form exposes the whole draft via `_provider_form_values_from_widgets()` (`provider`, `model`, `endpoint`, `api_key`, `credential_env_var`, …). The app already tracks per-field dirtiness in `SettingsDraft` for the `PROVIDERS_MODELS` category: `self._provider_draft()` → `SettingsDraft` whose `dirty_keys()` returns the keys whose staged value differs from the loaded original. This is the same signal the "unsaved changes" indicator and Save button use, so it is the authoritative provenance source.

New helper:

```python
def _provider_test_staged_config(self, provider: str) -> Mapping[str, object]:
    """Return app_config with the unsaved draft provider settings overlaid.

    Mirrors how ``_provider_discovery_staged_settings`` stages draft values,
    but produces a full merged config so ``get_provider_readiness`` evaluates
    the DRAFT the user is about to test. Only dirty fields are overlaid, so a
    provider with no unsaved edits tests exactly the saved config.
    """
```

Behavior:
- Resolve `provider_key`/`provider_save_key` (via `_provider_config_entry`).
- Read the draft `endpoint`, `api_key`, `credential_env_var` from the widgets (fallback to `_provider_setting_values_mapping()` on `QueryError`).
- `dirty = self._provider_draft().dirty_keys()` (empty set when there is no provider draft). Each of `endpoint`, `credential_env_var`, `api_key` is staged into the draft **only** from its `@on(Input.Changed, …)` handler, so a field appears in `dirty` exactly when the user edited it away from the loaded value (programmatic pre-fills stage `value == original` and drop out; the api-key field never pre-fills the saved secret). `dirty_keys()` is therefore the authoritative per-field "unsaved" signal.
- Deep-copy `app_config`; on the copy's `api_settings[provider_save_key]` (create the dict if absent), override only the dirty fields:
  - endpoint key (`_provider_endpoint_setting_key(provider)`) ← draft endpoint when `"endpoint" in dirty`.
  - `api_key_env_var` ← draft `credential_env_var` when `"credential_env_var" in dirty`.
  - `api_key` ← draft `api_key` when `"api_key" in dirty` (this includes the explicit **clear** action, which stages `api_key=""`; overlaying `""` makes readiness honestly reflect the removed key, since `_valid_api_key("")` is falsy).
- Preserve every other provider's settings and the overlaid provider's other keys (deep copy, targeted override).
- If none of the three fields is dirty, return `app_config` unchanged (the Test then evaluates exactly the saved config).

`get_provider_readiness` checks the `api_key` slot before the env var and defaults, so overlaying a non-empty draft key makes `readiness.ready`/`api_key_source` reflect the draft key.

**Testability structure (improvement):** the existing settings tests are heavyweight full-pilot (they settle a mount storm and scroll to controls), so the provenance permutations must not depend on a full mount. Factor the value logic into pure, resolved-input helpers:
- a module-level pure `overlay_provider_draft_config(app_config, *, provider_save_key, endpoint_key, draft_endpoint, draft_env_var, draft_api_key, dirty) -> dict` that does the deep-copy + targeted override (no `self`, no widgets);
- the findings/provenance assembly takes **resolved** inputs (`provider`, `provider_key`, `model`, `readiness`, `draft_endpoint`, `dirty`, the env value) rather than reading widgets, so it can be exercised on a bare screen instance (`SettingsScreen.__new__(SettingsScreen)` with `app_instance.app_config` set) — the same bare-instance pattern used elsewhere in the suite.

The thin `_provider_test_staged_config` / `_provider_readiness_test_report` wrappers read the widgets (and `_provider_draft().dirty_keys()`) and delegate to these pure pieces.

### 2. Provenance-tagged evidence in `_provider_readiness_test_report`

Compute a provenance set once: `dirty = self._provider_draft().dirty_keys()` (empty set when no draft). Then:

- Call `readiness = get_provider_readiness(provider, self._provider_test_staged_config(provider))` (unchanged aside from the staged config; production keeps the default `os.environ`, tests inject `environ=` as today).
- **model** already comes from the draft widget; tag `model=<v> (draft)` when `"model" in dirty`.
- **api key source line** — after the existing source is computed:
  - if `"api_key" in dirty` and the draft key is non-empty and `readiness.api_key_source == f"config:api_settings.{provider_key}.api_key"` → the resolved key is the unsaved draft key: replace the finding with `api_key_source=draft api_key (unsaved)` (never print the key; `redact_secret_text` still wraps the line).
  - else if `"credential_env_var" in dirty` and `readiness.env_var` is set → append ` (draft env var)` to the `env_var=…` finding.
  - (an explicit key-clear — `"api_key" in dirty` with an empty draft value — needs no special tag: readiness resolves honestly to env/default/none from the overlaid empty key.)
- **endpoint line** — resolve the draft endpoint (widget value, else saved), pass it to `_provider_endpoint_summary(provider, endpoint=draft_endpoint)`, and append ` (draft)` when `"endpoint" in dirty`.
- `status=…`, `passed`, and the toast `summary` follow from the draft-based `readiness` unchanged in shape.

The whole findings/summary strings stay wrapped in `redact_secret_text(...)` as today.

### 3. AC#2 / AC#3 — pin the already-delivered behavior

No production change needed; add regression tests so the behavior can't silently regress:
- AC#2: pressing the `#settings-test-provider` button runs the provider test (updates `#settings-provider-test-result`) — including when a provider input has focus (the button passes `allow_text_entry_focus=True`).
- AC#3: the button is a non-hotkey path — the same test runs on click without the `t` key and without moving focus out of an input.

## Testing

Unit tests on the **pure** helpers (`overlay_provider_draft_config` + the resolved-input findings builder on a bare `SettingsScreen` instance; construct `app_config` directly, inject `environ` / patch `os.environ`):
- **Endpoint honesty:** draft endpoint (`:9099`) ≠ saved (`:8080`) → the endpoint finding shows the draft value + `(draft)`, never the saved value.
- **Draft key honesty:** dirty non-empty draft `api_key`, no saved key → `readiness.ready` true via the draft key; finding reads `api_key_source=draft api_key (unsaved)`; the key value never appears in `detail`/`summary`.
- **Draft env-var:** dirty `credential_env_var` with the env value present → `env_var` finding tagged `(draft env var)`.
- **No draft (saved-only):** no dirty keys → `_provider_test_staged_config` returns the config unchanged and the evidence shows the saved values with **no** `(draft)` tags (today's behavior preserved).
- **staged-config isolation:** overlay does not mutate the real `app_config` (deep copy) and preserves other providers' settings.

Pilot tests (mirroring the existing settings-screen harness, kept minimal):
- AC#2: `#settings-test-provider` `Button.press()` populates `#settings-provider-test-result`.
- AC#3: the button runs the test with a provider `Input` focused (no `t` key pressed) — proving the non-hotkey path.
- One end-to-end wiring test: set the endpoint `Input` to a draft value, press Test, assert the result row shows the draft value + `(draft)` (guards the widget→pure-helper wiring).

Cross-check: the existing provider-test / readiness tests (`Tests/UI/test_settings_configuration_hub.py`) stay green.

## Risks / mitigations

- **Leaking a draft secret:** the api-key value is never placed in a finding — only the source label — and `redact_secret_text` wraps the whole line. Tests assert the key value is absent.
- **Mutating shared `app_config`:** the staged config is a deep copy; tested explicitly.
- **Provider/model draft edge:** provider/model already come from the draft widgets; the overlay keys off the draft provider's config section (`_provider_config_entry`), so switching provider without saving still tests the right section.
- **Blast radius:** `get_provider_readiness` is unchanged (only fed a different config), so Chat-side readiness is unaffected; the change is contained to the settings screen's Test path. It performs no logging of the config or key (verified).
- **`None` vs `""` dirty edge:** `dirty_keys()` compares with `!=`, so clearing a field that had no saved value (`"" != None`) reads as dirty. The only visible effect is a `(draft)` tag on an empty endpoint/env-var — harmless and arguably accurate (the user did edit the field); no special handling.
- **Provider/model provenance:** `model` is a discrete finding and is tagged when dirty. The provider name is not a discrete finding (it lives in `readiness.user_message` / the toast display name), so there is no separate provider tag — testing a draft provider still reads that provider's (draft-overlaid) section correctly.

## Non-goals

- Changing the `t` hotkey's focus-guard (intentional; the button is the discoverable path).
- Making the readiness function itself draft-aware (kept pure; the staged config carries the draft).
- A live provider chat/completion call (the Test stays local readiness + the existing short endpoint probe for URL-based providers).
- Persisting or auto-saving the draft as a side effect of testing.
