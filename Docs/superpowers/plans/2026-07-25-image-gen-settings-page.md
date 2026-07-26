# Settings ▸ Image Gen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Settings rail category that safely edits the `[image_generation]` config (backends, keys, generation defaults) with live status, on-demand probes, and draft/save semantics.

**Architecture:** New `SettingsCategoryId.IMAGE_GENERATION` composed in `settings_screen.py` like every sibling; ALL logic in a new focused module `settings_image_gen_defaults.py` plus a tiny loader extension (`key_sources`) and adapter extension (`delete_values`). Spec: `Docs/superpowers/specs/2026-07-25-image-gen-settings-page-design.md` — read it before any task.

**Tech Stack:** Python 3.11+/Textual 8.2.7, toml config helpers, httpx (probe), pytest.

## Global Constraints

- Writes come ONLY from user-edited draft fields — never from the effective `ImageGenerationConfig` (its secret fields contain env/keyring-resolved values; persisting them would copy secrets into plaintext config.toml).
- Clearing a field DELETES its key (`delete_settings_from_cli_config`); never write `""`/`None` as a cleared sentinel.
- All writes go to nested sections (`image_generation.<backend>`) or flat `[image_generation]` for global keys, in ONE `save_sections` call; after save call `reset_image_generation_config_cache()`.
- Probe summaries/badges never contain URLs, headers, credentials, or raw exception text.
- Draft/dirty idiom: rail dirty marker, explicit Save, Revert discards. No per-field immediate writes.
- Rail label exactly `Image Gen`. Category id string exactly `image_generation`.
- Textual 8.2.7: `Select.NULL` (no `Select.BLANK`). `run_worker(..., exit_on_error=False)` + try/except in worker bodies.
- CSS: source TCSS modules only → `./build_css.sh` + `python -m tldw_chatbook.css.check_bundle_sync`; color styling at app-tier CSS, never widget DEFAULT_CSS.
- Tests FOREGROUND only. venv: `source /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/activate`; run from the worktree.
- Baseline flake (NOT yours): `test_console_conversation_browser_search_ignores_stale_results` (ordering/contention; passes isolated).

---

### Task 1: Loader `key_sources`

**Files:**
- Modify: `tldw_chatbook/Image_Generation/config.py` (secret resolution ~lines 125–145: the loop over `_SECRETS` inside `_load_image_generation_section`, and the `ImageGenerationConfig` dataclass)
- Test: `Tests/Image_Generation/test_config_loader.py`

**Interfaces:**
- Produces: `ImageGenerationConfig.key_sources: dict[str, str]` — backend id → `"env:<VAR>"` | `"config"` | `"keyring"` | `"missing"`. Later tasks read it via `get_image_generation_config().key_sources["openrouter"]`.

- [ ] **Step 1: Write failing tests** (append to `test_config_loader.py`, mirroring its existing fixture style for building a section dict / env monkeypatching):

```python
def test_key_sources_env_wins(monkeypatch, tmp_path):
    """key_sources records env origin with the winning variable name."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-env-key")
    cfg = _load_config_with_section({"openrouter": {}})  # use the file's existing section-builder helper
    assert cfg.key_sources["openrouter"] == "env:OPENROUTER_API_KEY"

def test_key_sources_config(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = _load_config_with_section({"openrouter": {"api_key": "fake-config-key"}})
    assert cfg.key_sources["openrouter"] == "config"

def test_key_sources_missing(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = _load_config_with_section({})
    assert cfg.key_sources["openrouter"] == "missing"
    assert set(cfg.key_sources) == {"stable_diffusion_cpp", "swarmui", "openrouter", "novita", "together", "modelstudio"}

def test_key_sources_modelstudio_names_winning_env(monkeypatch):
    monkeypatch.setenv("QWEN_API_KEY", "fake-2")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    cfg = _load_config_with_section({})
    assert cfg.key_sources["modelstudio"] == "env:QWEN_API_KEY"
```

Adapt `_load_config_with_section` to whatever helper the file already uses to invoke the loader with a crafted `[image_generation]` table (there is one — read the file first; if keyring is monkeypatched in existing tests, follow that pattern and add a `keyring` source test too). `stable_diffusion_cpp` has no `_SECRETS` entry — decide and pin: it appears in `key_sources` as `"missing"` (uniform dict, simplest consumers).

- [ ] **Step 2: Run** `python -m pytest Tests/Image_Generation/test_config_loader.py -q` → new tests FAIL (`key_sources` attribute missing).
- [ ] **Step 3: Implement.** In the `_SECRETS` resolution loop record the winner: env hit → `f"env:{ev}"`; config value → `"config"`; keyring hit → `"keyring"`; nothing → `"missing"`. Backends without a `_SECRETS` entry get `"missing"`. Add `key_sources: dict[str, str]` to `ImageGenerationConfig` (default `{}` via `field(default_factory=dict)` if the dataclass style needs it) and thread the dict through the builder. Do NOT change what gets written into the secret fields.
- [ ] **Step 4: Run** the full file: all pass, EXISTING tests unmodified. Also `python -m pytest Tests/Image_Generation/ -q` green.
- [ ] **Step 5: Commit** `feat(imagegen): record per-backend key_source in config loader`.

---

### Task 2: Defaults module data layer + adapter `delete_values`

**Files:**
- Create: `tldw_chatbook/UI/Screens/settings_image_gen_defaults.py`
- Modify: `tldw_chatbook/UI/Screens/settings_config_adapter.py`
- Test: `Tests/UI/test_settings_image_gen_defaults.py` (new), `Tests/UI/test_settings_config_adapter.py` (extend if it exists, else cover delete_values in the new file)

**Interfaces:**
- Consumes: `ImageGenerationConfig.key_sources` (Task 1); `_SECRETS`/`_NON_SECRET` tables from `Image_Generation.config` (import them for the drift test); `delete_settings_from_cli_config` (exists, config.py:3868).
- Produces (all in `settings_image_gen_defaults.py`):
  - `BACKEND_IDS: tuple[str, ...] = ("stable_diffusion_cpp", "swarmui", "openrouter", "novita", "together", "modelstudio")`
  - `BACKEND_LABELS: dict[str, str]` (`"stable_diffusion_cpp": "SD.cpp (local)"`, `"swarmui": "SwarmUI (local)"`, `"openrouter": "OpenRouter"`, `"novita": "Novita"`, `"together": "Together"`, `"modelstudio": "ModelStudio"`)
  - `@dataclass(frozen=True) FieldSpec(toml_key: str, label: str, kind: str, min_value: float | None = None)` — kind ∈ {"text", "url", "path", "int", "secret"}
  - `FIELD_SCHEMA: dict[str, tuple[FieldSpec, ...]]` — exactly the spec's v1 table (sd.cpp: binary_path/model_path[path] + timeout_seconds[int,min 1]; swarmui: base_url[url], default_model[text], timeout_seconds[int,min 1], swarm_token[secret]; openrouter/novita/together: base_url[url], default_model[text], timeout_seconds[int,min 1], api_key[secret]; modelstudio adds region[text])
  - `@dataclass(frozen=True) ImageGenBackendRow(backend_id, label, configured: bool, enabled: bool, is_default: bool, key_source: str, secret_optional: bool)` — `secret_optional=True` only for swarmui
  - `build_backend_rows(cfg, raw_section: Mapping) -> list[ImageGenBackendRow]`
  - `effective_placeholder(cfg, backend_id: str, toml_key: str) -> str` — the resolved effective value for an unset field (reads the cfg flat field via `_NON_SECRET[(backend, key)]`, formats non-None as str, else "")
  - `@dataclass(frozen=True) ImageGenDraftValues(...)` — global keys (`default_backend`, `enabled_backends: list[str]`, `default_batch`, `max_variants_per_message`, `context_llm_enabled`, `context_llm_turns`, `context_llm_timeout_seconds`) + `backend_fields: dict[str, dict[str, str]]` (backend → toml_key → edited raw string) + `cleared_fields: dict[str, list[str]]`
  - `diff_to_sections(draft: ImageGenDraftValues, raw_config: Mapping) -> tuple[dict[str, dict], dict[str, list[str]]]` — (sections-to-save, sections-to-delete-keys-from); only keys that differ from the RAW config (not effective) are emitted; int fields coerced; secrets included only when the user typed one this session
  - `validate_draft(draft) -> list[str]` — returns human messages: default-not-enabled block; int/min violations; url shape violations (`urlparse` scheme in {http,https} and netloc); plus non-blocking hints as a separate `warnings` list return — signature: `-> tuple[list[str], list[str]]` (errors, warnings; warnings: all-disabled, batch>cap)
- Adapter: `SettingsConfigAdapter.delete_values(section: str, keys: list[str]) -> bool` delegating to `delete_settings_from_cli_config`.

- [ ] **Step 1: Write failing tests.** Core cases (write them all before implementing):

```python
from tldw_chatbook.Image_Generation.config import _NON_SECRET, _SECRETS
from tldw_chatbook.UI.Screens.settings_image_gen_defaults import (
    BACKEND_IDS, FIELD_SCHEMA, ImageGenDraftValues, build_backend_rows,
    diff_to_sections, effective_placeholder, validate_draft,
)

def test_field_schema_maps_to_real_loader_keys():
    """Drift guard: every schema field must be a key the loader actually reads."""
    for backend, specs in FIELD_SCHEMA.items():
        for spec in specs:
            if spec.kind == "secret":
                assert _SECRETS[backend], backend
            else:
                assert (backend, spec.toml_key) in _NON_SECRET

def test_diff_emits_only_changed_keys_to_nested_sections():
    draft = _draft(backend_fields={"openrouter": {"default_model": "openai/gpt-5-image-mini"}})
    sections, deletions = diff_to_sections(draft, raw_config={"image_generation": {}})
    assert sections == {"image_generation.openrouter": {"default_model": "openai/gpt-5-image-mini"}}
    assert deletions == {}

def test_diff_never_copies_env_resolved_secret(monkeypatch):
    """THE no-secret-copy pin: effective cfg holds env-resolved keys; the diff
    must not see them because it only reads the draft + raw config."""
    draft = _draft()  # user typed nothing
    sections, _ = diff_to_sections(draft, raw_config={"image_generation": {}})
    flat = {k: v for sec in sections.values() for k, v in sec.items()}
    assert "api_key" not in flat and "swarm_token" not in flat

def test_cleared_field_becomes_deletion_not_empty_write():
    draft = _draft(cleared_fields={"openrouter": ["default_model"]})
    sections, deletions = diff_to_sections(
        draft, raw_config={"image_generation": {"openrouter": {"default_model": "x"}}})
    assert "default_model" not in sections.get("image_generation.openrouter", {})
    assert deletions == {"image_generation.openrouter": ["default_model"]}

def test_validate_blocks_disabled_default():
    errors, _ = validate_draft(_draft(default_backend="openrouter", enabled_backends=["swarmui"]))
    assert any("Default backend must be enabled" in e for e in errors)

def test_validate_warns_all_disabled_and_batch_over_cap():
    _, warnings = validate_draft(_draft(enabled_backends=[], default_batch=9, max_variants_per_message=4))
    assert len(warnings) == 2

def test_build_backend_rows_status_and_sources():
    cfg = _fake_cfg(key_sources={"openrouter": "env:OPENROUTER_API_KEY", "swarmui": "missing", ...})
    rows = {r.backend_id: r for r in build_backend_rows(cfg, raw_section={})}
    assert rows["openrouter"].key_source == "env:OPENROUTER_API_KEY"
    assert rows["swarmui"].secret_optional is True

def test_effective_placeholder_shows_baked_default():
    cfg = _fake_cfg()  # nothing set
    assert effective_placeholder(cfg, "openrouter", "default_model") == "google/gemini-2.5-flash-image"

def test_adapter_delete_values(tmp_path, monkeypatch):
    # point config at a scratch file containing [image_generation.openrouter] default_model,
    # call SettingsConfigAdapter().delete_values("image_generation.openrouter", ["default_model"]),
    # re-read file: key gone, section otherwise intact.
```

Build `_draft`/`_fake_cfg` helpers with sensible defaults (a real `ImageGenerationConfig` via the loader with a crafted section is sturdier than a Mock for `_fake_cfg` — reuse Task 1's helper pattern). `configured` on rows: reuse the existing listing logic (`Image_Generation.listing` — read it; do not reimplement `is_configured`).

- [ ] **Step 2: Run** → FAIL (module missing).
- [ ] **Step 3: Implement** the module + adapter method per the Interfaces block. `diff_to_sections` signature discipline: it receives the RAW config mapping (from adapter.load()), never an `ImageGenerationConfig`.
- [ ] **Step 4: Run** new file + `Tests/Image_Generation/ -q` → green. `ruff check` both touched files.
- [ ] **Step 5: Commit** `feat(settings): image-gen defaults data layer + adapter delete_values`.

---

### Task 3: Probe module

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_image_gen_defaults.py` (add probe section)
- Test: `Tests/UI/test_settings_image_gen_defaults.py` (extend)

**Interfaces:**
- Consumes: `FIELD_SCHEMA`/`BACKEND_IDS` (Task 2); `Utils.egress` (`check_url_or_raise`, `origin_set`, `EgressBlockedError`); httpx (sync client — this runs in a thread worker).
- Produces:
  - `@dataclass(frozen=True) ImageGenProbeResult(ok: bool, badge: str)` — badge is one of the spec's exact strings: `"Reachable"`, `"Reachable (auth unverified)"`, `"Auth failed"`, `"Unreachable: <category>"`, `"Binary found"`, `"Binary missing or not executable"`, `"Model file missing"`. `<category>` ∈ {"connection refused", "timeout", "HTTP <status>", "blocked by egress policy"} — never exception text.
  - `probe_backend(backend_id: str, form_values: Mapping[str, str], secret: str | None) -> ImageGenProbeResult` — BLOCKING; caller runs it in a worker. `form_values` are the CURRENT editor fields (falling back to effective values for unset ones — caller resolves that); `secret` is the pasted-or-effective key.
  - `PROBE_TIMEOUT_SECONDS = 5.0`

Behavior per backend (spec table): sd.cpp → `shutil.which(binary) or (Path.is_file + os.access X_OK)`, then model file `is_file()`; no network. swarmui → GET `base_url` with `trusted_origins=origin_set(base_url)` egress pre-check, any HTTP answer (even 4xx) = `Reachable` (server answered). openrouter/together → GET `{base_url}/models`, `Authorization: Bearer <secret>` when secret present → 2xx `Reachable` / 401,403 `Auth failed` / other-status `Unreachable: HTTP <n>`; no secret → reachability GET without auth → answered = `Reachable (auth unverified)`. novita → implementer checks Novita's docs for a cheap authed GET; if none confirmed, unauthenticated reachability → `Reachable (auth unverified)`. modelstudio → reachability only → `Reachable (auth unverified)`. Every network probe: egress `check_url_or_raise(url, trusted_origins=origin_set(url))` FIRST (EgressBlockedError → `Unreachable: blocked by egress policy`), `httpx.Client(timeout=PROBE_TIMEOUT_SECONDS, follow_redirects=False)`.

- [ ] **Step 1: Failing tests** — fake httpx.Client via monkeypatch on the module (follow `Tests/Image_Generation/test_http_client.py`'s fake-client style): 2xx→Reachable; ConnectError→"Unreachable: connection refused"; ReadTimeout→"Unreachable: timeout"; 401 w/ key→"Auth failed"; no-key openrouter→"Reachable (auth unverified)"; sd.cpp with tmp_path fake executable/model → "Binary found" vs chmod-0 binary → "Binary missing or not executable". **Sanitization pin:** fake raising `httpx.ConnectError("secret sk-x in text http://10.0.0.1")` → badge is exactly `"Unreachable: connection refused"` (no substring of the exception message). **Egress pin:** private base_url (e.g. `http://127.0.0.1:7801`) with trusted origins → probe attempts the request (fake sees it); an API-shaped public URL still passes `check_url_or_raise` normally.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** file + ruff → green.
- [ ] **Step 5: Commit** `feat(settings): image-gen backend probes with sanitized badges`.

---

### Task 4: Category registration + read-only panel

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_config_models.py` (enum), `tldw_chatbook/UI/Screens/settings_screen.py` (rail/summaries/compose — grep-anchored, keep hunks minimal), TCSS source module for settings (find the settings TCSS file under `tldw_chatbook/css/components/` via `grep -rln "settings-dirty-category" tldw_chatbook/css/`), regenerate bundle
- Test: `Tests/UI/test_settings_image_gen_panel.py` (new)

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces: `SettingsCategoryId.IMAGE_GENERATION = "image_generation"`; the composed panel with widget ids later tasks rely on: rows `#settings-imagegen-backend-<id>`, badges `#settings-imagegen-status-<id>`, test buttons `#settings-imagegen-test-<id>`, editor container `#settings-imagegen-editor`, field inputs `#settings-imagegen-field-<backend>-<toml_key>`, globals `#settings-imagegen-<global_key>`, save/revert `#settings-imagegen-save` / `#settings-imagegen-revert`.

Process: FIRST read how an existing editing category registers end-to-end — grep `settings_screen.py` for `LIBRARY_RAG` and follow every hit (enum → summaries → rail label → compose → handlers). Mirror that skeleton for IMAGE_GENERATION with the panel content: backends block (six rows: label, badge from `build_backend_rows`, Enabled checkbox, Default radio-style selector, Test button), editor rendering `FIELD_SCHEMA` fields for the selected backend with `effective_placeholder` placeholders and the source line for secrets, generation-defaults block (model fields whose backend has an env override note it inline, e.g. openrouter: "env OPENROUTER_IMAGE_MODEL overrides this"), template count line ("13 built-in + N user" via `get_all_templates()` from `Media_Creation.generation_templates`), demo hint line, advanced-keys hint per backend. READ-ONLY in this task: inputs render values but Save/Revert/Test are wired in Tasks 5–6 (buttons present, disabled). Category summary keywords: image, generation, backend, swarmui, openrouter, model.

- [ ] **Step 1: Failing screen tests** (mirror an existing settings screen test's app-harness — find one via `grep -rln "SettingsScreen" Tests/UI/ | head -3` and copy its scratch-config + run_test scaffolding):

```python
async def test_image_gen_category_in_rail_and_search(...):
    # rail contains "Image Gen"; category search for "swarmui" surfaces it
async def test_panel_populates_from_scratch_config(...):
    # scratch TLDW_CONFIG_PATH with [image_generation] default_backend="openrouter",
    # enabled_backends=["openrouter"], [image_generation.openrouter] default_model="m-x"
    # → openrouter row enabled+default; model input value "m-x"; unset swarmui model
    #   input empty with placeholder == effective default
async def test_secret_input_never_echoes_saved_key(...):
    # config with api_key → secret input value is EMPTY, source line "local config key saved"
async def test_env_key_shows_env_source_line(...):
    # monkeypatch OPENROUTER_API_KEY → source line "env: OPENROUTER_API_KEY"; input empty
```

- [ ] **Step 2: Run** → FAIL (category absent). **Step 3: Implement** registration + compose. **Step 4:** tests green; `./build_css.sh && python -m tldw_chatbook.css.check_bundle_sync`; `python -c "import tldw_chatbook.app"`; spot-run 2 sibling settings test files for collateral.
- [ ] **Step 5: Commit** `feat(settings): Image Gen category — registration + read-only panel`.

---

### Task 5: Draft/dirty editing + Save/Revert

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py` (wire handlers), `settings_image_gen_defaults.py` (only if a pure helper is missing)
- Test: `Tests/UI/test_settings_image_gen_panel.py` (extend)

**Interfaces:**
- Consumes: `ImageGenDraftValues`, `diff_to_sections`, `validate_draft`, adapter `save_sections`/`delete_values`, `reset_image_generation_config_cache` (from `Image_Generation.config`).
- Produces: working edit→dirty→Save/Revert cycle for the whole panel.

Read the screen's existing draft lifecycle first (grep `is_dirty` sites ~2445/2654/3020–3177 and the draft object they consult) and join that idiom exactly — the rail dirty marker and `settings-dirty-category` class must light up for this category via the same path the siblings use, not a parallel mechanism. Save flow: build `ImageGenDraftValues` from widgets → `validate_draft` → errors: inline message widget + notify, NO write; warnings: inline only, proceed → `diff_to_sections(draft, adapter.load()["image_generation"] or {})` → ONE `adapter.save_sections(...)` + `adapter.delete_values(...)` per deletion section → `reset_image_generation_config_cache()` → re-read + re-render → clear dirty. Revert: discard draft, re-render from disk. Secret inputs feed the draft only when non-empty this session; Clear button (per secret) records a deletion and clears the source line optimistically to the post-delete winner.

- [ ] **Step 1: Failing tests:**

```python
async def test_edit_marks_dirty_and_save_writes_nested_toml(...):
    # type model "openai/gpt-5-image-mini" in openrouter editor → dirty marker on rail;
    # press Save → scratch config file parses with
    # ["image_generation"]["openrouter"]["default_model"] == "openai/gpt-5-image-mini";
    # get_image_generation_config().openrouter_image_default_model reflects it (cache reset)
async def test_save_blocked_when_default_disabled(...):
    # uncheck the default backend's Enabled, Save → file UNCHANGED, inline error visible
async def test_revert_discards_draft(...):
async def test_clear_key_deletes_not_blanks(...):
    # config-saved api_key; press Clear + Save → key absent from file (not "")
async def test_pasted_key_saves_and_input_resets(...):
    # paste into secret input, Save → file has key; input value empty again; source line "local config key saved"
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4:** panel test file green + `python -m pytest Tests/Image_Generation/ -q` green (loader untouched but config files get written in tests — ensure scratch isolation). **Step 5: Commit** `feat(settings): Image Gen draft editing, save, revert`.

---

### Task 6: Probe wiring + polish

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`, `settings_image_gen_defaults.py` (only helpers)
- Test: `Tests/UI/test_settings_image_gen_panel.py` (extend)

**Interfaces:**
- Consumes: `probe_backend`, `ImageGenProbeResult` (Task 3).

Wire each Test button: gather CURRENT form values for that backend (unset fields fall back to effective values; secret = pasted value if present this session else the effective resolved secret), dispatch `run_worker(thread=True, exit_on_error=False)` wrapping `probe_backend` in try/except (any escape → `Unreachable: probe error` badge, debug log), render the badge into `#settings-imagegen-status-<id>`, disable ALL test buttons while one runs, re-enable in the worker's finally via `call_from_thread`. Probe state resets when the category re-opens.

- [ ] **Step 1: Failing tests:**

```python
async def test_probe_uses_current_form_values(...):
    # monkeypatch probe_backend capturing args; edit base_url WITHOUT saving; press Test
    # → captured form_values["base_url"] == edited value
async def test_probe_renders_badge_and_reenables_buttons(...):
    # fake returns ImageGenProbeResult(True, "Reachable") → badge text "Reachable";
    # all test buttons disabled during run (assert via a gating fake), enabled after
async def test_probe_secret_uses_pasted_unsaved_key(...):
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4:** full new test file + `Tests/UI/test_settings_image_gen_defaults.py` green; `ruff check` touched files; `python -c "import tldw_chatbook.app"`; bundle sync if TCSS touched.
- [ ] **Step 5: Commit** `feat(settings): Image Gen probe wiring + polish`.

---

## Post-plan verification (controller)

Full sweep: both new test files + `Tests/Image_Generation/` + 2 sibling settings suites + `Tests/UI/test_console_native_chat_flow.py` (collateral); live tmux smoke of the category (open, edit, save, probe against the live llama-adjacent SwarmUI-absent environment → expect honest Unreachable); backlog task filing/Done per Backlog.md conventions happens at the campaign level, not in these tasks.
