# Settings ▸ Image Gen — design

Date: 2026-07-25. Status: approved pending user review.
Origin: image-gen UAT (2026-07-25) showed both config-surface findings (stale default model task-620, silently-ignored flat keys task-621) exist because image generation is configured only by hand-editing nested TOML with zero UI feedback. This adds a Settings category that edits that config safely.

## Scope

**v1 (this spec):**
1. Backends block — enable/disable, default-backend selection, per-backend status, on-demand reachability probe.
2. Per-backend editor — curated fields (schema-driven, see table) including API keys/tokens.
3. Generation defaults — batch/variant caps and the context-LLM trio.
4. Read-only style-template count line (management deferred to v2).

**Explicit non-goals (v1):** style-template create/edit/delete; advanced per-backend keys (`allowed_extra_params`, sd.cpp `vae/llm/lora/steps/cfg/sampler/diffusion_model_path`, novita/modelstudio `poll_interval_seconds`, modelstudio `mode`) — these stay config-file-only, and the editor shows one "Advanced keys live in config.toml → [image_generation.<backend>]" hint line per backend that has any; embedding test image generation (the existing command-palette `ImageGenDemoScreen` is linked by a hint line instead); auto-probing on open.

## Architecture

Approach A (approved): a new rail category composed in `settings_screen.py` like every sibling, with **all logic in a new focused module** `tldw_chatbook/UI/Screens/settings_image_gen_defaults.py` (the `settings_library_rag_defaults.py` pattern). The monolith gains only: the `SettingsCategoryId.IMAGE_GENERATION` enum member (in `settings_config_models.py`), rail/category registration, the compose block, and thin event handlers delegating to the module.

The defaults module owns, as pure/testable units:
- `ImageGenBackendRow` (dataclass): backend id, display name, configured flag, enabled flag, is_default flag, key_source, probe state.
- `build_backend_rows(cfg, raw_section) -> list[ImageGenBackendRow]` — merges effective config (`get_image_generation_config()`) with the raw `[image_generation]` TOML table (via `SettingsConfigAdapter.load()`) so the UI can distinguish set-in-config vs baked-default vs env-provided.
- `FIELD_SCHEMA` — the per-backend field-spec table (below). The editor renders from this table; no per-backend ad-hoc forms.
- `diff_to_sections(edits) -> dict[str, dict]` — maps edited fields to nested section writes (`{"image_generation": {...}, "image_generation.openrouter": {...}}`).
- `probe_backend(backend_id, cfg) -> ProbeResult` — blocking probe implementation (run from a worker).

## Registration checklist (all required)

New category must appear in: the `SettingsCategoryId` enum; the rail (label **"Image Gen"**, within rail width); `_category_summaries()` (so Settings search finds it — summary keywords: image, generation, backend, swarmui, openrouter, model); the Overview category list; TCSS via source modules + `./build_css.sh` + `check_bundle_sync` (any color styling at app-tier CSS, never widget DEFAULT_CSS — bundle-outranks trap).

## Panel layout (top → bottom)

1. **Backends** — six rows (`stable_diffusion_cpp`, `swarmui`, `openrouter`, `novita`, `together`, `modelstudio`): display name, status badge, Enabled checkbox, Default radio, **Test** button.
2. **Backend editor** — fields for the selected row, from `FIELD_SCHEMA`.
3. **Generation defaults** — `default_batch`, `max_variants_per_message`, `context_llm_enabled`, `context_llm_turns`, `context_llm_timeout_seconds` (written flat into `[image_generation]`).
4. **Style templates** — read-only: "13 built-in + N user templates · manage via [image_generation.styles.<id>] or <user_data_dir>/image_generation_styles/ (editing UI planned)".
5. **Hint line** — "Test a generation end-to-end: command palette → Image Generation demo".

## Per-backend field schema (v1)

| Backend | Fields (TOML key in its nested section) | Secret | Probe |
|---|---|---|---|
| stable_diffusion_cpp | `binary_path`, `model_path`, `timeout_seconds` | — | binary exists + executable (`shutil.which` or file + X_OK); model file exists. No network. |
| swarmui | `base_url`, `default_model`, `timeout_seconds` | `swarm_token` | HTTP GET reachability on `base_url` (trusted-origins from configured base_url, same as the adapter) |
| openrouter | `base_url`, `default_model`, `timeout_seconds` | `api_key` | GET `{base_url}/models` with key → auth-verified; without key → reachability only |
| novita | `base_url`, `default_model`, `timeout_seconds` | `api_key` | reachability; auth-verified only if a cheap authenticated GET exists (implementer verifies API docs; otherwise result says "reachable (auth unverified)") |
| together | `base_url`, `default_model`, `timeout_seconds` | `api_key` | same policy as novita (Together has GET /models — use it with key) |
| modelstudio | `base_url`, `default_model`, `region`, `timeout_seconds` | `api_key` | reachability only (polling API; no cheap auth check) — "reachable (auth unverified)" |

Every non-secret field shows the **resolved effective value as its placeholder** when unset in config (e.g. empty openrouter model shows `google/gemini-2.5-flash-image`; timeouts show their baked defaults) — the task-620 lesson: never let an empty field hide what will actually be used. Model fields note env overrides where they exist (`OPENROUTER_IMAGE_MODEL`).

sd.cpp path fields are operator-owned local paths: no `path_validation` confinement (there is no base directory to confine to). A code comment states this rationale (pre-empting the recurring compliance-rule finding, cf. #862/#867/#884 declines).

## API keys / tokens

Follows the Providers & Models convention exactly:
- Masked input, paste-to-save into the local config (`[image_generation.<backend>] api_key` / `swarm_token`), never re-echoed (placeholder text communicates saved-state).
- **Source line** per secret: `env: OPENROUTER_API_KEY` / `local config key saved` / `keyring` / `missing`. Runtime precedence is env → config → keyring and the line reflects the winner.
- **Loader extension (required):** the loader currently writes the resolved secret into the config field, discarding its origin. Extend `Image_Generation/config.py`'s secret resolution to also record per-backend `key_source: str` (`"env:<VAR>"` / `"config"` / `"keyring"` / `"missing"`), exposed on `ImageGenerationConfig` (e.g. `key_sources: dict[str, str]`). One implementation of precedence; Settings only displays it. Existing behavior/fields unchanged; existing tests must stay green unmodified.
- **Clear** removes only the config-saved value and says so: "Clears the locally saved key — env/keyring sources still apply." After clearing, the source line re-renders from the reloaded config (may flip to env/keyring/missing).
- Pasted keys never enter logs, notifications, probe results, or draft persistence. `redact_secret_text` guards any validation output that could contain them.
- ModelStudio's two env vars (`DASHSCOPE_API_KEY`, `QWEN_API_KEY`) render as `env: DASHSCOPE_API_KEY` etc. per whichever won.
- SwarmUI's `swarm_token` is optional for local installs: its `missing` source renders neutrally (no error styling), unlike the four remote API keys where `missing` implies not-configured.

## Enabled/default coupling

- The Default radio is selectable only on enabled backends.
- Disabling the backend currently marked default **blocks save** with an inline message ("Default backend must be enabled — pick another default first"). No silent reassignment.
- Disabling ALL backends is allowed (runtime already refuses generation cleanly) but shows a non-blocking inline warning.
- `default_batch` > `max_variants_per_message` shows a non-blocking inline hint ("runtime clamps to the cap") — the runtime `clamp_initial_batch` already guarantees safety.

## Save model & data flow

- **Draft/dirty idiom (mandatory):** the page participates in Settings' existing draft + dirty-marker system — edits accumulate in a draft, the rail shows the dirty marker (`*` / `settings-dirty-category`), and an explicit Save applies them. No per-field immediate writes. Revert discards the draft.
- **Read:** `get_image_generation_config()` (effective values incl. `key_sources`) + the raw `[image_generation]` table from `SettingsConfigAdapter.load()` (set-vs-default distinction). Re-read on category open and after every save.
- **Write:** known keys only, to the correct nested sections, in ONE write via `SettingsConfigAdapter.save_sections({...})` (not the per-key `save_values` loop — that rewrites the file N times). `save_setting_to_cli_config`/`save_settings_to_cli_config` handle dotted nested sections and force the CLI-config cache reload (verified).
- **Writes come ONLY from user-edited draft fields — never from the effective config object.** The loader resolves env/keyring secrets INTO the effective config's fields; persisting "current effective state" would silently copy an env-provided API key into plaintext config.toml. `diff_to_sections` takes the draft, not the cfg, and a module test pins that an env-resolved secret never reaches the write payload.
- **Clearing a field deletes its key** via `delete_settings_from_cli_config(section, keys)` (exists at config.py; the adapter gains a thin `delete_values(section, keys)` wrapper). Never write `""`/`None` as a cleared sentinel. After deletion the placeholder shows the baked default that now applies.
- **Live reload:** after a successful save, call `reset_image_generation_config_cache()` (exists) so the running Console's next `/generate-image` sees the change without restart. The task-621 unknown-key warnings are unaffected: this page writes only mapped keys to correct sections.

## Probes

- On-demand only (Test button). One probe at a time; all Test buttons disable while one runs.
- **Probes test the CURRENT FORM VALUES** (including a pasted-but-unsaved key/base_url) — the user is verifying what they just typed, independent of save state. The probe never persists anything.
- Runs in a worker (`run_worker(..., exit_on_error=False)`, try/except in body). Timeout ~5s per probe regardless of the backend's configured generation timeout.
- Network probes go through the egress policy with `trusted_origins=origin_set(<configured base_url>)` — identical trust shape to the adapters (local SwarmUI must probe successfully; API-returned URLs are not involved here).
- Transport: reuse `settings_endpoint_probe`'s machinery where it fits; otherwise a minimal guarded GET via `Utils/egress` helpers. Never import the image-gen adapters for probing (they're generation-shaped, not probe-shaped).
- Results: `Reachable` / `Reachable (auth unverified)` / `Auth failed` / `Unreachable: <sanitized reason>` / (sd.cpp) `Binary found` / `Binary missing/not executable` / `Model file missing`. Reasons are sanitized — no headers, no credentials, host+status only.
- Probe state is ephemeral (not persisted; resets on category open).

## Validation & errors

- Ints/floats validated to the loader's own clamps where they exist (`default_batch ≥ 1`, `context_llm_turns ≥ 1`, `context_llm_timeout_seconds ≥ 0.1`); backend `timeout_seconds` gets a UI-enforced `≥ 1` sanity floor (the loader only type-coerces these). Invalid input keeps focus with inline error, never writes.
- `base_url` must parse as http/https with a host (shape check only — `urlparse`; no network, no egress verdict at edit time).
- Save failure (adapter returns False / raises) → user-visible notify with sanitized reason; the form keeps the user's edits (no silent revert).
- All notify/toast content is markup-escaped.

## Testing

- **Module tests** (`Tests/UI/test_settings_image_gen_defaults.py` or sibling-consistent name): `build_backend_rows` from crafted cfg+raw pairs (configured/env/keyring/missing matrix); `diff_to_sections` mapping incl. nested section names and clear-key removal; `FIELD_SCHEMA` completeness vs `_NON_SECRET`/`_SECRETS` (a drift test: every schema field maps to a real loader key); probe-result mapping incl. sanitization (a fake error carrying a header/token must not surface it); the no-secret-copy pin (env-resolved secret never appears in `diff_to_sections` output).
- **Loader tests:** `key_sources` for each precedence winner; existing loader tests unmodified.
- **Screen tests** (Textual `run_test`): rail entry present + search finds the category; fields populate from a scratch `TLDW_CONFIG_PATH` config; edit → save → the scratch file contains the nested TOML and `reset_image_generation_config_cache` took effect (`get_image_generation_config()` reflects the edit); key paste → saved + input cleared + never re-echoed; env-provided key shows `env:` source and Clear behaves per spec; disabled-default block; probe button renders a faked ProbeResult and receives the CURRENT form values (edit base_url, don't save, probe → fake prober sees the edited value); draft dirty-marker appears on edit and clears on save/revert.
- CSS bundle reproducibility (`check_bundle_sync`).

## Risks / traps for the plan

- `settings_screen.py` is 13k lines — compose/wiring hunks must stay minimal and anchored; grep-verify no drift into sibling categories.
- Rail-width label limit; category id string becomes part of the nav contract (`valid_categories`).
- The recompose/value-aware dirty-marking trap (Skills-UX memory) applies to any reactive form state here.
- Textual 8.2.7: `Select.BLANK` does not exist — use `Select.NULL` (RAG v2 lesson) if a Select is used for the default-backend picker.
