# Internal Prompts P3 — Settings Editor UI Design

**Date:** 2026-07-22
**Status:** Approved design, pending implementation plan
**Relationship:** Implements §4 of `Docs/superpowers/specs/2026-07-21-internal-prompts-settings-page-design.md` (the program spec), against the registry as actually built by P1 (#741) and P2 (#748). This doc records the **deltas** from §4 that this session's code verification and one scope addition produced; where silent, §4 governs.

## Goal

Ship the "Internal Prompts" Settings page: browse the ~29 registry-backed internal prompts grouped by subsystem, see which are customized / have a changed default, edit any prompt with a live render preview, and reset to the shipped default — all from the GUI, no source or config.toml hand-editing.

## Non-goals (deferred per §4)

Diff-vs-default view; export/import of override sets; per-prompt bespoke sample values as a registry field; onboarding new prompts. Editing is via a modal; the page is in the Expert nav group.

## Verified wiring (current HEAD, `settings_screen.py` monolith — no registry/table refactor)

Adding the category is six in-place edits, plus a test-invariant update:

1. **Enum:** add `INTERNAL_PROMPTS = "internal-prompts"` to `SettingsCategoryId` — `UI/Screens/settings_config_models.py:13-31`.
2. **Summary:** add a `SettingsCategorySummary(INTERNAL_PROMPTS, "Internal Prompts", "<desc>", "<status>")` in `_category_summaries()` — `settings_screen.py:926`. Status string shows the customized count (e.g. "3 customized" / "Defaults").
3. **Group:** add `INTERNAL_PROMPTS` to the **Expert** tuple in `_category_groups()` — `settings_screen.py:1072`.
4. **Detail pane:** add a branch in `_render_detail_pane()` (after the SPLASH branch, ~`settings_screen.py:7191`) that yields `Static("Internal Prompts", classes="destination-section settings-column-title")` then `InternalPromptsPanel(id="settings-internal-prompts-panel")` — mirrors the THEME branch at 7186-7188.
5. **Impact pane:** extend the self-contained-editor guard at `settings_screen.py:7502` to `not in (THEME, SPLASH_SCREEN, INTERNAL_PROMPTS)` (so global Save/Revert stay informational), and add a branch mirroring THEME's at 7641 showing category stats + selected-prompt metadata.
6. **Guided set:** leave `INTERNAL_PROMPTS` OUT of `GUIDED_SETTINGS_MUTATION_CATEGORIES` (`settings_screen.py:355`) — the panel owns its persistence, like the Theme editor.
7. **Tests:** update `Tests/UI/test_settings_configuration_hub.py` invariants — total category count (~:2906), Expert group membership (~:747), guided-set exclusion (~:130).

Self-contained-panel template to copy: `SettingsThemeEditor(Vertical)` + `ThemeModifiedStatus(Message)` posted from `watch_is_modified` + screen-side `@on(SettingsThemeEditor.ThemeModifiedStatus)` → `theme_editor_modified = reactive(..., recompose=True)` (`Widgets/settings_theme_editor.py`; screen wiring `settings_screen.py:746, 8043`). Modal template: `ConsoleSystemPromptModal(ModalScreen[Optional[str]])` pushed via `push_screen(..., callback=...)`, returns value through `dismiss` (`Widgets/Console/console_system_prompt_modal.py`; push site `chat_screen.py:9365`).

## Component A — Registry authoring API (`Internal_Prompts/authoring.py`)

Pure functions over `CATALOG` + config helpers; no Textual imports; unit-testable without a TUI. Keeps all config-shape logic out of the widget. Public surface:

- `iter_specs_by_subsystem() -> list[tuple[str, list[PromptSpec]]]` — CATALOG grouped, stable subsystem order (registration order), specs sorted by title.
- `baseline_hash(text: str) -> str` — short (12-char) sha256 hex of the shipped default text; the staleness fingerprint.
- `override_state(prompt_id) -> OverrideState` — a small dataclass:
  - `customized: bool` = resolved text (`get_internal_prompt(id)`) ≠ shipped default. Covers **both** a tier-1 override table and a customized legacy key (which has no baseline).
  - `default_changed: bool` = an override **table** exists AND its stored `baseline` ≠ `baseline_hash(spec.default)`. Only meaningful when a table exists (a customized legacy key has no baseline → always False).
  - `active_text: str` = the currently resolved text (what the editor prefills).
  - `has_override_table: bool` = a `[internal_prompts.<subsystem>.<key>]` table is present (distinguishes tier-1 override from customized-legacy).
  - Reads the raw override table (for `baseline`) via `get_cli_setting("internal_prompts.<subsystem>", "<key>")` → the `{text, baseline}` dict; the resolver ignores `baseline`, so authoring reads it directly.
- `save_override(prompt_id, text) -> bool` — writes `save_settings_to_cli_config({"internal_prompts.<subsystem>": {"<key>": {"text": text, "baseline": baseline_hash(spec.default)}}})`. The write-side `_target_config_section` creates the nested tables; the shape round-trips to the resolver's `_extract_text` (dict → `text`). Returns False on IO/TOML failure (config helper already catches).
- `reset_override(prompt_id) -> bool` — removes the override table via `delete_settings_from_cli_config("internal_prompts.<subsystem>", ["<key>"])` **and**, **only if the legacy key's current value differs from its shipped default**, removes the legacy key via `delete_settings_from_cli_config(<legacy_section>, [<legacy_key>])`. (Delta 1: `[prompts.document_generation.*].prompt` are written into every config at first run at shipped values — deleting an *un*customized legacy key would churn a shipped-value key for no reason and is surprising; the differs-from-shipped guard mirrors the resolver's own rule. Chunking's key isn't shipped, so any value there is a real customization.) Reset always lands on the shipped default.

`OverrideState` and these functions are the entire interface the widget consumes.

## Component B — Panel (`Widgets/settings_internal_prompts_panel.py`)

`InternalPromptsPanel(Vertical)`, mirroring `SettingsThemeEditor`; title rendered by the screen, not the widget.

- **Compose:** a search `Input#internal-prompts-search` on top; below it the grouped list.
- **Grouping (Delta 3):** a `VerticalScroll` containing, per subsystem, a non-interactive `Static` header (subsystem name + count) followed by one row widget per prompt. Each row is a focusable custom widget (`Button`/`ListItem`-style) carrying its `prompt_id`, showing title + badge chips. This structure (rather than a single Textual `ListView`, which has no first-class non-selectable headers) gives clean headers, arrow/tab focus, and per-row show/hide filtering. Rows carry a `data-subsystem`/`data-title` for filtering; the plan finalizes the exact base widget.
- **Badges per row:** **customized** (`override_state.customized`) and **default changed** (`override_state.default_changed`) — rendered as small styled `Static` chips inside the row.
- **Filtering (perf, Delta 2 sibling):** on `Input.Changed`, toggle each row's visibility (`display`) and hide a subsystem header whose rows are all hidden — targeted updates only, config read once per refresh, **no recompose / no list rebuild** in the keystroke path (the task-284 class; Textual `mount()` no-ops during pruning — verify targeted paths in the live TUI).
- **Open editor:** Enter/click on a row pushes `InternalPromptEditorModal` for that `prompt_id` with `callback=self._on_editor_result`.
- **Post-save/reset refresh (Delta 2):** the callback receives `{action, text}` or `None`. On save/reset it calls `authoring.save_override`/`reset_override` in a **thread worker** (below), then on success updates **only the affected row's badges in place** (re-read `override_state` for that one id) — NOT a full recompose — so the user's search text and scroll survive. It then posts `InternalPromptsPanel.Modified(any_customized)` for the sidebar dirty-marker / summary count.
- **Persistence worker:** `@work(thread=True, exclusive=True, group="internal-prompt-save")`; the worker body wraps the `authoring.*` call in `try/except Exception` and marshals a success/error result back via `self.app.call_from_thread(...)` (Textual `@work` defaults `exit_on_error=True` — an uncaught worker exception crashes the app, so IO must be caught inside). On failure, `notify(..., severity="error")` and leave the row unchanged. Live-apply is automatic: the config helper reloads the cache; the resolver reads through it (except `applies`-flagged snapshot prompts).
- **`Modified(Message)`** + a `watch_`-driven post, consumed screen-side into a `recompose=True` reactive (`internal_prompts_dirty` or similar) for the sidebar marker — exactly the Theme pattern. (This reactive drives the *summary/marker*, not per-keystroke row updates.)

## Component C — Editor modal (`Widgets/settings_internal_prompts_editor_modal.py`)

`InternalPromptEditorModal(ModalScreen[Optional[dict]])`, pattern from `console_system_prompt_modal.py`. Constructed with the `prompt_id` (stable — captured at open; saves are by-id, so no stale-key hazard). Composes:

- Header: title + description.
- **Contract callout** when `spec.contract_note` is set (styled warning block — e.g. "model must output TRUE/FALSE; parsed by code").
- **Required-placeholder chips** from `spec.required_placeholders` (and optional ones muted).
- **`applies` note near Save (Delta 5)** when `spec.applies != "live"` — e.g. "Applies on next search" / "Applies on next app start" — so the user sees it where they act, in addition to the impact pane.
- `TextArea#internal-prompt-editor-text` prefilled with `override_state.active_text`.
- **Live preview (Delta 4, the scope addition) — only when `spec.required_placeholders` is non-empty:** a read-only preview region below the TextArea rendering the **current editor text** through `safe_substitute(text, **sample_values)`, refreshed on a debounced `TextArea.Changed`. `sample_values` come from a **modal-side provider**: a small map of common token names → realistic examples (`query`→"What is quantum computing?", `content`→"‹document text›", `title`, `url`, `published`, `results_list`, `tool_list`, `fence_open`/`fence_close`, `content_summary`, `change_percentage`, `type`, `name`), with a visible `‹token›` fallback for any unmapped declared token. Uses `safe_substitute` (never `render_internal_prompt`) so it can't raise, leaves JSON few-shots and `{{ .Prompt }}` literal, and never triggers the P1 render-time required-placeholder warning. For zero-placeholder prompts the preview is omitted (it would be byte-identical to the editor).
- Collapsible read-only "shipped default" (`spec.default`) for comparison.
- Buttons: **Save** / **Reset to default** / **Cancel**.
  - Save: validate inline — non-empty AND every `spec.required_placeholders` token present in the text; on failure show an inline error and do NOT dismiss. On pass, `dismiss({"action": "save", "text": text})`.
  - Reset: `dismiss({"action": "reset"})` (disabled/no-op when `override_state.customized` is False).
  - Cancel / escape: `dismiss(None)`.

The modal performs no IO itself — the panel's worker does the save/reset, so the modal stays pure and testable.

## Impact pane (right column)

Category-level: total prompts, customized count. Selected-prompt: `used_in`, placeholders, and the `applies` note. Mirrors the THEME impact branch shape.

## Error-handling model (unchanged from §5)

| Layer | Behavior |
|---|---|
| Modal (save) | Empty text or missing required placeholder → inline error, no dismiss. Only layer that talks to the user pre-persist. |
| Panel worker | `authoring.*` IO caught inside the worker; failure → `notify` error, row unchanged; success → in-place badge refresh. Never lets `exit_on_error` crash the app. |
| Resolver (runtime) | Unchanged: never raises for user-caused problems; invalid override → warn once → shipped default. |
| Reset | Deletes override table + differs-from-shipped legacy key; always lands on shipped default. |

## CSS (Delta 2 to §4's location claim)

New `settings-internal-prompts-*` / row / badge / chip classes go in **`css/components/_agentic_terminal.tcss`** (where the current destination-native `settings-*` classes actually live — NOT `_workbench.tcss`/`_tools-settings.tcss` as the program spec §4 stated). Never hand-edit the generated bundle `css/tldw_cli_modular.tcss`; add the partial content and let `build_css.py` rebuild (auto at app start when a partial is newer).

## Testing

- **Authoring-API unit tests** (no TUI, scratch config via the existing `scratch_config` fixture pattern): `override_state` precedence and the customized/default_changed/has_override_table combinations (no override; tier-1 override with matching and stale baseline; customized legacy key; legacy key equal to shipped → not customized); `save_override` round-trip (write → resolver returns it → `override_state.customized` True, baseline fresh); `reset_override` deletes the table AND a customized legacy key but LEAVES an uncustomized shipped-value legacy key; `baseline_hash` stability.
- **UI tests** (`app.run_test()`, scratch `TLDW_CONFIG_PATH` profile): category renders in Expert group; search filters rows without a rebuild; opening a row mounts the modal prefilled; Save validates (blocks on missing required placeholder) and round-trips to config; Reset returns to default; badges reflect config state after save/reset (targeted, no recompose); preview updates on edit for a templated prompt and is absent for a zero-placeholder prompt; worker-error path shows an inline/notify error without crashing.
- **Invariant updates:** `Tests/UI/test_settings_configuration_hub.py` category count / Expert membership / guided-set exclusion.
- **Import hygiene:** `authoring.py` may import config lazily like the resolver; the existing subprocess guard test must still show `tldw_chatbook.Internal_Prompts` importing without `tldw_chatbook.config` at module import (authoring's config use is call-time).

## Merge gate

Per this project's standing rule that every new Settings screen requires explicit user screenshot-approval before merge, P3 includes a live-TUI QA pass (real screenshots of the page, a filtered list, the editor modal with preview and contract callout, a saved/customized badge, and a reset) brought to the user for sign-off **before** any merge.

## Accepted limitations

- Panel-local search text and scroll reset when navigating away from and back to the category (category switch recomposes the detail pane) — consistent with every other Settings category; the in-place refresh (Delta 2) only preserves them across a save/reset, not across category navigation.
- Sample values for the preview are generic (common-token map + `‹token›`); a per-prompt sample field is deferred.
- The `applies`-flagged prompts (rerankers "next search", rolling-summarize "process restart") apply on the stated boundary, not instantly; the modal + impact pane say so, but the UI cannot force a live reload of a snapshot-at-init consumer.
- Reset leaves an empty `[internal_prompts.<subsystem>]` table header in config.toml when it removes the last override in that subsystem (`delete_settings_from_cli_config` does not prune now-empty parent tables). Cosmetic only — the resolver reads nothing from an empty table and falls through to the shipped default (verified). Not worth empty-parent-pruning complexity.
- `default_changed` cannot be detected for a prompt customized only via a hand-edited legacy key (no stored `baseline`); the first UI save migrates it to a `{text, baseline}` table, after which staleness is tracked. Only hand-edited legacy keys that never pass through the editor lack a baseline.

## Verification performed at design time

The load-bearing persistence assumption was checked empirically against the real merged code (scratch `TLDW_CONFIG_PATH`): `save_settings_to_cli_config({"internal_prompts.<subsystem>": {"<key>": {"text": ..., "baseline": ...}}})` writes a nested table that the resolver reads back as an override (both `{text, baseline}` table and plain-string shapes); `delete_settings_from_cli_config("internal_prompts.<subsystem>", ["<key>"])` returns the prompt to its shipped default; the raw table (with `baseline`) reads back via `get_cli_setting("internal_prompts.<subsystem>", "<key>")`. All confirmed before this plan.
