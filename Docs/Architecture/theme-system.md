# Theme system

This document describes theming: the shipped theme catalog and its readability guarantees, user themes, the switch/persist engine shared by the palette and the picker, the theme editor, and how themes interact with the `$ds-*` design tokens. The CSS build and token catalog themselves are covered in [app-shell.md](./app-shell.md).

## Authoritative files

| Concern | File | Key symbols |
| --- | --- | --- |
| Catalog | `css/Themes/themes.py` | `ALL_THEMES` (the shipped-theme catalog — the count drifts, trust the list), `create_theme_from_dict`, `load_user_themes`, `sanitize_theme_variables`, `ensure_readable_text_hues`, `GUARD_VARIABLES` (`$tldw-boundary`, `$tldw-focus-fill`, `$tldw-active-edge`), reserved prefixes `custom_` / `unreadable:` |
| Switch/persist engine | `css/Themes/theme_catalog.py` | `use_theme`, `use_theme_toast`, `revert_theme`, `start_revert`, `queue_launch_default` / `settle_launch_default` (single writer thread, latest-wins) |
| Palette | `app_command_providers.py` | `ThemeProvider` — "Theme: Change Theme" + per-theme switch hits |
| Picker | `Widgets/settings_theme_picker.py` | `ThemePicker` — Use / Try / Revert / Clone / New / Edit / Rename / Delete / Export / Import |
| Editor | `Widgets/settings_theme_editor.py` | `SettingsThemeEditor` — Apply registers a process-only `custom_<name>` palette; Save/Save-as writes TOML |
| Stylesheet | `css/tie_aware_stylesheet.py` | per-theme LRU parse cache (each parse is expensive; switching back stays fast) |
| Startup | `app.py` on_mount | register `ALL_THEMES` → load user themes from the themes dir → apply `general.default_theme` (fallback `textual-dark`) |

## Switch flow (dataflow)

1. Palette or picker invokes `use_theme(app, name, …)` — the one code path both surfaces share (identical toasts by spec). An unregistered name raises before any write.
2. The launch-default persistence (`general.default_theme`) is **queued, not awaited**: numbered writes go to a single writer thread; only the latest queued write records its outcome — rapid switching persists the final choice, off the UI thread, with quit draining the queue.
3. Try = switch without persist (a revert chip tracks the revert point); Revert restores it. The editor's Apply registers the working palette as a process-only `custom_<name>` theme and switches to it; Save/Save-as writes the `[theme]` + `[colors]` (+ `[variables]`) TOML into the user themes dir, which `load_user_themes` picks up on next startup.

## Readability and safety guarantees

- `ensure_readable_text_hues` blends/overrides generated text hues for contrast at import time (shipped themes and Textual built-ins both pass through it); contrast gates are pinned by `Tests/UI/test_theme_contrast.py`.
- Every theme must define the guard variables (`GUARD_VARIABLES`: `$tldw-boundary`, `$tldw-focus-fill`, and `$tldw-active-edge`, the active top tab's outline) or stylesheet parsing fails; a defaults mixin backstops themes that omit them.
- User-theme loading is hostile-input tolerant: unreadable files, files without `primary`, and non-regular files (symlinks/hardlinks) are skipped with a warning that carries only the file **name** — one bad file never blocks startup. `sanitize_theme_variables` rejects unsafe variable values: it keeps only a colour (optionally with " NN%"), space-separated text-style keywords, or `auto NN%`. Its rules are name-aware (TASK-33003.16): `auto NN%` is dropped on a guard variable and on every name Textual generates except the seven it generates as auto (`text`, `text-muted`, `text-disabled`, `link-color`, `link-color-hover`, `block-cursor-foreground`, `button-color-foreground`), and a text style on a colour name, a colour on a text-style name, and `none` combined with other words are dropped too.
- Reserved names: saved themes may not use the `custom_` or `unreadable:` prefixes.

## Editor Save completion

The editor snapshots the submitted theme and its live name, dark flag and raw
palette controls before Save/Save-as resolves the file target. Completion writes
and registers only that submitted theme. A later same-session edit stays visible
and dirty, including invalid raw text and controls whose `Changed` messages are
still queued; only an unchanged submission returns automatically to the picker.
The `Saved` handoff carries that session and raw draft snapshot; the pane checks
both again before switching, so child input accepted before its `Changed`
message cannot be hidden during message delivery either.

`_loaded_user_theme` records the name actually saved, independently of a later
Name-box draft. Save-as updates an unchanged Name box to the saved identity while
retaining a newer palette; a later name remains untouched. The next explicit
Save uses the existing name-to-file resolution and overwrite confirmation rules.
Session fences and the existing Back/category/destination/quit and backup guards
continue to apply. Regression coverage: `Tests/UI/test_theme_pending_save_drafts.py`
(TASK-36457).

## How themes relate to `$ds-*` tokens (the load-bearing rule)

A theme's `variables` dict only has runtime effect for names **no `.tcss` source defines**. Every `$ds-*` token is defined in `css/core/_variables.tcss`, and file tokens win over app tokens — so per-theme `ds-*` entries are **inert** (only `agentic_terminal` keeps any, as documented reference values pinned by a test). Real theming of the token layer happens through polarity-aware references inside `_variables.tcss` (e.g. `$ds-status-error-readable: $text-error`, `$ds-grid-line: $tldw-boundary`) that Textual regenerates per theme. A governance test fails the build when a `$ds-*` token is referenced but undefined. Consequence: to make a theme change a token's value, change what the referenced Textual theme variable resolves to — never try to override the token from the theme dict.

## Config keys and settings surface

`general.default_theme` (default `textual-dark`) is the launch default. Settings ▸ **Theme** owns the picker + editor ("Applies immediately"); Settings ▸ Appearance holds density/font/motion plus a read-only theme row linking to the picker — Appearance no longer sets or saves a theme.

## Governing decisions and docs

ADR-150 (`150-design-token-system-and-design-language.md`; constitution in `backlog/docs/design-language.md`). Specs/plans: `Docs/superpowers/specs/2026-09-24-theme-picker-redesign-design.md` (+ three PR plans), `Docs/superpowers/plans/2026-09-03-theme-editor-ux-fixes.md`. Implementation summaries: `Docs/Development/Theme-Editor-Implementation-Summary.md`, `Docs/Theme-Editor-Horizontal-Layout-Summary.md`. Design-system contract test: `Tests/UI/test_master_shell_design_system_contract.py`.

## Verified gotchas

1. Textual's `refresh_css` clears the stylesheet parse cache on every switch; only the tie-aware subclass's per-theme LRU makes switching back fast. Dev-mode CSS hot-reload drops the subclass (documented gap).
2. `custom_` themes are process-only and excluded from the palette's switchable list; switching away persists the underlying name.
3. Palette and picker share the persistence queue, so Use/Revert ordering holds across surfaces.
4. Per-theme `ds-*` variables are inert by construction — see the rule above; this is the single most common theming misconception.
