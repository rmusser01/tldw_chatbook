# tldw_server2 → Chat Dictionaries port research (digest)

Source of full analysis: research agent over `/Users/macbook-dev/Documents/GitHub/tldw_server2` (canonical, not `.next/standalone`) vs tldw_chatbook.

## Meta-finding
tldw_chatbook's dictionary backend is **already built** (local + server) via `chat_dictionary_scope_service.py` + `local_chat_dictionary_service.py` + `server_chat_dictionary_service.py`: CRUD, `reorder_entries`, **`process_text` (the preview/test-substitution call)**, `import/export_markdown`, `import/export_json`, `list_activity`, `list_versions`/`get_version`/`revert_version`, `get_statistics`, and `list_unsupported_capabilities` (degraded-capability reporter). **So the Dictionaries workbench is UI construction, not backend.** One backend gap: `bulk_entries` (server-only; no local impl / no scope wrapper).

## Server entry model — fields tldw_chatbook's `entries_json` lacks
Per-entry: explicit `type` literal|regex (+ regex flags i/m/s/x), `probability` 0–1, `case_sensitive`, `group` (best-of-group scoring), `timed_effects` {cooldown, delay} (SKIP `sticky` — unimplemented server-side too), `max_replacements`, per-entry `enabled`, `priority`/`sort_order` (distinct from group), per-entry usage_count/last_used, Jinja templates (gated).
Dictionary-level: `category` + `tags`, `included_dictionary_ids` (composition, cycle-checked), `default_token_budget`, `version` (optimistic lock), `processing_priority`, used-by-chat summary.
Regex safety: ReDoS heuristic blocklist + 500-char cap + timeout-capable `regex` compile that degrades to no-op instead of hanging.

## UX the server settled on
- **List:** sortable table, inline active/inactive Switch (not read-only), used-by count + scope badges, entry-count + regex/literal split badge, Duplicate, empty-state starter templates (Medical Abbrev / Chat-Speak / Custom Terminology / Character Speech).
- **Entry editor:** inline drawer (NOT modal-in-modal — was an audit defect); Simple ⇄ Advanced toggle (advanced = probability, group autocomplete, timed effects, case-sensitivity, max replacements); bulk multi-select actions; LIVE ReDoS validation as you type a regex.
- **Regex vs literal:** explicit `type` badge, never inferred from `/…/`.
- **Validation panel:** structured `{code, field, message}` errors/warnings + entry_stats + partial/timeout flag → a portable **validation taxonomy** (codes, not prose) that maps to a TUI results list with jump-to-entry.
- **Import/export:** JSON (full) + Markdown (LOSSY for advanced fields → warn banner); import preview/confirm + 409 rename/replace/cancel.
- **Attach-to-conversation:** searchable chat picker, count-labeled confirm ("Assign to 3 chats"). CAVEAT: server's version has a bug (omits workspace chats) + no reverse "which dicts attach to THIS conversation" index — don't copy the bug; DO add the reverse index.

## Prioritized ports (TUI, keyboard-first)
- **HIGH:** (1) **substitution preview/"Try it" panel** (paste text → diff + fired entries; backed by wired `process_text`). (2) inline enable/disable toggles (list-level + entry-level). (3) validation panel with structured codes. (4) expose per-entry `type`/`case_sensitive`/`enabled`/`priority` in the editor (backend dataclass mostly supports; add `type`/`case_sensitive`/`enabled`).
- **MED:** regex/literal badge; timed-effects (cooldown/delay) fields; Duplicate; bulk ops (needs local `bulk_entries` backend); import/export reuse + lossy-Markdown warning; attach-to-conversation + reverse used-by index.
- **LOW:** starter templates; statistics (incl. portable ~140-line pattern-conflict detector); version history/revert (verify local capability-gate isn't stale); activity trail; dictionary composition (backend gap); category/tags (backend gap).
- **DO NOT PORT:** server's quick-assign workspace-chat omission (a bug).

## "Try it" (substitution preview) — TUI design brief
Inputs: multiline sample text; optional token-budget + max-passes (advanced); **saved test-cases** (name+text, persisted). Run = a visible keybinding (not hidden in a collapsed accordion — server audit flagged that).
Output (from `process_text`/`ProcessTextResponse`): (1) **diff** — original with removals struck/dim + processed with additions highlighted (two Rich panes, or unified if narrow; explicit "No differences" empty state); (2) processed text (copyable); (3) stats strip — replacements, iterations, **entries_used (clickable → jump to entry)**, token_budget_exceeded warning badge. Graceful degradation: a bad regex → "entry did not fire", never a hard error. Plus a lightweight per-entry "test" (row keybinding `t` → matches yes/no + first-match highlight).

## Chatbooks (brief)
Export/import **bundle** feature (job-based, versioned manifest); `DICTIONARY` is a first-class ContentType serialized via the dictionary's own `export_json`. No management-UX pattern to mine; transferable idea = a dictionary is already a portable bundle unit (reuse its JSON export shape for any future backup/export).
