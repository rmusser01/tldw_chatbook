# Library ▸ Prompts + Console injection — design

Date: 2026-07-12 (approved in-session by user)
Status: Approved design; implementation plan to follow.
Sub-project 1 of 3 (user-confirmed decomposition: Prompts → Skills → MCP Hub, one spec each;
the slash-command registry built here is the extension point the Skills spec reuses).

## Purpose

Give prompts a first-class management home in the Library (the Personas "prompts" chip is an
unwired placeholder today) and make saved prompts usable from the Console: insert a prompt's
user part into the composer (`/prompt`, and an "Insert in Console" action from the Library) and
apply a prompt's system part to the session (`/system`), with a visible, editable session system
prompt. The native Console currently sends **no system message at all** — this design adds real
system-prompt plumbing.

Decisions locked with the user:
- Library is canonical; the Personas prompts placeholder is retired.
- Prompt = two-part record (system_prompt + user_prompt) plus name/author/details/keywords —
  the existing `Prompts_DB` model, surfaced as-is.
- Injection modes: insert-into-composer (editable) and slash commands. No staged-context mode
  and no dedicated insert-prompt button in the Console UI; insertion is discovered via Library
  actions, `/` commands, and two command-palette entries. (The System rail preview row is the
  one piece of new Console chrome, chosen explicitly for system-prompt visibility.)
- Session system prompt defaults to **none** (today's native behavior). No default seeding from
  `chat_defaults.system_prompt`; that key remains legacy-path-only and is documented as such.
  Consequently the system modal has no "Save as default" action.
- v1 scope: CRUD + keywords + filter/sort, import/export, Prompts as a Search source. Follow-ups
  (not v1): version-history UI (service already supports restore), bulk chatbook export if the
  chatbook format lacks prompts, prompt collections.

## Architecture

New units (each independently testable):
- `tldw_chatbook/Library/library_prompts_state.py` — pure state builders for list/editor
  (mirror `library_notes_state.py`).
- `tldw_chatbook/Widgets/Library/library_prompts_canvas.py` — list + editor canvases (mirror
  the notes canvas; `ds-*` classes, single-row toolbar).
- `tldw_chatbook/Chat/console_command_grammar.py` — pure slash-command registry + tokenizer.
  Registry entries: name, argument hint, handler id. v1 registers `/prompt`, `/system`. The
  registry also exposes a fallback-resolver hook consulted after exact-command lookup fails and
  before the unknown-command hint — the Skills spec registers a resolver there so bare
  `/skill-name` invocation needs no grammar changes. v1 ships the hook with no resolvers.
- Console prompt picker modal + system-prompt editor modal (Widgets/Console).
- Prompts search seam in the Library search service.

Existing seams used (never raw DB from UI): `PromptScopeService`/`LocalPromptService`
(policy ids `prompts.<action>.<local|server>`), `Prompts_Interop` import parsers
(markdown/JSON/YAML/TXT + `import_prompts_from_files`), prompts FTS5, `ChaChaNotes` conversation
persistence for the per-conversation system prompt, `flush_pending_work` nav-guard,
`LIBRARY_SOURCE_PAGE_SIZES`, route-alias pattern (`notes → library` precedent).

## Library ▸ Prompts

Rail: Browse gains `Prompts (N)` (after Notes); Create gains "New prompt". Count via a
`count_prompts` passthrough (or `list_prompts` pagination total) — one seam feeds badge and list
header so they cannot disagree. Exact count even when the list page caps at the configured page
size.

List canvas: filter input (`Filter prompts… (Enter)`), toolbar `sort: Newest ▸ · Import… ·
Export…` in one non-overlapping row, rows = escaped prompt name + secondary line (author ·
keywords · relative age).

Editor canvas: `‹ Back to list`; fields Name, Author, Details, System prompt (TextArea), User
prompt (TextArea), Keywords (comma-separated). Meta line `Created … · Modified … · vN`. Actions:
`Save · Insert in Console · Export .md · Copy · Delete(dim)`. **Explicit Save** (unique-name +
optimistic-locking conflicts make autosave hostile); dirty flag + `flush_pending_work` guard on
navigation (save/discard prompt, never silent loss).

Save outcomes (each distinct and actionable):
- Name in use → "Name already in use — pick another or open the existing prompt" (+ open link).
- Soft-deleted name (`add_prompt` returns id None + "soft-deleted" message) → "A deleted prompt
  holds this name — restore it or choose another." No auto-suffixing in the editor (the suffix
  retry stays Console-save-as-only).
- `ConflictError` (edited elsewhere) → Notes-style conflict bar: Overwrite / Reload.

Import: toolbar `Import…` → existing parsers, file or folder; duplicates by name are **skipped
and reported** ("3 imported · 1 skipped (duplicate name)"); per-file outcomes, no silent partial
imports.

Export: per-prompt `.md` from the editor, emitting **exactly the format the markdown import
parser reads** (frontmatter + parts); export→import round-trip is an acceptance criterion.
Filenames sanitized via path validation. Bulk export: only if prompts are representable in the
chatbook format; otherwise a named follow-up.

Search: Prompts becomes a fourth ✓ source in Library Search using prompts FTS
(name/details/system/user). `search_prompts` gains the same optional `fts_match_query`
pass-through the notes seam has, so task-185 plural/singular expansion applies unchanged.
Result rows carry prompt provenance; per-result Open lands in the prompts editor.

Retirement/routing: delete the Personas "prompts" mode chip + constant; `prompts` route alias
resolves to Library with nav-context selecting the Prompts rail row (re-check Personas
deep-links at implementation). Remove confirmed-dead `Event_Handlers/prompt_ingest_events.py`
and unused `CCPPromptHandler` (verify unreferenced by grep at implementation time).

## Console: slash grammar, /prompt, /system

Grammar (`console_command_grammar.py`): the composer submit path consults the registry **ahead
of** `controller.submit_draft`, only for drafts that are plain text (no paste tokens) starting
with `/`. Registered command → handled locally, never sent. Unregistered `/word` → transcript-
local hint row ("Unknown command /foo — available: /prompt, /system") and the draft stays put;
**pressing Enter again sends it as plain text** (covers `/usr/bin/...`-style messages). The
armed second-Enter state disarms on any draft modification. Commands respect the existing Enter
interception order (paste-token unfurl first) and the send-blocked readiness gates; commands
that do not send (picker, system modal) work whenever the composer is usable.

`/prompt [query]`: args = whole remainder (names contain spaces), case-insensitive; exact-name
match first, then unique prefix. Unique match → the draft is replaced by the prompt's **user
part**, inserted with paste semantics (short bodies inline; bodies over the paste-collapse
threshold become the standard collapsed-paste token). No/ambiguous match → keyboard-first picker
modal, pre-filtered with the query (type-to-filter on name/keywords, Enter inserts, Esc cancels,
focus returns to the composer either way). The picker's type-to-filter queries `search_prompts`
(FTS) with a bounded page — never load-all. Library's "Insert in Console" inserts that specific
prompt directly through the same insertion path (no picker); when the composer already holds a
draft it APPENDS with paste semantics rather than replacing — existing text is never clobbered.

`/system [name]`: with a name → resolve like `/prompt` (case-insensitive, exact then prefix) but
apply the **system part** to the active session; an ambiguous match opens the same picker in
apply-system mode (selection applies the system part instead of inserting; rows whose system
part is empty render dimmed with a "(no system part)" suffix and refuse selection with that
reason); a directly-named prompt with an empty system part → inline error, session unchanged. Bare `/system` → the system editor modal.

System-prompt plumbing: `ConsoleSessionSettings` gains `system_prompt: str | None = None`
(per session/tab). `_provider_messages_for_session` prepends it as a system message when
non-empty — submit, regenerate, and continue all flow through it. Persistence: the applied
system prompt is stored with the conversation so restart+resume restores it — first choice is an
existing conversation-metadata seam; if none fits, a `system_prompt` column migration on
conversations (schema version bump + migrations/ entry) is the accepted cost.

Visibility/editing: the Model rail section gains a `System: <truncated preview>` line
(nowrap/ellipsis; dim `System: none` when unset), reflecting the active session. Click or bare
`/system` opens the editor modal: full TextArea, actions `Apply (session) · Save to Library… ·
Clear · Cancel`, with a scope line stating it applies to this session. `Save to Library…`
creates/updates a prompt whose system part is the text (name prompt, same save-outcome rules as
the editor). Command palette gains "Insert prompt…" and "Edit system prompt".

Fresh reads: picker and command resolution read through the scope service at open time — no
boot-time snapshots.

## Error handling

- Empty prompt store → one quiet line ("No saved prompts yet — create them in Library ▸
  Prompts"); never an empty modal.
- Service failures (DB locked, policy denial) → honest toast naming the action; a failed command
  never consumes the composer draft.
- Library "Insert in Console" while the Console composer is blocked (first-run setup) → navigate
  + toast "Finish provider setup to insert prompts"; nothing lost.
- All user/server-derived text rendered into labels/rows is markup-escaped.
- Import/save/conflict outcomes per the Library section; system-part-empty error per `/system`.

## Testing

- Pure: grammar tokenizer (registered/unregistered/Enter-again/disarm-on-edit/paste-token drafts
  excluded), list/editor state builders, insertion paste-threshold behavior, export→import
  round-trip, system/user part resolution, case-insensitive and space-containing name matching.
- Service (real DBs): count/list seam parity; prompts search seam provenance + plural expansion;
  unique-name, soft-deleted-name, and ConflictError paths against real `Prompts_DB` behavior.
- UI (real App + `run_test`): rail count; list→editor→save→list freshness; nav-away dirty guard;
  `/prompt` insert + picker flow; `/system` apply + rail preview update; regenerate includes the
  system message; restart+resume restores a conversation's system prompt; Personas placeholder
  gone and `prompts` alias lands on Library ▸ Prompts.
- Live gate: served captures at 2050×1240 (list, editor, picker, system modal, rail preview) and
  one real send with an applied system prompt verified in the provider request. Per-screen user
  approval before merge.

## Out of scope (named follow-ups)

Version-history UI (restore exists service-side); bulk chatbook export of prompts (pending
chatbook-format support); prompt collections; templated variables in prompt bodies; Skills
`/skill` commands (next spec, plugs into this registry); MCP Hub (third spec).
