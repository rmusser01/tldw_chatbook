---
id: TASK-14801
title: Add roleplay chat identities and speaker theming
status: Done
assignee:
  - '@codex'
created_date: '2026-08-09 02:51'
labels:
  - roleplay
  - console
  - ux
dependencies: []
references:
  - backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md
documentation:
  - Docs/superpowers/specs/2026-08-08-task-14801-roleplay-chat-identity-design.md
  - Docs/superpowers/plans/2026-08-08-task-14801-roleplay-chat-identity.md
priority: high
type: feature
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give character-bound Console chats distinct accessible speaker styling, display the loaded character name, and let the human choose a global chat display name with a durable per-chat override that drives character-template user macros.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Character-bound Console transcripts render distinct theme-derived user and character row tints with stronger accessible speaker labels
- [x] #2 Assistant transcript labels use the loaded character name while generic and Persona identity semantics remain unchanged
- [x] #3 A display-only global chat name and durable per-chat override resolve predictably and renaming relabels current user rows
- [x] #4 The effective user name expands user macros only in trusted character-template content for display exports and model context
- [x] #5 Template provenance preserves safe resolved fallback content without guessing at legacy messages or changing protocol roles
- [x] #6 Related automated tests and focused live verification cover settings persistence restore rendering macro expansion and failure behavior
<!-- AC:END -->

## Implementation Plan

1. Add pure display-name validation, effective-name resolution, single-pass trusted-template expansion, and shared message presentation contracts with focused unit tests.
2. Add guarded version-one conversation metadata and closed seeded-greeting message provenance while preserving safe ordinary projections.
3. Extend the Console session/store and persistence service with per-chat identity, bounded optimistic metadata merge, projection materialization, edit-clears-provenance, and first-persist behavior.
4. Extract a shared raw character-card template composer, then seed and restore character source/projection pairs across Start Chat, picker, swap, screen state, and durable resume paths.
5. Add the canonical `[chat_defaults].user_display_name` setting and the separate per-chat Console Settings field, including validation, inheritance, warnings, and global-change refresh.
6. Route provider/context payloads, Copy, Save As, Chatbook, speech, and edit entry through the shared presentation resolver without changing protocol roles.
7. Render plain and Markdown rows with named speaker children, semantic roleplay tints, stronger theme-derived name accents, generated-bundle verification, and in-place transcript updates.
8. Run only tests related to touched files and reachable behavior, perform isolated dark/light live verification, self-review the diff, and complete Backlog evidence.

ADR required: yes

ADR path: `backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md`

Reason: the feature adds persisted display identity ownership, source/projection provenance, optimistic metadata behavior, and a cross-module presentation/context contract while extending ADR-037.

## Implementation Notes

- Added a display-only human identity contract with `[chat_defaults].user_display_name`, a durable per-conversation override, strict validation, and predictable override/global/`User` precedence. Character sessions now use the loaded character name while generic and Persona semantics remain unchanged.
- Added explicit, closed template provenance and single-pass expansion for trusted character system/greeting sources. Shared presentation now supplies transcript rows, provider/context payloads, Copy, every Save As projection, Chatbook, speech, and edit entry while preserving protocol roles and safe stored projections when provenance is absent.
- Added theme-derived roleplay row tints and stronger accessible name accents for plain and Markdown transcripts, including selection/streaming/failure/tool/system precedence and generated CSS-bundle synchronization.
- Modified the Console identity/metadata/store/persistence/controller layers, character-card composition and session/resume paths, canonical and per-chat Settings, transcript/actions/speech/export paths, theme CSS, and their directly related tests. Architecture and ownership follow [ADR-046](../decisions/046-roleplay-chat-display-identity-and-template-provenance.md).
- Automated evidence: 863 unique related test nodes passed in bounded fresh-`--basetemp` groups. Settings Hub coverage was intentionally narrowed to four display-name/save/revert/commit selectors, and selection/tail coverage to the roleplay/signature/selected/tail selector, honoring the request not to run unrelated tests. One stale terminal-status regression was reproduced on the feature head, shown to pass at base `f399cb3`, fixed in `52cbbbc8b`, and reverified with the exact node plus three adjacent selectors. Larger groups that reached the shell time cap were rerun in bounded groups; no full suite was run.
- Static evidence: Ruff over the 36 touched Python files reported the same 82 pre-existing findings at base and head, with no normalized new finding; the CSS bundle check and `git diff --check` passed. Fresh grep/self-review confirmed no new `role.title()` transcript fallback, no use of `[general].users_name`, no sequential trusted-template replacement, and no provenance on ordinary user/generated content. Independent regression and final feature reviews approved the result.
- Live harness: with `TLDW_CONFIG_PATH`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `HOME`, `USERPROFILE`, `APPDATA`, `LOCALAPPDATA`, `TEMP`, and `TMP` all rooted under `.task8-live-scratch-20260809-{d,f,h}` and `TLDW_TEST_MODE=1`, the repository interpreter `..\..\.venv\Scripts\python.exe` ran an ad hoc Textual probe against the real app and scratch SQLite store. The probe was deleted after execution and its filename was not retained, so no unreproducible placeholder is presented as an exact command; this is a documented evidence-recording deviation. No developer config or data was used.
- Live evidence: `textual-dark` at 160x48 and `textual-light` at 80x24 with character id `2`, `Alraune`, confirmed avatar rendering; `Global Rowan` -> per-chat `Captain Rowan` -> inherited `Global Rowan` -> global `Global Cecelia`; immediate transcript/template/context reprojection; literal manual `{{user}}`; edit-clears-provenance; selection and streaming/failure/tool/system legibility; Copy and Save As Note/Media/Prompt/Chatbook all projecting `Welcome, Captain Rowan.`; durable override/source reopen; and a provenance-free safe projection remaining readable and sendable. `app._notifications` was `[]`.
- Provider verification used the local/fake controller seam and inspected exact payloads; it made no network request and no real provider call. This intentionally substitutes deterministic provider-boundary evidence for the plan's live send. Pre-existing Ruff findings, inaccessible/aborted temporary basetemps, and unrelated tests were not changed or claimed as feature evidence. No new lessons entry was added because the encountered base-comparison and compositor-capture incidents are already covered by the repository's testing and live-verification lessons.
- Rebase integration renumbered this completed task from TASK-3795 to TASK-14801 because `dev` independently completed a different TASK-3795 first. The canonical task filename/frontmatter, spec, plan, and ADR links were updated together; the duplicate-ID CI check then reported no local duplicates.
