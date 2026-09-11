# Console archive and conversation recovery

Scope: complete individual/bulk archive, discoverable recovery, original-conversation resume, transcript review, visible archive state, workspace name collision recovery, accurate close copy, explicit search scope, and compact/wide usability.

## User outcome

Users can archive one/many saved chats, find archived chats by title or message text, read their transcripts without changing the current session, restore only or restore and resume the original conversation. Whole-workspace recovery stays distinct and is available where archive starts.

## Behavior

1. Library conversations has Active/Archived/All scope, exact count and paged search, row metadata including workspace/state/date, adjacent transcript review at wide sizes and a usable stacked/focused view at compact sizes. Message review is bounded/paged and supports finding and moving between matches. Single-page pager chrome is suppressed.
2. Archive and Restore support the selected conversation and multi-selection with explicit counts. Archive receipts retain changed IDs/versions for Undo and View archived. Busy/unsaved conversations are refused with actionable feedback. Stale targets are not silently mutated.
3. Resume conversation is a true original-ID activation; Use as source remains separate. Archived conversations offer Restore and resume. If the containing workspace is archived, disclose restoration of that whole workspace before proceeding. Restore-only never switches context.
4. Console exposes Archive chat and a direct full conversation search/archive route from the rail and quick switcher. The quick switcher explains title/workspace/status matching, and preserves typed text when opening full search. Normal history excludes archived saved chats.
5. Workspace switcher can show archived workspaces, restore them, recover collisions with Restore as, and provides Undo/View archived following archive. Settings renders literal archived state, provides the same restore naming ability, and confirms restoration without activating. Default remains protected.
6. Close distinguishes retained saved history from temporary/unsaved messages, drafts, queued prompts and running turns. Documentation teaches both novice and keyboard workflows accurately.

## Architecture and constraints

ADR: `backlog/decisions/147-conversation-archive-and-exact-resume.md`.
Python >=3.12; Textual >=8,<9; SQLite and existing helpers only. No new dependency. Existing theme/tokens and keyboard conventions are preserved. DB/large transcript work runs off the UI thread. Stable IDs, optimistic version checks and selected branch are retained. Input and markup boundaries sanitize user titles and paths. Full test suite requires separate user opt-in; use relevant targeted tests.

## Validation

Real-SQLite archive/search/restore roundtrip, migration from current schema, exact count/page semantics, name collision, stale Undo, partial bulk failure, duplicate/open sessions, unrelated drafts, busy/queued guards, read-only transcript review, exact branch and subsequent send ownership. Mounted screenshots and keyboard at 100x30 and 160x44 validate status labels, usable review/actions and focus. Do not claim a live provider send from a mocked response.
