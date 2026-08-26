---
id: TASK-2374
title: Media, notes, and conversation Use-in-Console handoffs delivered zero content to the model
status: Done
assignee:
  - '@claude'
created_date: '2026-08-04 20:07'
labels:
  - console
  - rag
  - honesty
dependencies: []
priority: high
---

## Description

While verifying the D1c sub-claim (task-2370), PR-T1 Task 2's review found a live content-loss bug distinct from anything in the original critique: `capture_console_staged_evidence_for_chat` returned `(None, None)` whenever no `evidence_bundle` was present on the staged handoff. This meant "Use in Console" handoffs from media, notes, and conversations displayed their content as staged in the strip and Inspector, while the actual send to the model delivered NOTHING — the model never saw the content the user believed they had attached.

## Acceptance Criteria

- [x] Media, notes, and conversation "Use in Console" handoffs deliver their staged content to the model when the message is sent
- [x] A capture round-trip test covers each of the three kinds (conversation added per review request)
- [x] The existing RAG-gated evidence branch, snippet cap, and send-gating behavior are unchanged (no regression to already-correct RAG staging)

## Implementation Notes

Fixed in PR-T1 Task 9, commits `f67b1b3d4` (initial) and `5cf126ed5` (added conversation round-trip test per review).

The initial approach dropped the `"rag" in source` gate in `console_chat_controller.py` entirely, rather than special-casing media/notes/conversations, because doing so surfaced a **fourth** affected source not named in the brief (`search-web`) — a narrow allowlist patch would only have relocated the same bug to the next unnamed kind. Seven new tests were added, including a capture round-trip per kind and a pin that the pre-existing RAG branch, snippet cap, and send-gating behavior are byte-unchanged.

Review (sonnet) required a fix round: the initial report overclaimed scope, describing the defect as resolved when in fact **five other live handoff kinds** (`skills`, `watchlists_collections`, `library-source-snapshot`, `study`/quiz, `personas`-attach) still deliver nothing — not because of the now-removed "rag" gate, but because of an unrelated, untouched allowlist (`_SOURCE_ALIASES` in `RAG_Search/local_citation_capture.py`) that rejects their source types. This was empirically confirmed (`context=None` for these kinds) and is tracked separately as task-2375. The review also ruled content fidelity for media/conversation snippets "thin-but-honest" (a real attributed reference, but a weak snippet) rather than a blocker — tracked separately as task-2376.
