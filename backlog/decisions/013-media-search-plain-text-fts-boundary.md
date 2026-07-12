# ADR-013: Separate media plain-text search from FTS MATCH expressions

Status: Accepted
Date: 2026-07-11
Related Task: [TASK-174 - Address PR 596 verified review findings](../tasks/task-174%20-%20Address-PR-596-verified-review-findings.md)
Supersedes: N/A

## Decision

Keep `search_query` as the raw user text used by media `LIKE` predicates and allow an optional preformatted `fts_match_query` to be used only by the SQLite FTS `MATCH` predicate, threaded through the local media-reading service with backward-compatible defaults.

## Context

`MediaDatabase.search_media_db` currently reuses one value for both FTS5 `MATCH` and SQL `LIKE`. Ordinary punctuation can make the FTS expression invalid, but quoting the value before the service call also quotes the `LIKE` text and prevents legitimate matches. Existing chat-sidebar callers intentionally preformat exact-phrase FTS input and rely on the current `search_query` contract, so globally reinterpreting every caller as plain text would be a regression.

Library Search/RAG accepts plain user text. It therefore needs to preserve the raw query for `LIKE` while supplying a safely quoted expression at the raw FTS boundary.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Always quote `search_query` inside `MediaDatabase` | Breaks existing callers that intentionally pass preformatted exact-phrase FTS expressions and changes their search semantics. |
| Quote the query before calling the media service | The quoted value is also used by `LIKE`, which then misses the raw title/content text. |
| Let Library call `MediaDatabase` directly | Bypasses the media-reading scope/service boundary and duplicates mode/runtime handling. |
| Add a second Library-specific media search method | Duplicates the existing search pipeline for one parameter distinction. |

## Consequences

- Existing callers remain unchanged because `fts_match_query` defaults to `search_query`.
- Plain-text consumers can provide a safe MATCH expression without corrupting raw `LIKE` behavior.
- The optional parameter must be threaded consistently through `LocalMediaReadingService` and its scope adapter.
- Tests must cover both existing preformatted exact-phrase behavior and raw-text-plus-override behavior.

## Links

- [PR 596 review-fix design](../../Docs/superpowers/specs/2026-07-11-pr-596-review-fixes-design.md)
- [PR 596 review-fix plan](../../Docs/superpowers/plans/2026-07-11-pr-596-review-fixes.md)
