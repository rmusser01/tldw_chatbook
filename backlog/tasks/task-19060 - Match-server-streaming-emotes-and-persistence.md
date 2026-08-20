---
id: TASK-19060
title: Match server streaming emotes and persistence
status: To Do
assignee: []
created_date: '2026-08-20 18:53'
labels: []
dependencies:
  - TASK-16319
references:
  - backlog/decisions/067-bundled-samira-visual-identity-pack.md
  - backlog/decisions/075-durable-character-emote-metadata.md
  - >-
    Docs/superpowers/specs/2026-08-20-actor-pack-persona-buddy-and-emote-programme-design.md
  - >-
    https://github.com/rmusser01/tldw_server/commit/385afa951922c8a9dc2002c675bb6cad65e4ac23
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Match the server's explicit streaming character-emote behavior so reaction directives drive live portraits while remaining absent from visible and persisted assistant text, with durable final-expression history restore.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Streaming and non-streaming character responses parse only assistant-visible text lines outside fenced code that match the standalone, case-insensitive `Emote: <state>` form; accepted states are trimmed and lowercased with internal whitespace replaced by hyphens, match `[a-z0-9][a-z0-9_-]{0,39}`, are capped at five events with consecutive accepted duplicates suppressed, and use pinned CRLF, arbitrary-chunk, unterminated-final-line, and JavaScript-compatible UTF-16 offset behavior. The stream buffer retains only a bounded possible directive/fence prefix, ordinary long prose is released immediately without waiting for a newline, and cancellation discards incomplete candidates with zero rendered or persisted leakage.
- [ ] #2 Valid, invalid, consecutive-duplicate, and over-cap standalone directive lines never reach rendered text, persisted content, search, or exports, while fenced directives and inline prose remain visible.
- [ ] #3 Character prompts deterministically project safe slugs from canonical expression keys in the active Shared Visual Identity version; invalid, ambiguous, colliding, and non-round-tripping projections are omitted, remaining slugs are deduplicated in first stored asset order, the first 25 are exposed with the exact ` (+N more)` suffix when states remain, and imported labels or arbitrary display text are excluded.
- [ ] #4 Live portrait precedence remains manual override, then operational thinking/speaking until the first accepted explicit event, then explicit expression; every accepted explicit event updates the live expression immediately in stream order, and the last accepted state becomes the persisted final expression and mood label; manual display choice suppresses automatic display changes but not parsing or persistence, the heuristic runs only when no explicit event exists, and an accepted state with no asset keeps the current or base portrait with a stable fallback reason.
- [ ] #5 Assistant metadata durably stores bounded final mood fields, at most five normalized `{state, at_char}` events, actor identity, immutable pack/version/expression/asset identity, and fallback reason; every offset is a nonnegative integer, offsets are nondecreasing and bounded by sanitized-text length in JavaScript-compatible UTF-16 units, references are immutable profile-local identities rather than server IDs, and no sync or server transport is authorized. Outside the bounded event records, durable visual metadata is bounded scalar metadata only; it excludes raw directives, assistant text, prompts, provider payloads, local paths, and manual overrides, and malformed metadata fails soft on load.
- [ ] #6 Activated immutable visual versions are retained with no physical version garbage collection; history restores only the exact final immutable expression when available, reports a deterministic fallback otherwise, and never replays historical intra-message beats.
- [ ] #7 Reasoning, tool arguments or results, citations, provider controls, Persona Buddy, and raw non-visible inputs never enter directive parsing or state control; parser, resolver, and asset failures never block or corrupt the sanitized assistant reply and produce a deterministic fixed-category fallback; diagnostics use fixed categories and identifiers and exclude assistant content, prompts, local paths, raw provider output, archive member names, bytes, and cleanup tokens.
- [ ] #8 Frozen cross-language vectors and focused streaming, non-streaming, provider-tool, manual, missing-asset, persistence, history, failure, and real SQLite repository and migration tests covering durable fields and reload provide born-RED-to-GREEN evidence, mutation proof for authority, precedence, cancellation, and persistence guards, assigned-worktree provenance, isolated HOME/XDG/config/data roots, and scoped Ruff, format, compile, diff, diagnostic, privacy, architecture, and governance gates.
<!-- AC:END -->
