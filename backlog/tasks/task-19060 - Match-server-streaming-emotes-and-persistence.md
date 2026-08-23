---
id: TASK-19060
title: Match server streaming emotes and persistence
status: Done
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

## Implementation Plan

ADR required: no
ADR path: `backlog/decisions/075-durable-character-emote-metadata.md`
Reason: ADR-075 already governs the pinned parser, prompt projection, local-only durable metadata, live-selection precedence, and history-restore boundary; this task implements that approved decision without changing storage schema or runtime ownership.

1. Add a pure server-compatible one-shot and bounded streaming directive parser plus frozen cross-language fixtures, proving chunk, fence, CRLF, cancellation, duplicate, cap, and UTF-16 semantics born red and then green.
2. Port the pinned server mood heuristic, then capture one off-thread/revalidated immutable run snapshot and use it for both safe prompt inventory and exact slug-to-asset resolution.
3. Extend the existing fail-soft local-only message metadata value object with bounded character-emote events and immutable visual-resolution scalars, with real SQLite round-trip coverage and no schema bump.
4. Integrate a per-message character-emote capture at the Console store streaming and citation-replacement seams so all rendered and persisted assistant text is sanitized; one store-owned atomic finalizer covers every terminal path and parser faults stay fail-closed.
5. Deliver every accepted event through a monotonic content-free feed, resolve live explicit expression state through existing Visual Identity authority while preserving manual/operational/explicit/heuristic/missing-asset precedence, and restore only the final immutable expression from history.
6. Run focused parser, prompt, store, controller, metadata, repository, migration, avatar, privacy, Persona Buddy boundary, architecture, Ruff, format, compile, and diff gates; record born-red/mutation/provenance evidence and concise implementation notes before marking Done.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Streaming and non-streaming character responses parse only assistant-visible text lines outside fenced code that match the standalone, case-insensitive `Emote: <state>` form; accepted states are trimmed and lowercased with internal whitespace replaced by hyphens, match `[a-z0-9][a-z0-9_-]{0,39}`, are capped at five events with consecutive accepted duplicates suppressed, and use pinned CRLF, arbitrary-chunk, unterminated-final-line, and JavaScript-compatible UTF-16 offset behavior. The stream buffer retains only a bounded possible directive/fence prefix, ordinary long prose is released immediately without waiting for a newline, and cancellation discards incomplete candidates with zero rendered or persisted leakage.
- [x] #2 Valid, invalid, consecutive-duplicate, and over-cap standalone directive lines never reach rendered text, persisted content, search, or exports, while fenced directives and inline prose remain visible.
- [x] #3 Character prompts deterministically project safe slugs from canonical expression keys in the active Shared Visual Identity version; invalid, ambiguous, colliding, and non-round-tripping projections are omitted, remaining slugs are deduplicated in first stored asset order, the first 25 are exposed with the exact ` (+N more)` suffix when states remain, and imported labels or arbitrary display text are excluded.
- [x] #4 Live portrait precedence remains manual override, then operational thinking/speaking until the first accepted explicit event, then explicit expression; every accepted explicit event updates the live expression immediately in stream order, and the last accepted state becomes the persisted final expression and mood label; manual display choice suppresses automatic display changes but not parsing or persistence, the heuristic runs only when no explicit event exists, and an accepted state with no asset keeps the current or base portrait with a stable fallback reason.
- [x] #5 Assistant metadata durably stores bounded final mood fields, at most five normalized `{state, at_char}` events, actor identity, immutable pack/version/expression/asset identity, and fallback reason; every offset is a nonnegative integer, offsets are nondecreasing and bounded by sanitized-text length in JavaScript-compatible UTF-16 units, references are immutable profile-local identities rather than server IDs, and no sync or server transport is authorized. Outside the bounded event records, durable visual metadata is bounded scalar metadata only; it excludes raw directives, assistant text, prompts, provider payloads, local paths, and manual overrides, and malformed metadata fails soft on load.
- [x] #6 Activated immutable visual versions are retained with no physical version garbage collection; history restores only the exact final immutable expression when available, reports a deterministic fallback otherwise, and never replays historical intra-message beats.
- [x] #7 Reasoning, tool arguments or results, citations, provider controls, Persona Buddy, and raw non-visible inputs never enter directive parsing or state control; parser, resolver, and asset failures never block or corrupt the sanitized assistant reply and produce a deterministic fixed-category fallback; diagnostics use fixed categories and identifiers and exclude assistant content, prompts, local paths, raw provider output, archive member names, bytes, and cleanup tokens.
- [x] #8 Frozen cross-language vectors and focused streaming, non-streaming, provider-tool, manual, missing-asset, persistence, history, failure, and real SQLite repository and migration tests covering durable fields and reload provide born-RED-to-GREEN evidence, mutation proof for authority, precedence, cancellation, and persistence guards, assigned-worktree provenance, isolated HOME/XDG/config/data roots, and scoped Ruff, format, compile, diff, diagnostic, privacy, architecture, and governance gates.
<!-- AC:END -->

## Implementation Notes

- Added the pinned one-shot/streaming directive parser, frozen cross-language vectors, safe prompt-state projection, and mood heuristic. Character dispatch now captures one revalidated immutable actor/pack snapshot and shares it between prompt composition, parsing, and final asset identity.
- Sanitized assistant-visible text at the Console store seam before rendering or persistence. The store owns bounded per-message capture, content-free ordered live events, fail-closed parser recovery, citation-body replacement, and one terminal metadata finalizer for complete, stopped, failed, retry, and variant paths.
- Added local-only bounded `character_emote` metadata with real SQLite round-trip and fail-soft load behavior. Search and export-source tests prove stripped control lines do not enter downstream text sinks; no schema bump, sync payload, server transport, Persona Buddy control, or physical Visual Identity garbage collection was added.
- Extended the existing avatar controller without a visual redesign: manual overrides suppress automatic painting while event cursors continue, same-chunk events paint in order, missing/corrupt explicit assets retain the current/base portrait, and completed history resolves only the recorded immutable pack/version/expression/asset identity. The Impeccable detector returned no findings.
- ADR check: no new ADR required. [ADR-075](../decisions/075-durable-character-emote-metadata.md) already governs this implementation and [ADR-067](../decisions/067-bundled-samira-visual-identity-pack.md) remains the Shared Visual Identity boundary.
- Evidence: the final touched-feature command passed 422 tests; Visual Identity lifecycle/publication/assets/contract passed 254 tests with one skip; metadata/migration/reaction-picker passed 94 tests; architecture/privacy/diagnostic/governance passed 89 tests plus the Backlog uniqueness/Windows-path guard and four provenance/CI tests. The reviewed persistent-diagnostic inventory was updated for six fixed-category/identifier-only calls.
- PR review follow-up: historical identity reads now use the shared transaction context; the three cited public helpers document Google-style Args/Returns contracts; manual display overrides use a dedicated typed avatar-request source while remaining outside `ConsoleExpressionSelection`; and the character-controller architecture oracle now requires the historical resolver dependency. Five born-RED review regressions and the affected controller/oracle suites pass.
- Born-RED evidence covered the missing parser, prompt/mood authority, metadata, store sanitizer, dispatch, precedence, and exact-history resolver. Explicit mutation checks proved the post-await authority fence, live-explicit precedence, cancellation discard, and durable metadata attachment by producing focused failures; the restored checks passed 7/7. A final self-review RED proved a server-shaped historical expression ID was incorrectly accepted before the resolver boundary was hardened; the resolver suite then passed 41/41.
- Provenance: branch `codex/task-19060-streaming-emotes`, package import `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-19060-streaming-emotes/tldw_chatbook/__init__.py`. An isolated process rooted at `/tmp/task19060-isolated.f5BsBV` created config and profile data only below that disposable HOME/XDG/config root.
- Static evidence: Ruff lint passed all touched Python files; Ruff format passed the new authorities and 14 clean touched files; byte-compilation and `git diff --check` passed. Black was not installed in the project virtualenv, so no Black result is claimed. Five pre-existing large legacy files remain outside Ruff-format scope to avoid unrelated whole-file churn; their changed code is covered by Ruff lint and the focused tests above.
- Plan deviation: historical loading was implemented as an explicit exact-identity resolver rather than extending the active-binding query, preserving the ADR requirement that history ignore later active-pack changes. No reusable incident beyond existing lessons was produced, so no lessons file was changed.
