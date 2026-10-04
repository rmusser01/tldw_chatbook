# Reproducible qualification fixture — approved checkpoint

Status: approved by the user on 2026-10-04; implementation in progress, not yet
qualified. The old ignored
`.superpowers/sdd/2026-09-05-character-keyword-release-isolation` corpus and
launcher are unavailable in this task checkout. The committed historical
Keyword JSON is a receipt, not the database or a current-head pass.

## Choice

Preferred: reconstruct a deterministic, checked-in **test-only** fixture builder
and guarded launcher, with a new fixture version and fresh measurements. If the
original controller-owned corpus is supplied instead, verify its digest and
provenance before reuse. Do not manufacture the old corpus hash or simply label
its historical timings as current evidence.

## Bounded design

- Use the existing Python/SQLite production schema and Character Keyword
  indexing APIs. No new dependency, embedding model, network source or production
  control is needed.
- Create only new, explicitly selected disposable directories. Refuse existing
  destinations and real profile paths. Establish private config/data/cache,
  offline mode and null keyring before importing the application. Never inspect
  or copy real conversations, credentials or another worktree's private data.
- Build the scale corpus with exactly 10,000 conversations, 250,000 eligible
  selected-branch user/assistant messages and four excluded canaries. Keep the
  existing `perf-00000` identity family and the 30-query manifest's independently
  declared expectations, including title/body, lexical no-match and Unicode/long
  cases. Deterministic timestamps/order must satisfy the existing real-owner
  benchmark's `Fixture` top-50 contract. Reconfirm exclusion and selected-branch
  semantics from the production implementation before writing the builder.
- Produce a fresh standalone receipt only after checking schema integrity,
  eligibility/counts, actual query identities, index readiness and terminal
  descriptor retirement. Record source and fixture versions, corpus/manifest
  digests, host, raw timings and limits. Keep source corpus checkpointed and
  immutable during UI qualification.
- Prepare a separate small, clearly labelled native fixture with ordinary,
  unavailable and zero-chat character cases, multiple saved chats and open tabs.
  It should support the existing native checklist without requiring a participant
  to browse a 10,000-chat dataset.
- A guarded launcher verifies the frozen source head and keeps all mutable
  paths under its disposable profile. It does not operate Terminal or replace
  native keyboard delivery with synthetic automation. The operator opens the
  dedicated terminal window manually; existing windows and real profiles remain
  untouched.

## Verification before further production work

1. Test path refusal, deterministic manifests/independent expectations, eligibility
   and failure cleanup on a tiny real SQLite fixture.
2. Commit the reviewed builder/launcher and freeze a clean head. Generate a new
   full corpus and standalone Keyword receipt at that head.
3. Run the existing actual-owner compositor latency matrix alone at 52x20 and
   120x50. Preserve all failed samples; limits remain 50 ms maximum loop gap and
   100 ms busy paint. Do not run benchmarks alongside tests or native activity.
4. Compare a frozen baseline and observer cost before proposing any GC fix.
   ADR-198's accepted boot freeze already exists; do not duplicate it, disable GC
   or introduce a global policy based on old measurements.
5. Use the native checklist on source-bound fixtures. Windows Terminal and three
   first-time participants require actual supplied host/people evidence.
   Automated compositor checks cannot satisfy those requirements.

ADR required: no new ADR for this fixture-only proposal.
ADR paths: `backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md`
and `backlog/decisions/198-gc-policy-freeze-boot-heap.md`.
Reason: test infrastructure follows existing local-only
eligibility, privacy and qualification contracts; it introduces no runtime,
authority, persistence or global GC policy. Any later policy change requires
separate ADR review before implementation.

TASK-31246 depends on TASK-31245. Missing native/external evidence is not silently
waived to start the semantic subsystem. The final combined PR against latest dev
remains pending the agreed workstream scope and qualification checkpoints.
