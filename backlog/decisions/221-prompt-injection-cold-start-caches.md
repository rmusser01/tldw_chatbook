# ADR-221: Shared prompt-injection cold-start caches keyed by store generation

Status: Accepted
Date: 2026-10-06
Task: [TASK-34415](../tasks/task-34415%20-%20World-info-injection-cache-ADR-212.md)
Plan: [Non-console efficiency remediation](../../Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md), Task 3 (world info; F3+F12)
Applies to: world info now (this ADR's first consumer); chat dictionaries adopt the same contract in the follow-up task (TASK-34416).
Numbered 221 at creation: the plan's provisional 212 was already taken by [ADR-212](212-shared-adaptive-pane-shell.md) (found by re-verifying at implementation time, per `backlog/docs/lessons-backlog-hygiene.md`, "ADR numbers collide across concurrent branches").

## Decision

Every per-send prompt-injection builder (world info today, chat dictionaries
next) pays its cold start once per store generation, not once per message. The
store exposes a **monotonic generation counter**; the send path caches the
built processor keyed by `(conversation_id, character_id, generation)` in a
**bounded LRU (8 conversations)**; invalidation falls out of the key because
every store mutation bumps the generation. Keyword and regex patterns are
**compiled once per entry per generation** at entry-process time, with a
module-level `lru_cache` fallback for ad-hoc string lookups.

### 1. Store generation counter (`WorldBookManager.generation`)

- `WorldBookManager.generation` is a read-only monotonic `int`, starting at 0.
- The counter cell lives on the `CharactersRAGDB` instance (one cell per
  database object), not on the manager wrapper: the send path constructs a
  `WorldBookManager` per call, so a per-instance counter would be invisible
  across calls and could never invalidate anything.
- Every mutating method bumps it exactly once on a successful write:
  book create/update/delete, entry create/update/delete, book↔conversation
  associate/disassociate, book↔character attach/detach (embedded-snapshot
  writes), and the import path (via its underlying creates). Reads never
  bump. Failed writes (zero rowcount, no-op updates) do not bump — the store
  did not change.
- The bump is over-invalidation by design: a mutation to a book attached to
  conversation A also invalidates conversation B's cached processor. Correct,
  cheap, and requires no attachment tracking.

### 2. Resolver cache (`world_info_resolver`)

- A module-level, lock-guarded `OrderedDict` keyed `(conversation_id,
  character_id)`; each value holds `(generation, weakref-to-db, processor,
  books)`.
- Hit requires all of: same db object identity (a weakref comparison, so a
  re-opened database never inherits a stale entry), same generation, same
  character. Any miss rebuilds and replaces the entry.
- LRU bound: 8 conversations; a touch moves the key to the front, insertion
  beyond 8 evicts the least-recently-used conversation. Bound is per process
  and per db identity; memory is bounded by 8 × the active book set.
- The cache is only consulted on the resolver's self-fetch path. The
  `books=` parameter (see §5) bypasses the cache — the caller owns
  collection.
- Thread safety: the Console controller offloads the applier via
  `asyncio.to_thread` (and other screens call it from workers), so cache
  mutations take a `threading.Lock`. The lock covers only dict access —
  never the DB fetch or processor build, which happen outside it. Two
  concurrent misses for the same conversation may both build; last write
  wins, and both results are correct for the same generation. That is the
  whole worst case.

### 3. Precompiled patterns

- `WorldInfoProcessor._process_entry` compiles each primary/secondary key
  once and stores `compiled_primary_keys` / `compiled_secondary_keys`
  (tuples of `re.Pattern`) on the processed entry. Literal keys compile as
  `re.compile(r"\b" + re.escape(k) + r"\b")` — built from the exact
  case-variant the matcher uses (lowercased key for case-insensitive
  entries, so pattern reuse preserves today's lowered-key-on-lowered-text
  semantics). Regex-flagged keys compile as the raw pattern with the
  entry's `IGNORECASE` flag baked in, after the existing fail-closed
  validation downgrade.
- `_keyword_in_text` accepts a stored `re.Pattern` and searches it directly;
  a raw string falls back to a module-level
  `@lru_cache(maxsize=4096) _compiled_keyword(keyword)`, which also
  survives `re`'s module-level cache (512 slots) when a book carries more
  than 512 distinct keys — today every key past 512 effectively recompiles
  per send.
- Regex entries get the same compile-once treatment via
  `world_info_regex.regex_search` accepting a precompiled pattern
  (validation already happens once per entry at process time).

### 4. Single process per entry build; id-set recursion dedup

- `_make_candidate` reuses the already-processed entry dict instead of
  running `_process_entry` a second time (previously every entry was
  processed twice per build — once for the active list, once for the
  diagnostics candidate list).
- Recursive-scanning dedup switches from `match not in matched`
  (list-of-dicts equality, O(matched²)) to a set of entry identities.
  Processed entries carry no stable id field (raw `id` is dropped at
  processing, and character-book entries may have none), so the identity is
  the entry dict's own `id()` — exact here because the recursion returns
  references to the same `self.entries` objects, never copies, and all of
  them are alive for the duration of the scan. First-match order is
  preserved.

### 5. Pre-collected books seam

- `resolve_world_info_injection(..., books: Sequence | None = None)` (and
  the `apply_world_info_to_message` wrapper) accept already-collected
  conversation books and skip the manager fetch. This is the interface the
  Console turn will use to stop collecting books twice per send
  (`capture_prompt_transform_inputs` already holds them); wiring the Console
  caller is deferred to console-maintenance coordination and is out of
  scope here.

## Context

Measured cold-start cost paid by every send in a lorebook-attached
conversation: the resolver re-fetched every attached book from SQLite and
re-ran `json.loads` three times per entry (keys, secondary_keys, extensions);
`WorldInfoProcessor.__init__` ran `_process_entry` on every entry twice; and
`_keyword_in_text` rebuilt and re-searched `r"\b" + re.escape(k) + r"\b"`
per key per message, with `re`'s 512-slot module cache thrashing on books
with more than 512 distinct keys. The dictionary injection path (TASK-34416)
has the same shape, so the cache contract is defined once here and adopted
there.

## Alternatives

- **Per-manager-instance generation**: invisible across the per-call manager
  constructions on the send path; cannot invalidate anything. Rejected.
- **TTL-based invalidation**: serves stale entries after an edit until the
  timer expires — wrong for a single-user TUI where the edit and the next
  send are seconds apart. Rejected.
- **Unbounded cache / `functools.lru_cache` on the whole resolve call**:
  would strong-ref the db object (leaking closed connections) and key on
  unbounded (conversation, generation) pairs. The OrderedDict + weakref
  design keeps 8 conversations and no db references. Rejected.
- **mtime/counter on each book row**: requires schema changes and per-book
  read queries on every send — the exact queries this exists to remove.
  Rejected.
- **Equality-based dedup kept, made faster**: still O(matched²) in the worst
  case and slower per comparison than a set probe. Rejected.

## Consequences

- With an unchanged book set, the second and later sends in a conversation
  perform zero book queries, zero entry JSON parses, zero entry processing,
  and zero pattern compiles; the acceptance spy pins exactly this.
- A second process editing books while this one runs is not observed:
  staleness lasts until this process's next mutation (which re-bumps its own
  counter) — acceptable for this single-user, single-process TUI. Documented,
  not solved.
- Character-card content is keyed by `character_id` only: edits to a
  character's native embedded `character_book` that bypass `WorldBookManager`
  (hand-edited card JSON) do not invalidate. Embedded-snapshot attach/detach
  goes through the manager and does bump. Same staleness class as the
  second-process case; documented residual.
- Processed entries gain `compiled_primary_keys` /
  `compiled_secondary_keys`. They are internal to the processor (never
  persisted, never exported); diagnostics' signature matching is unaffected
  because it keys on `(insertion_order, content, position)`.
