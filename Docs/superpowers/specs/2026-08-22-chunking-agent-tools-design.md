# Chunking Agent Tools — Design Spec

**Date:** 2026-08-22
**Status:** Draft, maintainer-approved in brainstorming (four rulings in §8) plus
six design-review deltas (§8.9-8.14); awaiting maintainer's review gate.
**Sub-project:** 4 of 6 in the Chunking Parity & Agent Tools program
**Depends on:** sub-projects #1 (PR #1852, merged — vendored engine),
#2 (PR #1938, merged — template store v7, re-chunk action, stored-chunk
stamps), #3 (PR #1952, merged — auto-selection, `resolve_for_rechunk`).
Branches off `origin/dev`.
**Author:** brainstormed with the maintainer. Chatbook-side facts verified
against `origin/dev` at `95eadbc108` (post-#3 merge). No upstream facts are
load-bearing this time — every mechanism consumed already lives in chatbook
(#1-#3 shipped it).

---

## 1. Why

The program's motivating story — a student wants per-chapter notes from an
ingested book — is still undeliverable to a Console agent. Today
(`Library/local_library_tool_service.py:581 _get_media`) the only media-read
tool pages content by a **blind character cursor**
(`DEFAULT_MAX_CHARS = 8_000`, `MAX_MAX_CHARS = 16_000`,
`MAX_RESULT_BYTES = 32 KiB`): asked for "chapter 7", an agent walks fixed
windows and guesses where chapters begin.

Everything needed to do better is now in the tree and merely unreachable from
the agent runtime: stored chunks with real spans
(`UnvectorizedMediaChunks.chunk_index/start_char/end_char`, engine stamps,
template names), a heading/section navigation tree
(`Media/local_media_reading_service.get_media_navigation`), the validated
template store (#2), per-media re-resolution (#3's `resolve_for_rechunk`), and
the one-item re-chunk machinery behind the Library action (#2's
`library_rechunk_service`). #4 is the bounded `library_*`-shaped surface that
exposes them.

## 2. Goals

1. An agent can ask "where are the chapters?" and get a structure map with
   chunk-unit addresses — no window-walking.
2. An agent can fetch a unit by address and **reuse the stored chunks**
   (deterministic, version-stamped, never re-chunking behind its back).
3. An agent can list and save chunking specs (the template store's agent view)
   and reuse them.
4. An agent can opt into re-chunk-and-persist for one item, synchronously,
   with the vector re-index as a separate opt-in.
5. The student story works end-to-end from stored chunks alone.

## 3. Non-goals

- **Attached-but-not-ingested files** — follow-ups already filed
  (#1's §11: attachment extraction, attached-file chunking source).
- **Sub-project #5's ergonomics** — per-chapter fan-out, note conventions,
  flashcard output format. #4 ships the tools; #5 builds the workflow on them.
- **Batch re-chunk tooling** — one item per call; agents drive loops.
- **Any UI.** The Library action from #2 remains the human surface.
- **Schema changes.** Everything needed is already stored.

## 4. The four tools

All four are `media`-item operations in the descriptor table
(`Library/library_tool_contract.py`), following the existing
name→item_type.operation→route pattern. The dispatch table
(`local_library_tool_service.py`) gains operation kinds beyond
list/get/search.

### 4.1 `library_get_media_structure`

```
input:  {id: opaque-id, max_nodes?: int ≤ 500 (default 200),
         node_cursor?: continuation-cursor}
output: {item metadata, revision, navigation tree (paginated),
         per-node: {node_id, title, level, span, chunk_span?},
         item chunk summary: {chunk_count, chunk_family_report,
         engine_versions present, template_name?, stale: bool},
         available: bool, truncated: bool, node_cursor?}
```

- Wraps `get_media_navigation()` (node ids, depth, bounded count — unchanged),
  **annotated** with chunk facts: for each node, the `chunk_index` span
  overlapping the node's source span, computed per call (O(nodes×chunks) at
  these caps — accepted, noted).
- **Pagination by nodes, never byte-slicing** (§8.11): a truncated tree pages
  via `node_cursor`; `MAX_RESULT_BYTES` bounds only text fetches.
- **Degradation (§8.13):** items with no stored chunk rows still return the
  source-heading tree with `chunk_summary.available = false` and the note
  "no stored chunks — use library_rechunk_media to enable unit fetches".
  Pre-v6 unstamped rows count as available and `stale: true`.
- **Revision token:** the payload carries the media `version`; consumers pass
  it back on unit fetches (§4.2).

### 4.2 `library_get_media_chunk`

```
input:  {id, chunk_index: int ≥ 0, chunk_type?: str (default: primary family),
         context?: int ≤ 10 (default 0), revision?: revision-token}
output: {item metadata, the chunk: {text, chunk_index, chunk_type,
         start_char, end_char, word_count, metadata},
         neighbors included under the byte budget,
         notes: [neighbors-truncated...], revision}
```

- Reads `UnvectorizedMediaChunks` directly — **reuse-stored-chunks as the read
  path**; nothing re-chunks implicitly.
- **`chunk_type` dimension (§8.10):** the table's unique key is
  `(media_id, chunk_index, chunk_type)`; flat ingest rows carry NULL while
  ECS parent/child rows carry type values. The structure tool's
  `chunk_family_report` names the families present; the fetch's
  `chunk_type` filter defaults to the primary (flat/NULL) family. An
  ambiguous address without the filter and multiple families present → named
  error listing the families, never a silent pick.
- **Byte budget wins over context (§8.12):** neighbors are added until the
  budget would be exceeded; the payload says how many were dropped.
- **Revision check (§8.9):** when `revision` is supplied and the item's
  current `version` differs, a named stale-address error returns — the agent
  re-fetches the structure. Unverified-index (out of range / wrong family) →
  named error, never silent clamping.

### 4.3 `library_list_chunk_specs` / `library_save_chunk_spec`

- **List:** the v7 template store's agent view — name, method, tags,
  `is_builtin`, the #2 AC-24a validity flag, the reserved-name flag
  (`name_reserved`). Bounded page like other list tools.
- **Save:** create/update a **custom** (`is_builtin = 0`) template through
  the existing validated CRUD (`ChunkingInteropService.create_template` /
  `update_template`). Refusals return the **validator's full errors array**
  (§8.15 — agents self-correct), plus the CRUD's own named refusals
  (reserved `auto` name case-insensitively, per #3; built-in mutation).
  Agents never mutate built-ins; a save naming one is refused with the
  "duplicate as custom first" hint.

### 4.4 `library_rechunk_media`

```
input:  {id, spec?: {template?: name, method?, max_size?, overlap?},
         reindex?: bool (default false)}
output: {item metadata, new chunk summary (count, spans present,
         engine_version, template used), reindexed?: {done|skipped|failed},
         notes: [...]}
```

- One item per call; **synchronous chunk-row replacement** in one
  transaction (the #2/Qodo-hardened atomic pattern), reusing the per-item
  machinery extracted from `library_rechunk_service.rechunk_legacy_items`
  (a refactor into a one-item function, not a reimplementation).
- **Spec override through #3's resolution:** explicit template name → stored
  per-media mode → plain options; unresolvable name → named
  `TemplateResolutionError`-family error, never silent fallback (#3
  semantics preserved).
- **`reindex` opt-in (ruling §8.4):** default call touches chunk rows only; the
  forced vector re-index (delete-by-deterministic-id → mark → add,
  §10.2.1's path, best-effort, cache-clear) runs only when `reindex: true`.
- **Concurrency (§8.14):** no cross-process lock vs the UI action/backfill —
  per-item transactions make corruption impossible; double-work is possible
  and accepted (documented, same class as #2's ruling).

## 5. Service layer and wiring

- New `Library/local_media_chunk_tool_service.py`: the four handlers,
  backend-agnostic, mirroring the sibling services' error discipline
  (`LibraryToolError` payloads; `sqlite3.Error`/`OSError` scrubbed).
- Dispatch: `local_library_tool_service.py` routes the new operation kinds;
  the constructor gains the chunk-tool service (and whatever handles it needs
  — see the wiring verification item).
- Backend reads: one new method on `Media/local_media_reading_service`
  (`get_library_media_chunks(media_id, *, chunk_index, chunk_type, context,
  budget)`) alongside `get_library_media_text`; navigation wraps the existing
  `get_media_navigation`.
- Re-chunk + spec CRUD need a `MediaDatabase` handle — the Console wires the
  tool service with `media_service=local_media_reading_service` today; the
  plan resolves the cleanest handle source (reading service's db, or the
  app's media_db like `chat_screen`'s other seams).
- **MCP/Console registration derives from the descriptor table** per the
  contract header — the plan must verify no separate allowlist
  (e.g. `Agents/builtin_tool_gate.py`, MCP delegation lists) needs the four
  names added.

## 6. Policy

- Read tools (structure, chunk fetch, spec list): the existing library read
  path — no new verbs.
- `library_save_chunk_spec`: new registry resource `library.templates` with a
  `save` action.
- `library_rechunk_media`: new resource `library.media` with a `rechunk`
  action. (Deliberately not `rag.admin.launch` — that verb belongs to the
  RAG-admin surface per ADR-003; this is a Library-media action, consistent
  with #3's Task-13 precedent of picking the semantically-owning verb.)
- Both pinned in `Tests/RuntimePolicy/` (equality-literal pattern). Denials
  surface as named errors before any backend call.

## 7. Testing

1. Tool-contract tests per sibling: schema acceptance/rejection, bounds
   (`max_nodes`, `context`, byte budget), error payload shapes.
2. Structure: seeded v7 chunks (real spans) → node annotations correct;
   pagination pages nodes (never byte-truncated); no-chunks degradation;
   stale flag on pre-v6 rows.
3. Chunk fetch: reads the stored rows verbatim (mutation-verify: no chunking
   call happens); `chunk_type` disambiguation both ways; byte-budget
   neighbor truncation with note; revision-mismatch named error.
4. Spec tools: list carries flags; save routes through the real validator
   (invalid body → full errors array); built-in/refused-name paths.
5. Re-chunk: refactored one-item path (the #2 flip tests still green through
   the batch wrapper); spec-override resolution per #3; `reindex`
   default-off (mutation-verified) and opt-in path; policy denial.
6. **Student story end-to-end:** ingest fixture book → structure →
   "chapter 7" → unit fetches → notes derived from stored chunks only.

## 8. Decisions taken

**Brainstorm (2026-08-22):**

1. **Four sibling tools** in the descriptor pattern (not a composite, not
   overloading `library_get_media` — its cursor behavior stays unchanged).
2. **Chunk-index primary addressing** (`library_get_media_chunk`); structure
   tree provides the mapping; char-span stays `library_get_media`'s job.
3. **Specs ARE templates** — the v7 store is the spec store; list/save are
   the agent view of it. No second store (the one-store invariant).
4. **Re-chunk contract:** one item; spec override via #3 resolution;
   synchronous chunk replacement with opt-in `reindex`; new
   `library.media.rechunk` policy action.

**Design-review deltas (2026-08-22), all maintainer-accepted:**

9. §8.9 → revision tokens round-trip and are checked (stale-address named
   error), reusing the cursor-check precedent.
10. §8.10 → the `chunk_type` dimension is explicit (families reported,
    filterable, ambiguity is a named error).
11. §8.11 → structure paginates by nodes; JSON is never byte-sliced.
12. §8.12 → byte budget wins over context count (dropped-neighbor note).
13. §8.13 → no-chunks degradation keeps the story alive on unchunked items;
    pre-v6 rows readable and flagged stale.
14. §8.14 → cross-process re-chunk races accepted (transactional safety,
    possible double-work, documented).
15. §8.15 → spec-save failures return the validator's full errors array;
    spec-list carries validity + reserved flags.
