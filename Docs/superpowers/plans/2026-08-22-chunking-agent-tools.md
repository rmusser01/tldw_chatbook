# Chunking Agent Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Four `library_*` sibling tools that give Console agents structure-aware, stored-chunk-reusing media reads plus spec management and an opt-in re-chunk — the program's original student story.

**Architecture:** New operation kinds on the existing descriptor→dispatch→service pattern (`library_tool_contract.py` → `local_library_tool_service.py` → new `local_media_chunk_tool_service.py`), reading stored `UnvectorizedMediaChunks` rows and wrapping the existing navigation tree; re-chunk refactors #2's batch into a reusable one-item function; specs ride the v7 template CRUD; two new policy actions for the mutating tools.

**Tech Stack:** Python ≥3.11, SQLite (no schema change), the #1-#3 landed machinery (engine stamps, template store, `resolve_for_rechunk`, re-chunk service), runtime_policy registry.

**Spec:** `Docs/superpowers/specs/2026-08-22-chunking-agent-tools-design.md` — §4 tool contracts, §5 wiring, §6 policy, §7 testing, §8's fifteen rulings. The plan argues from the spec.

## Global Constraints

- Reuse-stored-chunks is the read path: `library_get_media_chunk` reads `UnvectorizedMediaChunks` verbatim and never chunks implicitly (mutation-verified).
- Structure paginates **by nodes** (`max_nodes`, default 200, ≤500 + `node_cursor`); JSON payloads are never byte-sliced.
- `chunk_index` addresses disambiguate `chunk_type` (default = primary/NULL family; ambiguity across families → named error listing them).
- Revision tokens round-trip: structure/fetch payloads carry the media `version`; a mismatched supplied revision → named stale-address error (`ERROR_CONTENT_CHANGED` precedent).
- Byte budget (`MAX_RESULT_BYTES = 32 KiB`) wins over `context` (≤10): dropped neighbors are counted in a note, never truncated mid-payload.
- No-chunks degradation: structure still returns the heading tree (`chunk_summary.available = false` + re-chunk hint note); fetch → named `ERROR_FEATURE_UNAVAILABLE`-family error naming `library_rechunk_media`. Pre-v6 unstamped rows readable, `stale: true`.
- Spec save = custom templates via the existing validated CRUD only; refusals return the validator's **full errors array**; built-ins and the case-insensitive reserved `auto` name refused.
- Re-chunk: one item, spec override via #3 resolution (unresolvable → named error), atomic row replacement, `reindex` default false (mutation-verified).
- New policy resources `library.templates` (save) + `library.media` (rechunk); denials before any backend call; pinned in `Tests/RuntimePolicy/`.
- Repo rule: targeted test runs only; venv python `.venv/bin/python`.
- `library_get_media`'s cursor behavior stays **byte-identical** (existing pins must stay green).

---

### Task 1: One-item re-chunk extraction (the refactor #4 builds on)

**Files:**
- Modify: `tldw_chatbook/Library/library_rechunk_service.py` (extract `rechunk_one_item` from `rechunk_legacy_items`)
- Test: `Tests/Library/test_library_rechunk_service.py` (extend)

**Interfaces:**
- Produces: `async rechunk_one_item(media_db, media_row, *, spec: dict | None = None, rag_service=None, indexing_db=None, reindex: bool = False) -> dict` — returns the per-item outcome shape `{status, notes, chunk_summary?}`. `spec` is a PRE-RESOLVED chunking dict (or None = stored config); callers resolve names themselves.

- [ ] **Step 1 — failing test:** the one-item function called directly on a seeded legacy item (reuse the existing test fixtures) re-chunks exactly that item; a `spec` override (`{"method": "sentences", "max_size": 3}`) governs the new rows; `reindex=False` default (mutation pin: force-index never called); `reindex=True` runs the forced path. Red (function absent).
- [ ] **Step 2 — refactor:** extract the per-item body from the batch loop into `rechunk_one_item`; the batch becomes resolution + loop over the same call (behavior-identical); spec handling per Interfaces.
- [ ] **Step 3 — green:** the batch tests (existing flip/stale/atomic pins) + the new tests all green.
- [ ] **Step 4 — commit:** `refactor(chunking): extract rechunk_one_item from the legacy batch (agent-tools groundwork)`

### Task 2: Backend chunk-read method

**Files:**
- Modify: `tldw_chatbook/Media/local_media_reading_service.py` (one new method near `get_library_media_text:155`)
- Test: `Tests/Media/test_media_chunk_reads.py` (new)

**Interfaces:**
- Produces: `get_library_media_chunks(self, media_id, *, chunk_index: int, chunk_type: str | None = ..., context: int = 0, budget: int) -> dict | None` — `{chunks: [{chunk_index, chunk_type, text, start_char, end_char, word_count, metadata}], families: [distinct chunk_type values], engine_versions: [...], dropped_neighbors: int, media_version}` or None when the item has no stored rows. `chunk_type=None` sentinel for "primary (NULL) family"; out-of-range index → `KeyError`-shaped None-entry the service layer turns into the named error.

- [ ] **Step 1 — failing tests:** seeded v7 rows → exact fetch by index; NULL vs typed families (`families` correct, filter works, ambiguity when filter omitted AND >1 family); context neighbors under budget; budget overflow → fewer neighbors + `dropped_neighbors`; no-rows item → None; pre-v6 unstamped rows readable with their (absent) stamp reflected.
- [ ] **Step 2 — implement** the read method (parameterized SQL only, `deleted = 0` filter, ordering by `chunk_index`).
- [ ] **Step 3 — green + commit:** `feat(media): get_library_media_chunks backend read (family-aware, budget-bounded)`

### Task 3: The four descriptors + dispatch + read tools (structure, chunk fetch)

**Files:**
- Modify: `tldw_chatbook/Library/library_tool_contract.py` (four descriptors + schemas), `tldw_chatbook/Library/local_library_tool_service.py` (dispatch kinds + delegation), `tldw_chatbook/UI/Screens/chat_screen.py` (service wiring — constructor deps)
- Create: `tldw_chatbook/Library/local_media_chunk_tool_service.py`
- Test: `Tests/Library/test_media_chunk_tool_service.py` (new), `Tests/Library/test_local_library_tool_service.py` (extend: dispatch routing)

**Interfaces:**
- Consumes: Task 1's `rechunk_one_item`, Task 2's `get_library_media_chunks`, existing `get_media_navigation`, #3's `resolve_for_rechunk`.
- Produces: `LocalMediaChunkToolService(media_db, media_reading_service, template_interop)` with `.invoke(tool_name, arguments) -> dict` handling all four names (re-chunk/spec handlers land in Tasks 4-5; this task's versions of those two return the named not-yet error only if somehow reached — they ship together).

- [ ] **Step 1 — failing tests** (per §4.1-4.2 + §7.1-7.3): schema acceptance/rejection; structure wraps `get_media_navigation` (stubbed tree) with per-node chunk spans + summary + revision; node pagination (max_nodes + node_cursor, never byte-slice); degradation (no rows → tree + `available:false` + hint); fetch reads rows verbatim (mutation pin: no chunking call), family disambiguation, revision mismatch → `content_changed`, bounds (`context ≤ 10`), error payload shapes (`invalid_argument`/`not_found`/`feature_unavailable`).
- [ ] **Step 2 — implement:** descriptors (schemas per §4), dispatch kinds (`structure`/`chunk` route to the new service; the local service constructor gains it), the two read handlers, the node→chunk span annotation (single pass over chunks per call).
- [ ] **Step 3 — wiring verification item (spec §5):** confirm Console + MCP derive registration from descriptors (grep `builtin_tool_gate`/MCP delegation for allowlists); if a list needs the names, update it and note in the report. `chat_screen` wiring gains `media_db` + interop handles (the `getattr(app, ...)` pattern).
- [ ] **Step 4 — green:** new suites + `Tests/Library/test_local_library_tool_service.py` + existing `test_callsite`-style pins for `library_get_media` (byte-identical) still green.
- [ ] **Step 5 — commit:** `feat(library): structure + chunk-fetch agent tools (stored-chunk reads, node pagination, revision tokens)`

### Task 4: Spec tools (list/save)

**Files:**
- Modify: `tldw_chatbook/Library/local_media_chunk_tool_service.py` (the two handlers), `tldw_chatbook/Library/library_tool_contract.py` (schemas — if not already final in Task 3)
- Test: `Tests/Library/test_media_chunk_tool_service.py` (extend)

- [ ] **Step 1 — failing tests (§4.3 + §7.4):** list carries name/method/tags/`is_builtin`/validity/reserved flags (seeded store incl. an invalid + an `Auto`-cased legacy row); save routes through the REAL interop CRUD — invalid body → the full errors array in the payload; built-in mutation refused with the duplicate-hint; reserved name refused (case-insensitive); valid save round-trips into a listed custom spec.
- [ ] **Step 2 — implement** both handlers against the interop service (no new storage; list reads the decorated listing from #2's AC-24a surface).
- [ ] **Step 3 — green + commit:** `feat(library): chunk-spec agent tools over the v7 template store`

### Task 5: Re-chunk tool + policy

**Files:**
- Modify: `tldw_chatbook/Library/local_media_chunk_tool_service.py` (the handler), `tldw_chatbook/runtime_policy/registry.py` (two resources), `tldw_chatbook/UI/Screens/chat_screen.py` (policy enforcer wiring if the seam needs it — follow the collections-service precedent)
- Test: `Tests/Library/test_media_chunk_tool_service.py` (extend), `Tests/RuntimePolicy/` (extend)

- [ ] **Step 1 — failing tests (§4.4 + §7.5):** spec override via #3 resolution (explicit name → stored mode → plain; unresolvable → named error, no silent fallback); `reindex` default-off (mutation pin) and opt-in path; outcome shape (new summary, notes, never bare done); policy: both mutating tools denied before backend call when the enforcer refuses; registry pins (equality-literal pattern) for `library.templates/save` + `library.media/rechunk`.
- [ ] **Step 2 — implement** the handler (resolve spec → `rechunk_one_item` → summarize), the two registry resources, the enforcement seam.
- [ ] **Step 3 — green:** all four tools' suites + `Tests/RuntimePolicy/ -q`.
- [ ] **Step 4 — commit:** `feat(library): agent re-chunk tool + policy actions (spec override, opt-in reindex)`

### Task 6: Student story end-to-end + docs + close-out

**Files:**
- Test: `Tests/Library/test_agent_chunk_student_story.py` (new)
- Modify: `Docs/User_Guide/console/agent-runs-and-tools.md` (or the matching agent-tools page — locate at execution), `CHANGELOG.md`

- [ ] **Step 1 — the story test (§7.6):** ingest a fixture ebook (task-9 pattern) → `library_get_media_structure` → find "Chapter 7"'s chunk span → `library_get_media_chunk` fetches → assert the notes-content derived from the fetched chunks matches the source chapter text — **stored chunks only** (no re-chunk called; mutation pin). Include the no-chunks degradation leg (ingest with chunking off → structure hint → re-chunk tool → fetch works).
- [ ] **Step 2 — docs + CHANGELOG** (four tools, the reuse posture, the opt-in write).
- [ ] **Step 3 — targeted close-out:** `pytest Tests/Library/ Tests/Media/test_media_chunk_reads.py Tests/RuntimePolicy/ Tests/Chunking/ -q --ignore=Tests/Chunking/test_sync_script.py` — zero new failures vs dev baseline.
- [ ] **Step 4 — commit:** `test(library): student-story end-to-end; docs + changelog for agent chunking tools`

## Self-Review (run at save)

1. **Spec coverage:** §4.1→T3, §4.2→T2+T3, §4.3→T4, §4.4→T1+T5, §5→T3 (wiring item) + T3's constructor deps, §6→T5, §7.1-7.5→T2-T5, §7.6→T6. All fifteen §8 rulings: 1-4 (T3/T4/T5 shape), 9 (T2+T3 revision), 10 (T2+T3 family), 11 (T3 pagination), 12 (T2 budget), 13 (T3 degradation), 14 (T1/T5 atomicity doc), 15 (T4 errors array). Mapped.
2. **Ordering:** T1 (refactor) before T5 (consumes); T2 before T3; T3 before T4/T5 (service shell); T6 last. The read tools (T3) don't depend on T1 — but T3's service shell declares all four names, so T1-first keeps the not-yet window empty for read-only testing... adjusted: T1 and T2 are independent of each other; the order T1→T2→T3 is safe and puts the riskiest refactor first with the batch pins guarding it.
3. **Type consistency:** `rechunk_one_item(media_db, media_row, *, spec, rag_service, indexing_db, reindex)` (T1↔T5); `get_library_media_chunks(media_id, *, chunk_index, chunk_type, context, budget)` (T2↔T3); `LocalMediaChunkToolService(media_db, media_reading_service, template_interop).invoke(name, args)` (T3↔T4↔T5).
4. **Placeholders:** T3 Step 3's wiring check is a genuine verification item the spec itself mandates (§5), not a dodge — its output is a confirmed-or-updated registration path, evidenced in the report.
