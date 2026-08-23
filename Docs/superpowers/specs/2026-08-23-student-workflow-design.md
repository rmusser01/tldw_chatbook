# Student Workflow — Design Spec

**Date:** 2026-08-23
**Status:** Draft, maintainer-approved in brainstorming (four rulings in §7) plus
two design-review passes (§7.5-7.11); awaiting maintainer's review gate.
**Sub-project:** 5 of 6 in the Chunking Parity & Agent Tools program
**Depends on:** sub-projects #1-#4 (all merged; #4 = PR #1976, `684c6aba4` —
the library agent tools this workflow rides). Branches off `origin/dev`.
**Author:** brainstormed with the maintainer. Facts verified against `origin/dev`
at `684c6aba4`. No upstream facts are load-bearing — every mechanism consumed
ships in chatbook.

---

## 1. Why

The program's motivating story — "a student wants per-chapter notes from an
ingested book, or flashcards per section" — is one sub-project away from done.
#4 shipped the agent tools (structure map, stored-chunk fetch, specs, re-chunk)
and proved the read path end-to-end. What the acting agent still cannot do:

- **Write its output anywhere.** The library tool namespace has exactly two
  write tools (spec-save, re-chunk); notes are list/get only. The per-chapter
  notes the workflow exists to produce have no landing surface.
- **Group or provenance-tag what it would write.** Note folders exist in the
  notes UI (the v36 schema behind `NotesScopeService`) but nothing
  agent-reachable creates them; note↔media links do not exist at all.
- **Emit flashcards** in any form a student can see.

Meanwhile the fan-out half needs **no new machinery**: `spawn_subagent` is a
runtime tool (gated on `max_subagents > 0`), `FleetCoordinator` coordinates
concurrent sub-agents, and budgets are wired. The ergonomics leg is
conventions and documentation riding what exists.

## 2. Goals

1. An agent can save a note (create by default, update by id+version), with a
   folder affordance that is safe under concurrent sub-agent fan-out.
2. Study outputs are visible to the user the moment they land (notes screen),
   grouped per book, with structured provenance back to the source media.
3. The flashcard output decision is made deliberately and its trade-off
   recorded (ruling §7.3).
4. The end-to-end pattern — structure → spawn-per-chapter → fetch → save →
   re-run updates — is documented and proven by an upgraded story test.

## 3. Non-goals

- **New orchestration code** — `spawn_subagent`/fleet/budgets exist; the
  fan-out leg is documentation (§5).
- **A prompt preset** — the pattern lives in the user guide; a preset would
  drift from the tools (docs first; preset only if the maintainer asks).
- **Note↔media link tables** — provenance rides a content header (§4.2);
  schema work nobody asked for.
- **Note deletion** — the workflow never needs it; refusing to add it is the
  scope guard.
- **A flashcards viewer/SRS** — filed as follow-up (§6); #5's flashcard
  output rides notes.
- **Title-keyed upsert semantics** — notes have no unique title constraint;
  inventing one is out (ruling §7.7).

## 4. The tool — `library_save_note`

One new write descriptor in the `library_tool_contract.py` pattern (`note`
item, `save` operation, the `Writes local Library data only` tail).

### 4.1 Contract

```
input:  {title: str (1..TITLE_MAX), content: str (1..CONTENT_MAX),
         folder?: str (1..FOLDER_MAX), note_id?: opaque-id,
         expected_version?: int ≥ 1}
output: {item: {id: opaque-id, "note", title, folder},
         version: int, created: bool, notes: [...]}
```

- **Create by default**; **update when `note_id` + `expected_version`** are
  supplied. Version mismatch → the named `content_changed` error
  (`update_note` returns bool — a falsy result maps to the named error;
  exact failure mode verified at implementation). The returned id + version
  make idempotent re-runs the agent's explicit choice.
- **Bounds are input-side** (ruling §7.6): `maxLength` on `title`, `content`,
  and `folder` in the schema — mirroring the caps Task 4's spec-save body
  used (verify-and-match at implementation; a sensible default is
  content ≤ 100_000 chars, title ≤ 512, folder ≤ 256) — so an agent cannot
  push a megabyte into the notes DB through the tool.
- **Folder is one segment** (ruling §7.5 corrected on third review: the
  underlying model is a **tree** — `get_folder_by_path(folder_segments)` and
  `parent_id` are native — so the v1 one-level choice is simplicity, not a
  model limitation; a path-taking variant is a trivial future extension).

### 4.2 Provenance header convention

No note↔media link exists (verified: `create_note_link` is note↔note only)
and none is built. The convention is a structured header at the top of the
note content, documented in the tool description so agents emit it:

```
source: <media opaque-id>
revision: <media version>
chapter: <chapter title>
chunks: <first>-<last>
```

The `revision` line is load-bearing (ruling §7.8): a chunk span is meaningless
for staleness without the media version it was derived from. The header is a
convention, not enforced code; tests assert the pattern round-trips.

### 4.3 Write seam (the split reconciled — ruling §7.5)

The tool's note backend is `app.notes_service` (the legacy
`NotesInteropService`: `add_note(title, content)`, `update_note(note_id,
update_data, expected_version)`) — that is where note rows are written.
Folders live only in the async `NotesScopeService`
(`app.notes_scope_service`). The handler lives in
`local_library_tool_service.py` as a new `save` operation branch on the note
backend, takes **both** handles, and:

- writes/updates the note via the legacy interop (the notes UI's own
  row-writer);
- ensures the folder via an **idempotent helper**, now with the verified
  API: `get_folder_by_path([name])` (repository, path-native) →
  `create_note_folder` on miss → **re-query on conflict**
  (create_note_folder is not idempotent; the race is tolerated by
  re-reading); placement via `attach_note_to_folder(scope="local",
  folder_id, note_id)` (scope-level, verified; repository `attach_manual`
  revives the latest membership history, so re-attach is safe). **Scope is
  pinned to `local` (`ScopeType.LOCAL_NOTE`)** — the notes UI's own scope;
  any other scope makes folders invisible in the screen.
- bridges the async scope calls with the established `asyncio.run` pattern
  (#4's precedent; same caller-threading guarantees).
- Both construction sites (chat_screen factory, MCP/server build) gain the
  `notes_scope_service` handle via the `getattr(app, ...)` degrade pattern.

### 4.4 Duplicate-window ruling (§7.7)

Notes have no unique constraint on title. Create-default + explicit-update
means a re-run that does not list-first can mint duplicates, and two
concurrent identical creates both succeed. **Accepted deliberately** (same
class as #2's cross-process races): the documented convention is
**list-before-rerun** (`library_list_notes` in the folder, match by title,
update via id); title-keyed upsert would invent natural-key semantics the
schema does not have. The `note_import_executor` reconcile-by-id precedent
is noted and deliberately not extended to titles.

## 5. Conventions and the fan-out pattern (docs only)

The user guide gains the end-to-end pattern, documented once:

```
1. library_get_media_structure(id)          → the chapter map
2. for each chapter (spawn_subagent):       → the existing runtime tool
     library_get_media_chunk(...) for the node's span
     (optional) summarize/extract per the user's ask
     library_save_note(title, content-with-provenance-header,
                       folder="Study/<book title>")
3. re-run: library_search_notes(query=title) first (the list tool has no
   folder filter — ruling below) → update via note_id + version
```

**Re-run disambiguation ruling (third review):** `library_list_notes` has no
folder filter (verified — the generic `_list_schema` is limit/offset only)
and its payload carries no folder info, so the cross-session re-run
convention is **search-based**: `library_search_notes(query=<title>)`, with
the agent disambiguating by reading. Within one session the orchestrating
agent holds the saved ids directly. A folder-filtered `library_list_notes`
is filed as a follow-up candidate if false positives bite in practice.

Study-note conventions (the `Study/<book>` folder name, chapter-titled notes,
the provenance header) are documented patterns the tool makes easy — not
enforced code. Flashcard output is the Q/A markdown convention below.

## 6. Flashcards — the deliberate decision

**Ruling §7.3: #5's flashcard output is Q/A markdown inside notes.** The
real-rows path (`decks`, `flashcards`, `flashcards_fts`… — the data layer
exists) has **no screen route**: writing real rows ships output the student
cannot see anywhere in the app. QA-in-notes is visible the moment it lands.
The real-rows path stays open for whenever a flashcards viewing/SRS surface
exists — **filed as a follow-up task at implementation close-out** (house
pattern; the spec's §11-equivalent is this section).

## 7. Testing

1. Tool-contract tests: schema acceptance/rejection (bounds!, required,
   unknown keys), error payload shapes.
2. Create/update/conflict: create → id+version; update with matching version
   → version bumps; stale version → `content_changed`; unknown note_id →
   `not_found`.
3. Folder idempotency: same name twice sequential → one folder; concurrent
   (simulated interleaved) creates → one placement target after the
   re-query; folder-less create → no folder touched.
4. Provenance-header pattern round-trip (assert the documented shape).
5. Policy: `library.notes`/`save` resource, mapping on BOTH Console-direct
   and MCP paths (the #4 lesson — landed with the handler), denial before
   any backend call (mutation-pinned), RuntimePolicy equality pins.
6. **The upgraded student story (§7.6 of #4's test, extended):** the full
   read path from #4's test now *saves* the Chapter-7 note, re-reads it
   through `library_get_note`, and a **re-run leg** proves
   search-first (the third-review correction to "list-first") +
   update-not-duplicate; the flashcard convention leg saves a
   QA-note and re-reads it.
7. Suites: `Tests/Library/`, the notes scope/interop suites touched,
   `Tests/RuntimePolicy/`, MCP local-control.

## 8. Decisions taken (brainstorm + two review passes, 2026-08-22/23)

1. **Scope shape:** one write tool (`library_save_note`) + the flashcard
   target + conventions; fan-out leg is conventions-only (no orchestration —
   `spawn_subagent`/fleet already exist); no structured fan-out tool
   (duplicates the runtime, hides the reasoning).
2. **Write posture:** create-default, update via `note_id` +
   `expected_version` (mirrors #4's spec-save shape; optimistic locking is
   real — the notes sync triggers carry `NEW.version`).
3. **Flashcards: Q/A-in-notes now;** real-rows follow-up filed (the
   invisible-output problem rules it out for #5).
4. **Conventions as affordances + docs**, not prompt presets: folder param,
   id/version returns, provenance header in the tool description; the
   pattern in the user guide.
5. **(Review pass 1)** The seam split reconciled: rows via the legacy
   interop, folders via the scope service with an idempotent ensure-helper;
   both handles wired at both sites; sync→async bridge per #4.
6. **(Review pass 1)** Input bounds are schema-level `maxLength` (mirror
   spec-save's caps; verify-and-match).
7. **(Review pass 1)** Duplicate-window accepted; list-before-rerun is the
   documented convention; no title-keyed upsert.
8. **(Review pass 2)** The provenance header carries `revision:` (span +
   revision = staleness-detectable).
9. **(Review pass 2; resolved on pass 3)** `update_note` **raises
   `ConflictError` on version mismatch** (verified, ChaChaNotes_DB.py:13657)
   — the handler catches it and maps to the named `content_changed`.
   `library_get_note` returns the note's `version` (the payload's
   `revision` field), so the update path is fully servable today.
10. **(Review pass 2)** The handler lives in `local_library_tool_service`
    (the note-backend dispatch), not the media-chunk service.
11. **(Review pass 2)** The flashcards follow-up files at implementation
    close-out.
