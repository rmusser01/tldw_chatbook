---
id: TASK-42010
title: "The RAG retained roundtrip test fails at capture preview"
status: Done
assignee: ['@claude']
created_date: 2026-11-19 19:39
updated_date: 2026-11-20 17:05
labels:
- follow-up
- tests
- backup
- rag
priority: low
references:
- Tests/Backup_Recovery/test_rag_retained_roundtrip.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41718 (round 17), red at base. `test_native_rag_retained_roundtrip_with_fresh_passive_reads` fails `assert preview.complete` with `('missing_required',)` before capture runs. It is the only end-to-end RAG roundtrip and runs in the backup-recovery-qualification workflow. TASK-42000 fixed a similar `missing_required` in the pet roundtrip by seeding the required dependencies.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The cause is named and fixed, and the test passes
- [x] #2 Tests/Backup_Recovery/test_mcp_retained_history_roundtrip.py, red for the same `research.local` `missing_required` cause, gets the same setup and passes
<!-- AC:END -->

## Implementation Plan

1. Name the blocking item: a temporary assertion message shows it is `research.local` at the fixture's `custom/research.db`, status `missing_required`.
2. Trace why: the fixture's config explicitly selects `research_db_path`; the research store creates its file on first use (TASK-21105), and nothing in the RAG seed uses Research. `_sqlite_inventory_status` treats an explicitly selected, absent database as `missing_required` by design (it cannot tell never-used from deleted or unmounted), so the product refusal stays.
3. Complete the fixture's setup the way the TTS retained-references sibling does: initialize the lazy research store in the seed child.
4. Run the test through to its readback (red before, green after).

## Implementation Notes

**2026-11-20, coordinator decision:** criterion #2 added. The MCP retained-history roundtrip uses the same fixture config and was red for the same cause, so it gets the same setup. `test_legacy_chat_context_roundtrip.py` fails for another cause and stays out of this task.

**Cause.** The one blocking item is `research.local` at the fixture's `custom/research.db` (`missing_required`, named by a temporary assertion message). The fixture's `config.toml` explicitly selects `research_db_path`, but the RAG seed never uses Research, and since TASK-21105 the research store creates its file on first use. `DB/recovery_operations._sqlite_inventory_status` treats an explicitly selected database that is absent as `missing_required` on purpose: it cannot tell a store that was never used from one that was deleted or sits on an unmounted volume. That refusal protects user data, so the product is unchanged; the fixture's setup was incomplete. The TTS retained-references roundtrip (same fixture shape, same commit) already initializes its lazy store for this reason; this fixture did not. The qualification workflow is `workflow_dispatch` only and has no recorded runs, so nothing caught it.

**Fix.** `Tests/Backup_Recovery/test_rag_retained_roundtrip.py`: the seed child calls `app.local_research_service.list_sessions()` before seeding, as the TTS sibling does.

**Evidence** (`-p no:xdist --timeout=600`, load 17-22).
- Red at base: `assert preview.complete` failed with `('missing_required',)`; the item list was `[('research.local', ..., '<fixture>/profile/custom/research.db', 'missing_required', ('profile:<id>:config',))]`.
- Green: 1 passed in 50.9 s, through capture, isolated restore and the fresh passive readback.
- `ruff check` and `ruff format --check`: clean.

**Criterion #2 (MCP retained-history roundtrip).** `test_mcp_retained_history_roundtrip.py` uses the same fixture config: `research_db_path` is selected and nothing in its seed uses Research. Its seed child now calls `app.local_research_service.list_sessions()` as well.
- Red at base: `assert preview.complete` failed with `('missing_required',)`; a temporary assertion message named `research.local` at `<fixture>/profile/custom/research.db`.
- Green: 1 passed in 55.1 s (`-p no:xdist --timeout=600`, load 34-36). `ruff check` and `ruff format --check` are clean.

**Not in this task:** `test_legacy_chat_context_roundtrip.py` fails earlier, at an assertion on its seed's line 29. That is a different cause, listed as a proposed follow-up in the round report.
