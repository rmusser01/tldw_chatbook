---
id: TASK-18941
title: 'XML_Ingestion dead module: second broken import (add_media_to_database) — fix, rewire, or delete'
status: Done
assignee: []
created_date: '2026-08-19 20:30'
updated_date: '2026-08-19 20:30'
labels:
  - cleanup
  - ingestion
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during Chunking Parity Phase A (task-18905, Task 5 characterization): `Local_Ingestion/XML_Ingestion.py:13` imports `add_media_to_database` from `Client_Media_DB_v2` — a name that module has never exported (verified via `git log --all -S` at the branch merge-base; zero occurrences ever). The chunking-parity work restored the module's OTHER broken import (`chunk_xml` — spec §7.1, fixed by the Chunk_Lib shim), but this second import keeps the module unimportable. Nothing in the tree imports `XML_Ingestion`, so it is dead code that fails at import.

Decision needed: (a) fix the import to the current DB API (`add_media_with_keywords` or the appropriate replacement), (b) rewire the module to the modern ingestion path, or (c) delete the dead module. Spec `2026-08-18-chunking-engine-parity-design.md` §11 already defers the "XML ingestion reachability" product question — this task is the code-side answer to the same question.

The characterization test `Tests/Chunking/test_callsite_characterization.py::test_xml_ingestion_import` carries `xfail(strict=False)` with this exact reason — the fix self-announces via XPASS.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A decision is recorded (fix / rewire / delete) with rationale in this task's Implementation Notes
- [ ] #2 If fix/rewire: `import tldw_chatbook.Local_Ingestion.XML_Ingestion` succeeds and `test_xml_ingestion_import` XPASSes (mark then removed)
- [x] #3 If delete: the module, its test xfail (converted to assert-not-importable or removed), and any references are removed together
<!-- AC:END -->

## Closeout audit 2026-10-11

Board-prune audit (branch chore/board-prune-c, base a3c47c6c72): completed-but-unmarked. Evidence gathered 2026-10-11 against origin/dev:

AC dispositions:
- AC#1: met - Decision: DELETE (dead module, zero importers) - executed by commit dce252f841 'chore(local-ingestion): delete XML_Ingestion and API_Endpoint_Sample (TASK-32899)'.
- AC#3: met - Module and its xfail characterization test removed together (no test_xml_ingestion_import remains under Tests/Chunking/); Chunking/auto_selection.py:135 keeps only an explanatory 'retired' comment.
- Commit dce252f841 (TASK-32899 follow-up) deleted the module; the repo-wide grep for XML_Ingestion finds only the retirement comment.
