# Library Source Workbench Stage C QA

Date: 2026-06-25
Backlog: `TASK-136`
Status: verified

## Scope

Library Search/RAG mode was verified as a destination-native retrieval workbench. This pass covers the center workbench and inspector only:

- Query entry and run readiness.
- Source scope rows for All Library, Workspace eligible, Notes, Media, Conversations, Collections, and Import/Export recovery.
- Evidence empty/search/result structure.
- Selected-evidence inspector contract.
- Console handoff blocked/ready state copy.
- Future citation/snippet carry-through placeholders.

## Rendered Evidence

Approved actual CDP/Textual-web screenshot:

- `Docs/superpowers/qa/library-source-workbench-stage-c/library-stage-c-search-rag-cdp-2026-06-24-polish.png`

Captured from `http://127.0.0.1:8981/?fontsize=12` after launching Textual-web from the Stage C worktree and opening Library > Search/RAG.

## UX/HCI Findings Addressed

- The center pane no longer reads like one paragraph. It now has visible Query, Scope, and Evidence sections with terminal rule labels.
- Query blocked state is visible next to the action, including a blocked callout and disabled-run reason.
- Scope is now table-like: `Scope | Count | Eligibility | Next action`.
- Empty evidence state teaches the workflow: add/import sources, run a query, then select evidence for Console.
- The inspector is now a decision panel with Retrieval Status, Console Handoff, Selected Evidence, Recovery, and Future Attribution sections.
- Disabled Console handoff is no longer a stray button; it is paired with a blocked reason and recovery path.

## Verification

Commands run:

```bash
python -m pytest -q Tests/UI/test_library_content_hub.py -k "stage_c_search_rag" --tb=short
python -m pytest -q Tests/UI/test_library_content_hub.py --tb=short
python -m pytest -q Tests/UI/test_product_maturity_phase3_library_contract_layout.py Tests/UI/test_post_ux_product_roadmap_handoff.py --tb=short
git diff --check
```

Observed results:

- Stage C focused mounted regressions: 2 passed.
- Library content hub mounted suite: 21 passed.
- Library contract and roadmap handoff tests: 11 passed.
- Diff hygiene: clean.

## Boundaries

No new `tldw_server` runtime calls, sync promotion, schema/persistence changes, or fake retrieval results were introduced. Stage C remains a Library-native presentation, state, and QA evidence pass over existing local/service seams.
