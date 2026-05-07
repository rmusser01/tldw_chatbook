# Product Maturity Phase 3.3 Library Contract Layout Shell

Status: verified

## Scope

Phase 3.3 verifies that the running Library destination exposes the approved Phase 3.0 Library layout contract enough for users to orient and act without knowing legacy route names.

This is a shell layout gate only. It does not implement deeper source-selected generation, full Workspaces management, full Library Collections flows, or new Search/RAG/Import/Export internals.

## Evidence

- Runtime screen: `tldw_chatbook/UI/Screens/library_screen.py`
- Contract spec: `Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md`
- Regression tests: `Tests/UI/test_product_maturity_phase3_library_contract_layout.py`
- Related regressions: `Tests/UI/test_product_maturity_phase3_knowledge_entry.py`, `Tests/UI/test_product_maturity_phase3_library_study_context.py`, `Tests/UI/test_destination_shells.py`
- Backlog task: `TASK-10.3`

## Walkthrough

Mounted Library with local Notes, Media, and Conversations source services.

Verified compact default and large terminal sizes:

- compact: 90x32
- default: 140x42
- large: 180x50

Observed contract regions:

- `#library-status-row`
- `#library-mode-bar`
- `#library-contract-grid`
- `#library-source-browser`
- `#library-source-detail`
- `#library-source-inspector`

Observed visible contract labels:

- Sources
- Search/RAG
- Import/Export
- Workspaces
- Collections
- Study
- Flashcards
- Quizzes
- Source Browser
- Source Detail / Search Results
- Source Inspector
- Authority: local

Observed source snapshot evidence:

- Research Note
- Transcript A
- Planning Chat

## Functional Result

Existing Library actions remain reachable and labelled:

- Open Notes
- Open Media
- Open Conversations
- Import/Export Sources
- Search/RAG
- Study Dashboard
- Flashcards
- Quizzes
- Use in Console

The first implementation attempt placed action buttons in a wide horizontal mode bar. Focused regression replay caught that several buttons were outside the visible screen region at tested terminal sizes. The final layout uses noninteractive mode labels in the mode bar and keeps route/action buttons in vertical browser and inspector regions.

## Verification

Focused verification:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_phase3_library_contract_layout.py Tests/UI/test_product_maturity_phase3_knowledge_entry.py Tests/UI/test_product_maturity_phase3_library_study_context.py Tests/UI/test_destination_shells.py --tb=short
```

Result:

```text
87 passed, 1 warning
```

Warning:

- Existing `RequestsDependencyWarning` for `urllib3` / `chardet` / `charset_normalizer` versions.

## Defects

No P0/P1 defects found after the final layout change.

## Residual Risk

- Search/RAG, Import/Export, Workspaces, Collections, Flashcards, and Quizzes still route into existing or shallow surfaces; their deeper contract implementations remain later Phase 3 child gates.
- Workspaces are visible as scope language but not yet a full Library-native workspace manager in this slice.
- Collections are now Library-owned, but this slice only verifies their visible mode placement. Full collection create/edit/review behavior remains later Phase 3 work.

## Exit Decision

Phase 3.3 Library contract layout shell is verified. Continue Phase 3 with deeper source-selected generation, Search/RAG, Import/Export, Workspaces, Library Collections, and citation/snippet workflow gates.
