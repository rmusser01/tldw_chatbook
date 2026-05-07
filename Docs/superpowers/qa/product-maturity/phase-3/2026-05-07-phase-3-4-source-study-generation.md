# Product Maturity Phase 3.4 Source-Selected Study Generation

Date: 2026-05-07
Task: TASK-10.4
Branch: codex/product-maturity-phase3-4-source-study-generation
Workflow: Library -> Study Dashboard -> Generate Source Study Pack

## What Was Verified

Phase 3.4 verifies the first source-selected Study generation action after the Phase 3.2 source-context handoff.

Verified contract:

- Library-originated Study context carries concrete note and media source items alongside the visible source titles and summary.
- Study Dashboard can launch a server-backed study-pack generation job from the selected Library source items.
- The generation request uses the current server runtime mode and preserves workspace routing when a workspace scope exists.
- Local mode keeps the generation action disabled and explains that server mode is required.
- Conversation records remain visible in the material context but are not sent as study-pack source items until message-level selection exists.

## Automated Evidence

Initial red run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase3_source_study_generation.py -q
```

Result:

- Failed during collection because `StudySourceItem` did not exist.

Focused verification:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase3_source_study_generation.py -q
```

Result:

- `4 passed, 1 warning`.

Adjacent Phase 3 verification:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase3_source_study_generation.py Tests/UI/test_product_maturity_phase3_library_study_context.py Tests/UI/test_product_maturity_phase3_library_contract_layout.py Tests/UI/test_study_dashboard.py -q
```

Result:

- `23 passed, 1 warning`.

Study scope regression verification:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_study_screen.py Tests/UI/test_study_flashcards_screen.py Tests/UI/test_study_quizzes_screen.py -q
```

Result:

- `54 passed, 1 warning`.

## UX Notes

- The Study Dashboard now has an explicit source-pack generation action only when source items are present.
- Server mode gives a direct queued-job path instead of requiring users to reconstruct the source selection in another surface.
- Local mode avoids a false affordance by keeping the action disabled with a server-mode requirement.

## Defects Found

No P0/P1 defects found.

## Residual Risk

- This slice verifies job launch, not completed study-pack polling, review, or generated deck/quiz reuse.
- Conversation source records are still material context only; message-level source selection is needed before they can safely become study-pack `message` source items.
- Workspaces, Collections, Import/Export, and Search/RAG-to-study generation remain later Phase 3 gates.

## Exit Decision

Pass for Phase 3.4. A user can enter Study from Library sources and queue server source-selected study-pack generation with honest local-mode recovery.
