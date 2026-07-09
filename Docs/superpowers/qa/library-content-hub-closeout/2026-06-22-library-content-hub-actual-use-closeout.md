# Library Content Hub Actual-Use QA Closeout

## Scope

Task: `TASK-89.8`

Branch: `codex/library-content-hub-qa-closeout-2`

Runtime: Textual-web at `http://127.0.0.1:8978/?fontsize=12`

Viewport: `2048x1220`

Profile: isolated local QA state under `/private/tmp/tldw-library-qa-89-8`

This pass verifies the current Library destination as a content hub across Content Hub, Search/RAG, Import/Export, Workspaces, Collections, Conversations, Study, Flashcards, and Quizzes. The pass uses actual browser/CDP screenshots. Generated mockups, SVGs, and code layouts are not used as acceptance evidence.

## Visual Evidence

Approval status: approved by the user for the final fixed-state screenshots.

| Area | Evidence |
| --- | --- |
| Initial app state | `Docs/superpowers/qa/library-content-hub-closeout/task-89-8-initial-cdp-2026-06-22.png` |
| Library hub | `Docs/superpowers/qa/library-content-hub-closeout/task-89-8-library-hub-cdp-2026-06-22.png` |
| Search/RAG empty state | `Docs/superpowers/qa/library-content-hub-closeout/task-89-8-library-search-rag-empty-cdp-2026-06-22.png` |
| Search/RAG typed query | `Docs/superpowers/qa/library-content-hub-closeout/task-89-8-library-search-rag-query-typed-cdp-2026-06-22.png` |
| Import/Export | `Docs/superpowers/qa/library-content-hub-closeout/task-89-8-library-import-export-cdp-2026-06-22.png` |
| Workspaces empty/rules | `Docs/superpowers/qa/library-content-hub-closeout/task-89-8-library-workspaces-cdp-2026-06-22.png` |
| Workspaces create action | `Docs/superpowers/qa/library-content-hub-closeout/task-89-8-library-workspaces-create-action-cdp-2026-06-22.png` |
| Collections empty | `Docs/superpowers/qa/library-content-hub-closeout/task-89-8-library-collections-cdp-2026-06-22.png` |
| Collections typed name | `Docs/superpowers/qa/library-content-hub-closeout/task-89-8-library-collections-name-typed-cdp-2026-06-22.png` |
| Collections created item | `Docs/superpowers/qa/library-content-hub-closeout/task-89-8-library-collections-created-cdp-2026-06-22.png` |
| Conversations empty | `Docs/superpowers/qa/library-content-hub-closeout/task-89-8-library-conversations-empty-cdp-2026-06-22.png` |
| Conversations recovery to Console | `Docs/superpowers/qa/library-content-hub-closeout/task-89-8-library-conversations-open-console-cdp-2026-06-22.png` |
| Return to Library | `Docs/superpowers/qa/library-content-hub-closeout/task-89-8-library-return-from-console-cdp-2026-06-22.png` |
| Mode-chip hit-target defect before fix | `Docs/superpowers/qa/library-content-hub-closeout/task-89-8-library-study-empty-valid-cdp-2026-06-22.png` |
| Fixed hub mode chips | `Docs/superpowers/qa/library-content-hub-closeout/task-89-8-library-mode-chip-width-fixed-hub-ready-cdp-2026-06-22.png` |
| Fixed Search/RAG mode | `Docs/superpowers/qa/library-content-hub-closeout/task-89-8-library-mode-chip-width-fixed-search-ready-cdp-2026-06-22.png` |
| Fixed Study mode | `Docs/superpowers/qa/library-content-hub-closeout/task-89-8-library-mode-chip-width-fixed-study-ready-cdp-2026-06-22.png` |
| Fixed Flashcards mode | `Docs/superpowers/qa/library-content-hub-closeout/task-89-8-library-mode-chip-width-fixed-flashcards-ready-cdp-2026-06-22.png` |
| Fixed Quizzes mode | `Docs/superpowers/qa/library-content-hub-closeout/task-89-8-library-mode-chip-width-fixed-quizzes-ready-cdp-2026-06-22.png` |

Historical note: several child task implementation notes reference `Docs/superpowers/qa/product-maturity/screen-qa/library/...`; that directory is not present on current `dev`. This closeout replaces those stale screenshot references with current local CDP evidence.

## Workflow Matrix

| Workflow | Status | Friction / Block | Severity |
| --- | --- | --- | --- |
| Open Library hub from top navigation | Works | Three-column Library fills the browser canvas and exposes module ownership, counts, workspace scope, and disabled Console handoff when no sources exist. | P3 |
| Search/RAG with no Library sources | Works as blocked state | Query input is visible and typing is not obscured. Retrieval is blocked with clear recovery copy because no sources exist in the isolated profile. Seeded retrieval was not re-run in CDP; focused tests cover the panel logic. | P2 |
| Import/Export source entry | Works as handoff surface | Open Ingest/Open Media are visible. Export remains disabled until sources exist. Copy makes the owner boundary clear. | P2 |
| Workspaces scope review | Works | Workspace mode explains that browse/search remains global while staging is workspace-scoped. | P2 |
| Create local workspace | Works | Action creates and switches to `Workspace 1`, with visible status feedback and handoff copy updated to the new workspace. | P2 |
| Collections empty state | Works | Explains local actions, deferred collection-scoped Search/RAG/Study/Console, and server sync WIP. | P2 |
| Create local collection | Works | Name entry is visible, creation succeeds, created collection is selected, and membership/handoff boundaries are visible. | P2 |
| Conversations empty state | Works | Empty state explains saved conversation browser and disables handoff actions until a conversation exists. | P2 |
| Conversations recovery to Console | Works | `Open Console` routes to Console and preserves active workspace context. | P2 |
| Study/Flashcards/Quizzes mode switching | Fixed in this slice | Actual CDP use exposed too-small mode-chip hit targets. Regression added and min width restored. Fixed screenshots verify all three modes switch and show global fallback copy. | P1 fixed |
| Console/RAG handoff from Library | Blocked by no sources | `Use in Console` remains disabled with recovery copy until source or evidence exists. | P2 |
| Collection-scoped handoffs | Deferred | Collection-scoped Search/RAG, Study, Console handoff, and server sync promotion are explicitly labeled WIP/deferred. | P3 |

## Nielsen Norman Heuristic Review

| Heuristic | Finding |
| --- | --- |
| Visibility of system status | Strong for empty/blocked states. Status rows and inspector copy show `Empty`, `Blocked`, `Ready`, workspace, and local/server scope. |
| Match between system and real world | Workspaces copy now correctly states that browsing/search remain global while staging/manipulation is workspace-scoped. |
| User control and freedom | Top nav, mode strip, and left action rail provide recovery. The fixed mode-chip width improves direct click recovery between modes. |
| Consistency and standards | Three-column grammar is consistent with other redesigned screens. The mode strip needed a hit-target correction to match visible affordance expectations. |
| Error prevention | Disabled actions prevent invalid Console/RAG/Study handoffs when prerequisites are missing. |
| Recognition rather than recall | Mode labels, owner copy, and inspector copy keep the product model visible. Dense text remains a lower-priority readability risk. |
| Flexibility and efficiency | Keyboard hints exist for Search/RAG. Full keyboard traversal across every Library submode was not exhaustively re-verified in this closeout. |
| Aesthetic and minimalist design | Layout is functional and aligned with the terminal workbench system. Inspector text is still dense in some modes. |
| Error recovery | Empty and blocked states generally include a next action. Collection-scoped handoff copy correctly says WIP rather than implying hidden functionality. |
| Help and documentation | In-product help is usable. A Library-specific CDP runbook does not exist; this document provides the current QA recipe and evidence set. |

## Defect Found And Fixed

### Library mode chips had fragile click targets

Evidence: in actual CDP use, clicking visible Study/Flashcards/Quizzes labels did not reliably switch modes before the fix.

Root cause: `Button.library-mode-chip` and `Button.notes-mode-chip` declared `min-width: 0` even though `$ds-library-mode-chip-min-width: 10` already existed. Short labels such as `Study` rendered with tiny practical hit targets.

Fix:

- `tldw_chatbook/css/components/_agentic_terminal.tcss` now uses `$ds-library-mode-chip-min-width`.
- `tldw_chatbook/UI/Screens/library_screen.py` fallback CSS now uses `min-width: 10`.
- `tldw_chatbook/css/tldw_cli_modular.tcss` was regenerated via `tldw_chatbook/css/build_css.py`.
- Regression added: `Tests/UI/test_product_maturity_phase3_library_contract_layout.py::test_library_mode_chips_keep_minimum_click_target_width`.

## Residual Risks

- User visual approval is still required for the final fixed-state screenshots before the Library UI is accepted.
- Actual seeded Search/RAG retrieval with indexed sources was not re-run in CDP in this isolated empty profile. Existing focused tests cover the Search/RAG panel and handoff behavior.
- Collection-scoped Search/RAG, Study, Console handoff, and server sync promotion are intentionally deferred and should not be treated as broken if disabled.
- The previous child-task screenshot directory is absent on current `dev`; future QA should use the closeout directory or recreate a durable Library evidence protocol.
- Inspector copy is accurate but dense in several modes. This is a P3 readability issue, not a blocker for basic use.

## Verification

Red test before fix:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_phase3_library_contract_layout.py::test_library_mode_chips_keep_minimum_click_target_width --tb=short
```

Result before fix: failed for `#library-mode-study` with `Region(... width=9 ...)`.

Green test after fix:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_phase3_library_contract_layout.py::test_library_mode_chips_keep_minimum_click_target_width --tb=short
```

Result after fix: `1 passed, 1 warning`.

Focused epic verification:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_library_content_hub.py Tests/UI/test_product_maturity_phase3_library_contract_layout.py Tests/UI/test_product_maturity_gate16_library_search_rag.py Tests/UI/test_product_maturity_phase3_library_study_context.py Tests/UI/test_post_release_workspaces_library_depth.py Tests/UI/test_destination_visual_parity_correction.py Tests/QA/test_agentic_terminal_css_tokens.py -k 'library or agentic_terminal' --tb=short
```

Result: `73 passed, 70 deselected, 8 warnings in 75.21s`.

Diff hygiene:

```bash
git diff --check
```

Result: passed with no output.
