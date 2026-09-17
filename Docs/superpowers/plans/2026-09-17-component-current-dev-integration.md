# Current-dev component integration plan

> **For agentic workers:** Use executing-plans inline, with independent code review after resolution. Steps use checkboxes for evidence tracking.

**Goal:** Reconcile the reviewed component branch with dev `1c0327b3bb3d95b61e3e1b9a83b30e7030453ad6` and save the verified candidate to draft PR #2704.

**Architecture:** Preserve existing runtime ownership and the ADR-161 source-sheet decomposition. Resolve behavioral conflicts from both parents, then regenerate derived files from their owners.

**Tech stack:** Python 3.12, Textual 8, SQLite, TCSS, pytest.

**Spec:** `Docs/superpowers/specs/2026-09-13-component-pattern-library-design.md`; remaining integration gate in `Docs/superpowers/reports/2026-09-17-design-system-completion-audit.md`; TASK-32749.

ADR required: no
ADR path: existing `backlog/decisions/150-design-token-system-and-design-language.md` and `backlog/decisions/161-component-pattern-library.md`
Reason: reconcile existing contracts without new storage, security or application boundaries. Reassess if resolution actually requires such a change.

## Constraints

- Work only in `.worktrees/component-pattern-library`, branch `feat/component-pattern-library`; leave unrelated `tmp/` alone.
- Zero source dimension literals and ad hoc Python visual values in the approved scope; no raised ownership/literal/boot ratchets.
- Generated styles come from `css/build_css.py`, never hand resolution of generated content.
- Targeted verification only. No full suite or merge into dev.
- Read both parent versions for semantic conflicts; no blanket production ours/theirs resolution.

## One integration unit

Consumes: saved head `9c2edfd67f2ade2df43a90b00ef82ee40d828a91` and the recorded dev commit. Produces: a two-parent candidate retaining both branches' behavior, review evidence and generated artifacts.

- [ ] Record task-ID allocation, parents and conflict inventory; commit this plan/task before beginning the merge.
- [ ] Run `git merge --no-commit --no-ff 1c0327b3bb3d95b61e3e1b9a83b30e7030453ad6`.
- [ ] Resolve `chat_screen.py`, `console_bounded_section.py`, `library_file_notes_workspace.py` and their conflicting tests, preserving sidebar scheduling/covered-screen reconciliation, reviewed focus behavior and incoming File Notes recovery/path behavior. Compare merged functions against both parents.
- [ ] Resolve the import/export guide and lesson additions by retaining compatible claims from both sides. Rebuild the diagnostic inventory from the resolved production sources after inspecting its diff.
- [ ] Transplant incoming `_agentic_terminal.tcss` changes into the decomposed owning files. Compare incoming changed declarations after token expansion; tokenize new fixed values using existing tokens or explicit additions if required.
- [ ] Run `.venv/bin/python tldw_chatbook/css/build_css.py`. Check governance/build/budget/Backlog tests, including `test_design_token_governance.py`, `test_component_pattern_governance.py`, `test_python_style_inventory.py`, `test_css_build_integrity.py`, `test_css_bundle_sync_guard.py`, `test_boot_css_byte_budget.py` and `test_backlog_task_id_uniqueness.py`.
- [ ] Run targeted conflicting/affected journeys: sidebar-state debounce, live-work handoffs, bounded sections, File Notes layout/recovery, the incoming picker behavior and existing compact picker/Settings journeys. Narrow existing large modules by the changed symbols; record exact selections. Reproduce failures before repair and add regression coverage where the merge creates an uncovered interaction.
- [ ] Run `PYTHON=.venv/bin/python ./scripts/preflight.sh` and fatal Ruff across changed Python files. Use an existing verified Mermaid input cache or permitted hash-pinned download. Inspect every failed check rather than regenerating blindly.
- [ ] Review the integration diff independently. Resolve findings, rerun affected checks, verify Backlog ID uniqueness after integration and scan the final candidate for conflict markers.
- [ ] Run a private-profile real terminal app with production CSS. Exercise representative integrated Console/Library navigation, dark/light wide/compact paint and clean shutdown. Record exact hashes, capture inspection, private DB checks, default-file isolation and process absence. These checks do not claim external generation or whole-feature qualification.
- [ ] Update integration report and full completion ledger, close TASK-32749 only after its gates pass, commit the merge, push the existing draft PR and verify its base/head/state. Keep remaining feature review work active.
