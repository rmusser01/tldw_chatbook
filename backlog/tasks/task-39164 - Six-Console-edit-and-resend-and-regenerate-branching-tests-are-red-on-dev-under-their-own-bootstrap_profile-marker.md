---
id: TASK-39164
title: "Six Console edit-and-resend and regenerate-branching tests are red on dev under their own bootstrap_profile marker"
status: Done
assignee:
- '@claude'
created_date: 2026-11-08 06:54
updated_date: '2026-11-10 18:40'
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/Chat/test_console_edit_resend.py
- Tests/Chat/test_console_regenerate_branching.py
dependencies:
- TASK-39045
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down. TASK-39045's notes (round-12 branch fu/r12-ctests) name three edit-and-resend reds with "causes not read", and the ctlr stream's covering runs found the regenerate ones. Checked on dev b590a15afd and on the round-12 stack 9091820bf2, 2026-11-08 (run.sh, `-n 4`, load 24-35). The same six nodes fail on both trees, plain and under a temporary marker. Both files carry `pytestmark = pytest.mark.bootstrap_profile`, so these failures are their own.

**`Tests/Chat/test_console_edit_resend.py`:**
- `test_edit_and_resend_forks_user_sibling_and_streams_reply`: `RuntimeError: Generation projection persistence is unavailable.`
- `test_edit_and_resend_skill_refusal_leaves_no_pending_node`: `TypeError: …_refuse() takes 1 positional argument but 2 were given`.
- `test_edit_and_resend_carries_anchor_attachments`: `assert False is True`.

**`Tests/Chat/test_console_regenerate_branching.py`:**
- `test_regenerate_persists_new_sibling_when_store_has_persistence`: `RuntimeError: Generation projection persistence is unavailable.`
- `test_regenerate_stream_failure_retains_failed_sibling_and_restores_anchor`: the restored anchor id differs (`assert '…' == '…'`).
- `test_regenerate_mid_conversation_failure_restores_selected_anchor_not_former_tail`: the selected-anchor list differs.

Neither file runs on a PR lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each of the six nodes passes on dev, or is retired with the commit that changed the behaviour it pinned named; each red is shown first with the same command
- [x] #2 Each is classified as test drift or a product defect; a product defect (for example a failed regenerate that restores the wrong anchor) gets a test that is red before its fix
- [x] #3 Both files run on a PR lane (never the fast PR contract), with three timings recorded
<!-- AC:END -->

## Implementation Plan

1. Read each of the six reds' full tracebacks on base ae9f9e8d21 (run.sh, `-n 4`).
2. For each, find the commit that moved the behaviour (`git log -S`, blame on the controller/store seam) and classify drift or product defect.
3. Drift: re-pin the test to the shipped seam (fakes, signatures) without loosening what it proves. Defect: fix the product, keep the test as the red-before evidence.
4. Re-run both files at `-n 4`; time them serially three times for the lane (lane wiring by the round's parent).

## Implementation Notes

**All six reds are test drift; no product change.** Base ae9f9e8d21, `run.sh ... pytest Tests/Chat/test_console_edit_resend.py Tests/Chat/test_console_regenerate_branching.py -n 4` (load ~69): 6 failed / 10 passed, the six named nodes. After, same command: 16 passed.

| Node | Red on base | Moved by | Re-pin |
|---|---|---|---|
| `test_edit_and_resend_skill_refusal_leaves_no_pending_node` | `_refuse() takes 1 positional argument but 2 were given` | b7dd8e53f2 (2026-09-08) passes the turn context to `_apply_skill_substitution` | the stub takes `(provider_messages, _turn_context=None, **_kwargs)`, as the controller suite's stub does |
| `test_edit_and_resend_carries_anchor_attachments` | blocked "vision-model can't accept images" | 3e618d7119 (2026-10-15) computes the captured `capabilities["vision"]` in `console_configuration_capture`, which the resend gate reads (since 37158c8975) | also patch `console_configuration_capture.is_vision_capable` (precedent: `test_console_automatic_library_preparation.py`); the non-vision sibling test patches it too, so it still proves the gate rather than the real capability table |
| `test_edit_and_resend_forks_user_sibling_and_streams_reply`, `test_regenerate_persists_new_sibling_when_store_has_persistence` | `MixinSnapshotUnavailableError: mixin_persistence_unavailable`, then `Generation projection persistence is unavailable` | the narrow create-message fakes were outgrown by 67599acd6e (generation projection writer), 7970fbcada (pre-provider mix-in attempt save) and the branch-mutation transaction, each refusing a fake by design | both drive `ChatPersistenceService` over a real `CharactersRAGDB` file and read the rows back (content, and the sibling's durable parent); the two fake classes are deleted. Adding only the mix-in seam to the fakes (bd9378d02d's pattern) just moved the failure to `db.transaction` |
| `test_regenerate_stream_failure_retains_failed_sibling_and_restores_anchor`, `test_regenerate_mid_conversation_failure_restores_selected_anchor_not_former_tail` | active leaf / path one item longer than `[u1, a1]` | a14733851c (2026-09-02) re-homes the transcript-only provider-failure SYSTEM row beneath the restored anchor (docstring: "original branch becomes active again with the transcript-only failure row appended beneath its anchor") | path is `[u1, a1, failure_row]`, the leaf is that SYSTEM row carrying `result.visible_copy`; the anchor restore of 473e7c9298 (task 571) is still pinned, and the provider payload still ends at the anchor |

The anchor restore is not a defect: the selected anchor (not the former tail) is restored, the failed sibling stays stored, and the failure row is excluded from the provider payload.

**Negative controls** (temporary test edits, reverted): forcing `create_sibling(persist=False)` turns the edit-resend persistence node red (`None is not None`); a no-op `store.set_active_leaf` for the anchor turns the restore node red at the path assert; a no-op `replace_assistant_generation_projection_with_contributions` turns the regenerate persistence node red at `'' == 'hello'` (forcing `persist=False` there does not: the durable dispatch persists the sibling anyway, so the DB-row read is what guards the original regression).

**Lane facts for the parent:** both files are `bootstrap_profile` (module mark), guard-clean under `--storage-lease-guard` (16 passed). Serial `--timeout=180 --tb=short -q`, three runs: 16 passed in 16.1 / 14.9 / 14.7 s pytest, 25 / 25 / 24 s wall, load 74 / 71 / 67 on 18 CPUs.

Files: `Tests/Chat/test_console_edit_resend.py`, `Tests/Chat/test_console_regenerate_branching.py`.

**Lane (AC#3).** Both files join `recovered-suites` after its sandboxed block (both carry a module bootstrap_profile mark), and both are clean under its `--storage-lease-guard`. File timings, serial: 25, 25 and 24 s wall together. `recovered-suites`' whole pytest command, serial through run.sh: 943 s, 1088 s and 1020 s wall, 871 passed and 1 skipped each (1-minute load 32-44). In CI the same step before this round ran 733 passed in 307 s (run 42841528703). This round's additions take about 330 s serial locally at most, so the job should stay near half its 20-minute cap.
