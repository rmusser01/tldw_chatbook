# TASK-37145 implementer report

Implementation ready for the root's fresh scoped Superpowers review. The task is
**In Progress**: AC1–3 are locally evidenced; AC4 and successor-head delivery/CI
remain with root. Scoped commit:
`f9d4ad7a297e36aa5615158bfe8eda9aa4f1cbf1` —
`fix(console): keep acknowledgement visible through echo repaint`.

## Scope and ownership

Baseline: `cc907deb187f4e3de99e9f1bab2f73983ac1870a`, branch
`codex/textual-task-37031-37034`. No rebase, fetch, push, merge, CI-enrollment,
workflow, other-task, or root integration-plan edits were performed. App runs
were serial. Root's concurrent integration-plan edit is excluded from staging.

Read actual AGENTS/task, the complete focused CI gap review, relevant design
language/component catalog, Console architecture, TASK-33620.5's Known residue,
TASK-34982, ADR-094 and accepted ADR-230, native testing/profile/timer lessons,
Textual 8.2.8 batch/update source, and SDD implementer/TDD/debugging/Ponytail/
verification instructions. No worker or reviewer agents were dispatched.

ADR required: **no**. Existing
`backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md` and
`backlog/decisions/230-console-send-preparation-and-io-ownership.md` retain
runtime/store/view authority. This repairs the residual physical replacement
gap under TASK-33620.5, preserving TASK-34982's cancellation-safe removal.
Proposed ADR-231's broader polling redesign is excluded.

## Implementation

`ConsoleTranscript.refresh_messages()` now uses native `App.batch_update()`
around its awaited `_reconcile_rows` inside the existing refresh lock. There is
no new lock, await, identity map, row id, owner, timeout or scheduling policy.
The existing remove/mount/reorder/reuse/attachment checks and `finish_removal`
remain untouched. Textual unwinds its batch with `finally` on cancellation.

The new native regression uses production `APP_STYLESHEETS` through
`ConsolidatedCSSApp`, a real `ConsoleSendAcknowledgement`, custody/admission,
the exact real USER echo object/projection, real `set_interval` callback with
`set_messages` and awaited `refresh_messages`, and a held real row Unmount.
The echo projection is checked to already contain only the exact echo before
the handoff. Recorded active-screen non-None `_display` updates retain the
last committed compositor frame; no strip rerender is requested from the
suspended DOM. Eight requested native refresh opportunities exercise the gap.
All holds are released/drained in `finally`, with timer/recorder cleanup.

The cancellation arm stops the actual interval while native removal is held,
then requires an unrelated Static to commit a repaint **before releasing the
native removal**. Both arms settle to exactly the echo id/body once, without
Sending or a cancelled row pump. This checks batch unwind as well as the
unchanged physical-removal contract.

The original core test keeps **`len(gaps) <= 1`**. Its comments now describe
headless compositor commits, and failure-only context includes timestamp,
active-screen/update identity, ack activity, supplied ids, refresh-lock state
and row attachment/regions. No headless terminal-write or elapsed blank-time
claim remains in the recorder description. Architecture and the actual
TASK-37145 incident lesson record the same boundary.

## Native TDD and verification evidence

All pytest runs used only:

```text
C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe -X utf8
  .test-tmp/textual-37031-37034/run-targeted-external-profile.py LABEL SELECTORS...
```

The prepared launcher retained its original 180s timeout and created native700
disposable profiles beneath verified native744 AppData/Local, selecting native
TEMP/TMP/bootstrap before imports. No profile/source guard, ACL, chmod, global
Git config or marker relaxation was used. Native paths required escalation.

| Label | Actual parent result | Qualification |
| --- | --- | --- |
| `ack-gap-cc907-baseline03` | 1 passed, 0 errors/skips; suite45.52s | Existing exact end-to-end node; passing local baseline does not waive CI |
| `ack-37145-red01` | 1 failed, 1 passed, 0 errors/skips;35.00s | Initial widget-consolidated CSS echo RED/cancel control |
| `ack-37145-red02` | 1 failed, 0 errors/skips;20.28s | Initial CSS echo RED, explicit missing-commit failure count |
| **`ack-37145-red03`** | **1 failed, 0 errors/skips;21.02s** | **Authoritative full production CSS product RED:8 missing compositor commits** |
| `ack-37145-green01` | **3 passed plus3 setup errors**,0 skips;55.62s | Both new native arms and exact private-profile timer node pass;3 plain helper setup refusals |
| `ack-37145-controls01` | 7 wrapper nodes passed,0 errors/skips;165.06s | Original unmodified control bodies under equivalent private-profile fixtures |
| **`ack-37145-final02`** | **12 passed,0 errors/skips;90.04s** | **Final regression source, exact timer control, whole original9-node core** |

The root's baseline01/02 native profile-permission failures occurred before
setup and are not product RED. Initial new RED01/02 lacked the full boot feature
stylesheet set; RED03 corrected that test-harness omission **before production
code was edited** and still produced8 missing LayoutUpdate commits. The failing
context names `supplied_ids=['store-echo']`, `ack_active=False`, replacement
phase, the physically attached/pruning old row with zero region, timestamps,
and actual compositor text. Test data is synthetic. Source baseline SHA256 on
disk was `ea68503e08f62659851a34913e4e60a0fd2ef5b53270ad4557a0b683816ab673`.
RED03 test source before formatter/final cleanup:
`6ba835a5b734cfbbc5cd86b5d1df9a3322ae265185e96d378477df811576a98c`.

Final command selectors:

```text
ack-37145-final02
Tests/UI/test_console_transcript_ack_repaint.py
Tests/UI/test_console_transcript_poll_stop_removal.py::test_stopping_the_paint_timer_mid_removal_still_removes_the_row
Tests/UI/test_console_send_acknowledgement.py
```

Final parent XML case times (include that case's fixtures; not whole-suite wall):
new echo16.269s, cancel15.569s, exact timer17.394s, exact original Enter22.476s.
The new file's two cases sum31.838s in this run; the ack file's9 cases sum24.379s.
These are local receipt timings, not successor-head CI or latency certification.

### Actual report paths, not inferred worktree children

Every `verification.json` report path/hash/count was reopened and hash-verified
by `record-ack-37145-evidence.py`. Its complete per-parent/per-child case/timing
receipt is `.test-tmp/textual-37031-37034/ack-37145-verified-receipts.json`.

- Authoritative RED parent:
  `.test-tmp/textual-37031-37034/ack-37145-red03/result.xml`, SHA256
  `f0a82b14fae536a36678838f0803c70b66f26d509a9cc650a37881207130fe44`.
- Authoritative RED native child:
  `C:/Users/GDesktop-1/AppData/Local/codex-textual-test-f7d5ee7808bc4af788360e026faf6e17/pytest/test_acknowledgement_replaceme0/private-profile-process/pytest.xml`,
  SHA256 `a07cd9c6fc990b2ca16c7f8702521b8cfd5d7f7e9e343b109674fa909ae8ff0f`.
- Final parent:
  `.test-tmp/textual-37031-37034/ack-37145-final02/result.xml`, SHA256
  `11a219c7d55fd6c55f94ed1c6b532d95e1e50cd9926ce40cfb9b9da811f9118e`.
- Final echo/cancel/timer children are under actual private root
  `C:/Users/GDesktop-1/AppData/Local/codex-textual-test-416d469fca1f4c25a293df25d1f0fb70/pytest/`,
  respectively `test_acknowledgement_replaceme0`, `test_acknowledgement_replaceme1`,
  `test_stopping_the_paint_timer_0`, each `private-profile-process/pytest.xml`.
  Their SHA256 values are respectively
  `3d4eca404fc44a0bfa99ec806fdf4f5f41f8e6235e21d6b4ab9ce138d8d52627`,
  `02a270464ca0cb9cff79d5c3eeef06e3ee80ab84031a7d991bd8a98afea3fec8`,
  `f635be17ff931a55323467c8eb8f1afd59d36029e4960d3defb72fdd18df995d`.

### Existing controls and setup refusals

GREEN01's3 errors were the plain original nodes:

- `test_finish_removal_outlives_the_cancelled_caller`
- `test_finish_removal_from_the_removed_widgets_own_pump_completes`
- `test_a_removal_that_fails_after_its_caller_is_cancelled_is_logged`

Each failed before its body at UI conftest's app import with
`RecoveryRequired(raw_source_selection_changed)`. This is neither a removal
regression nor a green6-case batch. Their actual original bodies then passed
through temporary `@private_profile_test` wrappers, together with native
steady-state zero moves, reorder, session switch and branch-suffix controls.
Those7 wrapper nodes are not represented as unchanged original-node passes.
Source retained at
`.test-tmp/textual-37031-37034/ack-37145-controls01/wrapper-source.py`; the
temporary `Tests/UI/_ack_37145_native_controls.py` entry file was removed before
commit. Actual7 native child XML paths/hashes appear in that run's receipt and
the verified aggregate. Only the exact original private-profile stopped-timer
node is claimed as an unchanged original-node pass.

Known output warnings: pytest-asyncio's unset fixture-loop-scope deprecation,
existing Pydantic Field-extra deprecation, existing audioop deprecation and
optional audio/dataset availability notices. No test errors/skips in final02.

## Static evidence, caps and self-review

`.test-tmp/textual-37031-37034/ack-37145-static.json` compares actual HEAD
baseline/current AST, Ruff JSON and format results, with normalized source
hashes and current on-disk hashes. All3 Python files parse; `git diff --check`
passes. Both touched test files pass Ruff lint and format checks.

The transcript has the same3 baseline/current Ruff E721 findings at4838,4850,
7181 and the same existing formatter debt. No unrelated cleanup was attempted.
The module-size pin remains8,353; baseline/current source are both8,652
physical lines, already299 over before this task. The source diff is2 additions/
2 deletions; a redundant nearby comment was trimmed for the one-line batch
scope. No guard cap was raised, and a green global ratchet is not claimed.

Final Python source hashes:

| File | On-disk SHA256 | LF-normalized SHA256 |
| --- | --- | --- |
| `tldw_chatbook/Widgets/Console/console_transcript.py` | `4830027463b1f4331324369fe2ca57065d0dfc68866e05f13868714c7c97a027` | `d068c16e02d7d92ebc4e167b02a3f8b7f8893d33eccc55d998ffbee3ac904238` |
| `Tests/UI/test_console_send_acknowledgement.py` | `37bd6a12464fd787a8c1ec87c545925a228867912c096c8bfdb8f1daafadd49d` | `84edd785bf7367d39a6644077080687bd4800cc6e40a929a9633b6edec26900b` |
| `Tests/UI/test_console_transcript_ack_repaint.py` | `663f0cd3918ce5c02c250e1b7e52d07325989b3a5016329a0f47fe3aad803319` | same |

Self-review traced all `refresh_messages`/`_reconcile_rows` callers and preserved
the narrower poll/refresh presentation seam. Other existing direct reconciliation
paths were not redesigned. The native context introduces no task/ownership
replacement and exits on ordinary completion/cancellation. Tests assert the
committed message and native cleanup rather than a mocked batch counter. The
production diff is the minimum native scope, with no code golf or cap changes.

## Files for the scoped commit

1. `tldw_chatbook/Widgets/Console/console_transcript.py`
2. `Tests/UI/test_console_transcript_ack_repaint.py`
3. `Tests/UI/test_console_send_acknowledgement.py`
4. `Docs/Architecture/console.md`
5. `backlog/docs/lessons-testing-evidence.md`
6. `backlog/tasks/task-37145 - Keep-Console-acknowledgement-visible-through-the-store-echo-repaint.md`

The task was updated through native Backlog CLI with local evidence; it remains
In Progress and AC4 stays open. This ignored SDD report and scratch receipts are
handoff artifacts, not members of the implementation commit.

## Root handoff and exclusions

Enroll whole new file; exact new nodes:

```text
Tests/UI/test_console_transcript_ack_repaint.py::test_acknowledgement_replacement_keeps_committed_frame[echo]
Tests/UI/test_console_transcript_ack_repaint.py::test_acknowledgement_replacement_keeps_committed_frame[cancel]
```

Root owns fresh focused review, latest-dev798 integration, census/shard/guard
enrollment, QA persistence, PR publication and actual successor-head CI. The
original CI failure `[5,6,7,8]` against unchanged `<=1` remains valid evidence;
passing an unforced local gate alone never disproved it.

The native batch pauses app repaints while its existing awaited native
reconciliation settles; the cancellation control proves repaint resumption,
not an unprofiled live responsiveness bound. No natural race-frequency,
terminal-write/elapsed-blank-duration, real-provider/server latency, paid-call,
deduplication, durable acceptance, received-intent/full-send or full-suite
qualification is claimed. No full suite was run, per repository/root scope.
