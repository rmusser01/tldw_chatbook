### Spec Compliance

- **❌ Issues found.** Steps 1–7 retain most of the required contracts, but the exact native replay path accepts incompatible hook provenance (`tldw_chatbook/Chat/console_dispatch_repository.py:148`), and the prepared primary-only path disables the existing advertised child `new_chat` contract (`tldw_chatbook/Chat/console_chat_controller.py:19053`). Both need fixes before this task passes its independent gate.
- **⚠️ Controller-owned qualification remains pending.** Task brief Step 8 covers latest-dev merge, isolated live destination/mode/start/refusal/reopen verification, task-ID reconciliation, final preflight, QA publication, push and PR attachment. None is claimed complete here, and their pending state is not a Task 1 quality defect. The unchanged TASK32873 timer XFAIL and TASK15743 historical archaeology SKIP do not qualify their excluded behaviors (`task-1-report.md:100`).

Review identity: BASE `ec8eda1d39a5d8ae8ed043b4270da173b95f6652`; HEAD `8fbec5a3a3943003b357e02e619291876e096b90`. This review covers both commits through HEAD, not HEAD~1. Requirements are `task-1-brief.md`, `global-constraints.md`, the approved `Docs/superpowers/specs/2026-10-02-console-chat-destinations-and-starts-design.md`, and ADR-211. All scratch references below are relative to `.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/`.

### Strengths

- **Fresh destination preparation and routing are captured before approval.** `tldw_chatbook/Chat/console_chat_controller.py:19167` resolves destination availability and fresh defaults, captures source lifetime and runtime authority, and scopes the grant by incarnation/tool/resolved destination/mode. `:19100` retains gated routing from destination settings, enabled routed presets and their parameters; it does not apply subagent default routing. The prepared executor at `:19294` persists the captured settings. The actual preparation/approval routing tests preserve the old success/refusal controls in `Tests/Chat/test_console_chat_create_integration.py`, rather than continuing to certify the obsolete raw executor seam.
- **Physical ownership has concrete failure-path coverage.** `tldw_chatbook/Chat/console_chat_start.py:297` retains owned preparation through cancellation; `:337` retains the provider worker and captured owners through final cleanup; `:445` drains draft persistence and checks currentness before synchronous owned ledger acceptance. `:482` verifies the conversation receipt before setting the process-local accepted latch. The held real bridge worker and generic adapter tests (`Tests/Chat/test_console_chat_start.py:1477`, `:2013`) exercise retained capacity through actual worker exit. Maintenance and runtime replacement controls (`:2343`, `:2462`, `:2596`) address the integration's named lifetime risks.
- **Native starts remain separate from queue Stop scheduling.** `Tests/Chat/test_console_chat_start.py:2625` asserts no queue chain, Stop parent or hook receipt after a genuine native start. `:2653` exercises configured v2 initialization success/refusal and a configured Stop proposal. The exact recorded `native-configured-v2-control.log` reports 2 passing cases. The tests check the important absence of scheduling authority through actual owners, not only a mocked dispatch flag.
- **Shared allowance and schema history are preserved structurally.** `tldw_chatbook/DB/automatic_work.py:283` resolves the canonical allowance root, `:362` projects its aggregate balance, and `:683` prepares native attempts against that root. The integration test at `Tests/Chat/test_console_chat_start.py:1686` runs native start, child and wake against the original allowance/deadline. `tldw_chatbook/DB/AgentRuns_DB.py:316` installs schema 22 through the new 21→22 migration; the existing shipped migrations remain. `Tests/DB/test_automatic_work_migration.py:136` checks genuine predecessor rows across constructor/standalone upgrade and reopen. The v76 checkpoint migration retains prior fields and indexes; the dev v75 receipt tests retain older upgrade paths while expecting final version 76.
- **Recovery stays exact.** `tldw_chatbook/DB/recovery_operations.py:1337` derives v22 catalogs from explicitly qualified historical variants. `Tests/Backup_Recovery/test_agent_runs_recovery_schema.py:103`, `:154` and `:202` retain wrong-schema refusal, exact temporary migration authority and real v21 historical-route preservation. The schema allowlist adjustment is limited to a staging table with a later rename in the same SQL fragment; negative controls cover absent, wrong and earlier renames. It does not relax the production allowlist.
- **Draft durability and machine provenance have real SQLite controls.** `tldw_chatbook/Chat/console_chat_store.py:9779` writes revision-bound handoffs with native connection ownership; `:9798` drains their coalescing writer and `:9818` publishes consumption. `Tests/Chat/test_console_chat_start.py:480` checks transactional receipt/consumption, `:546` checks literal slash/@ machine input, and `Tests/DB/test_chachanotes_v76_agent_chat_starts_migration.py:143` rejects changed native content. Those controls are useful, although they miss the mixed-origin replay described below.

### Issues

#### Critical (Must Fix)

None found in the reviewed task scope.

#### Important (Should Fix)

1. **Mixed native/hook provenance bypasses validation on an exact replay.** `tldw_chatbook/Chat/console_dispatch_repository.py:148–170` returns an existing native checkpoint when its native identity, content, provenance and captured settings match. The hook receipt validation at `:188–210` runs only after this return. `_validate_acceptance` at `:1600–1605` rejects root-fork/native and root-fork/hook combinations, but does not reject native/hook together.

   A valid native insert followed by the same acceptance plus a `ContinuationReceipt` therefore succeeds as an exact receipt replay. The first-insert negative test (`Tests/DB/test_chachanotes_v76_agent_chat_starts_migration.py:171`) does not exercise this branch. This breaks task Step 4's requirement that incompatible machine origins refuse before publication and that only an exact retry can recover a native receipt. The demonstrated consequence is an invalid callback being certified with a valid native acceptance receipt. The probe created no additional messages or hook receipt; it does **not** establish an extra provider send or actual hook scheduling.

   Reject native-plus-hook in the common acceptance validator, or validate hook provenance before the existing-checkpoint return. Add a duplicate replay negative beside the existing mixed-origin and exact native dedupe tests. Keep the valid native replay success control.

2. **Existing child `new_chat` is advertised but always refused by primary-only preparation.** `tldw_chatbook/Chat/console_agent_bridge.py:11221–11235` routes every `new_chat` call through preparation and stamps the actual current run as `source_run_id`. `tldw_chatbook/Chat/console_chat_controller.py:19053–19083` requires this identity to equal the live primary run and its stored `agent_kind` to be `primary`. A genuine child run has its own identity/kind and therefore receives `creation_preparation_refused` before a per-request approval can appear.

   This is an existing contract regression, not a request to extend the new start modes to children. The approved spec explicitly keeps child access under existing TASK32531 (`Docs/superpowers/specs/2026-10-02-console-chat-destinations-and-starts-design.md:23`). The unchanged service still discloses both creation tools to children and dispatches the inherited tool closures (`tldw_chatbook/Agents/agent_service.py:751`, `:8509–8510`). The passing schema tests explicitly require child `new_chat` disclosure (`Tests/Agents/test_agent_chat_create_tools.py:136–172`), but child execution/confirmation coverage remains fork-oriented and does not catch this regression.

   Preserve the existing child draft-creation/per-request-approval path with actual child identity and cleanup; keep primary-only destination/start authority appropriately gated. Add an integration control using a genuine child actor, actual preparation/confirmation/execution, successful draft creation and fresh approval on a later child request. Do not resolve this by hiding the existing child tool or bypassing currentness/approval gates.

#### Minor (Nice to Have)

1. **FD growth remains an unresolved verification finding.** `launch-owner-current.log:4` reports growth of 444 descriptors (14→458, limit 200) despite all 11 owner tests passing. The original complete-owner run reports 748 (`original-complete-owners.log:4469`), and the current seam run reports 403 (`current-seam-owners.log:939`). The emitted warning site is `Tests/conftest.py:593`. The report accurately leaves these warnings unsuppressed and claims no causal fix. Nevertheless the output is not pristine and this signal limits confidence in repeated lifecycle use. The present evidence does not identify the leaking owner or prove this patch introduced it, so it is not calibrated as a demonstrated production regression. Controller qualification should retain the warning and assign a bounded cleanup investigation with a baseline comparison; do not raise the limit or infer causality from the child stderr adaptation.
2. **Inherited invalid escape warnings remain visible.** `derived-guards-red.log:403–417` identifies `tldw_chatbook/Utils/Splash_Screens/environmental/train_journey.py:31–32`. They are outside this feature's changed production behavior and do not justify expanding the task, but they must remain reported as warning debt.
3. **New duplicate imports add avoidable cleanup noise.** `tldw_chatbook/UI/Console_Modules/workspace.py:31–36` repeats `asyncio`, `datetime/timezone`, `inspect`, `re` and `time` already imported at `:20–27`. There is no demonstrated runtime consequence. Remove the duplicates while retaining the new `json` import and standard import ordering.

### Evidence Assessment

- **Immutable package checked.** The full package is 4,218,501 bytes with SHA-256 `1c39917e5c1814b9f4991e34909762375f3b7c70519bfbd5feb980296f53ea55`; the navigation supplement is 674,618 bytes with SHA-256 `803addcd93773ea543421a76d2e2732e22de9e370652094d66b736d01a097c2d`. Both match `task-1-review-package-manifest.json`. The supplement was used for navigation and does not replace the full two-commit package. The review proceeded in passes because the package includes large catalogs, historical QA and command ledgers.
- **Historical evidence currentness is limited correctly.** All 230 entries in `historical-qa-current-bytes.json` independently match their actual current file lengths and hashes. This establishes retained bytes against the supplied historical hash manifest. It does not turn old QA runs for source `53065745187aaf4d47fc4357bb9096aa28fd5673` into integrated-HEAD or latest-dev live qualification.
- **Recorded runs were inspected, not rerun.** Literal command records, return codes and actual log outputs were checked. `baseline21-final.log` records all 21 nodes passing. `native-provenance-green.log` records 129 passing cases; `routing-recovery-controls-green.log` records 92; `preview-owner-green2.log` records 16; `schema-rebuild-guard-green2.log` records 47; `recovery-core-artifact-green.log` records 67; `affected-static-guards.log` records 12 selected privacy cases. These overlap and are not summed.
- **Mixed RED/GREEN histories were preserved honestly.** `original-complete-owners.log` is a failing run with 453 passes, 7 launch failures, the unchanged XFAIL and warnings; `launch-owner-current.log` is the repaired complete launch owner. `current-seam-owners.log` is a failing 200-pass/16-fail run; later complete owner selections qualify repaired maintenance/lifetime fixtures. `native-final-owner.log` and `final-seam-controls.log` retain the unconfigured-model, schema-stamp collision and preview-fixture failures; subsequent exact controls and the complete preview owner qualify those repairs. No aggregate claim that every recorded run was green is accepted.
- **Static evidence has its stated bounds.** `committed-head-source-static.json/.log` records 64 Python files matching committed bytes, owned-path equality, absence of obsolete feature migration names, and fatal Ruff checks. Committed-HEAD formatter ratchets record success against the frozen-dev baseline. This is a scoped ratchet, not a claim that inherited files are fully formatted; Black was unavailable and was not installed.
- **Excluded behavior remains excluded.** The strict timer XFAIL is unchanged and its bare-screen attach overlap remains unqualified. The historical archaeology SKIP is unchanged because pinned objects are unavailable. Mounted handoff, accepted Stop and prepared manual Send tests qualify their actual feature behaviors; they do not substitute for that timer race or missing archaeological objects (`task-1-report.md:100–102`).
- **Additional reads were risk-scoped.** A diff hunk cut off the receipt validator, so its surrounding function was read to evaluate native replay validation. The named child compatibility risk required focused unchanged service disclosure/dispatch call sites; they demonstrate the actual closure binding above. No repository audit, app launch, suite rerun, installation, branch/index/source edit or publication occurred.

### Focused Reviewer Check

One real-SQLite probe was run only for the missing mixed-origin replay coverage. CWD was the review worktree; scoped escalation was used because this worktree is outside the writable sandbox. The existing interpreter, root `Tests.conftest` isolation, `TLDW_TEST_GC_EVERY=1`, Python `-B`, and a temporary database under `/private/tmp` were used. No test or production source was written. Command:

```sh
TLDW_TEST_GC_EVERY=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B - <<'PY'
import Tests.conftest
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import replace
from Tests.ChaChaNotesDB.test_console_dispatch_checkpoint_repository import _db_and_conversation, _acceptance, _insert
from tldw_chatbook.Chat.console_dispatch_repository import ConsoleDispatchRepository
from tldw_chatbook.Chat.message_metadata import AgentChatStartMetadata
from tldw_chatbook.Agents.hooks_v2.continuations import ContinuationReceipt
with TemporaryDirectory(prefix='console-review-replay-', dir='/private/tmp') as directory:
    db, conversation = _db_and_conversation(Path(directory) / 'replay.sqlite')
    try:
        native = replace(_acceptance(conversation), origin='agent_chat_start', agent_chat_start_attempt_id='native-start', agent_chat_start=AgentChatStartMetadata('native-start', 'source-run', 'source-conversation'), handoff_draft_revision=1)
        repository = ConsoleDispatchRepository(db)
        receipt = _insert(db, repository, native)
        assert _insert(db, repository, native) == receipt
        mixed = replace(native, continuation_receipt=ContinuationReceipt('parent-turn', 'stop-event', 'parent-assistant', 'scheduler-chain', 1))
        refused = False
        try:
            returned = _insert(db, repository, mixed)
        except ValueError:
            refused = True
        print('EXACT_NATIVE_REPLAY_PASS=True')
        print(f'MIXED_NATIVE_HOOK_REPLAY_REFUSED={refused}')
        print('MESSAGE_COUNT=' + str(len(db.get_messages_for_conversation(conversation))))
        print('HOOK_RECEIPT_COUNT=' + str(db.get_connection().execute('SELECT COUNT(*) FROM console_hook_continuation_receipts').fetchone()[0]))
        assert refused, 'matching native replay accepted incompatible hook continuation metadata'
    finally:
        db.close()
PY
```

Exit code **1**. Decisive output preserved from the tool transcript:

```text
EXACT_NATIVE_REPLAY_PASS=True
MIXED_NATIVE_HOOK_REPLAY_REFUSED=False
MESSAGE_COUNT=2
HOOK_RECEIPT_COUNT=0
AssertionError: matching native replay accepted incompatible hook continuation metadata
```

The tool truncated routine startup/logging output; no separate full probe log was saved. The exact command and decisive result above establish the named failure without rerunning a reported suite.

### Assessment

**Task quality: Needs fixes.**

**Reasoning:** The schema, native ownership, allowance and configured-hook integration have substantial real-owner evidence and preserve the important boundaries. The replay provenance acceptance and advertised child creation regression are concrete missed contracts; both must be repaired and specifically qualified before this task passes, independently of the controller's pending Step 8 work.
