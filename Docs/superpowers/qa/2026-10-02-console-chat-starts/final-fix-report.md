# Final whole-branch fix wave

Status: DONE_WITH_CONCERNS. All four findings repaired and verified, committed as `9cb68f456ee749ea2fc9209ba1edc73f22f15e73`. Inherited fork-fixture/resource qualifications remain below; controller-owned scoped re-review is next.
Worktree: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`
FIX_BASE: `459e666970ef9e7aa4e148705cddafb9621a72d1`
Requirements: `final-fix-brief.md`; completed findings: `final-branch-review.md` and exact probes in `final-branch-review-full.md`.

## Changes

1. `ConsoleChatController` validates the exact new-chat destination through existing persistence/registry owners before approval and immediately before durable creation. Archived and removed destinations fail closed without retargeting; generic persistence and fork behavior are unchanged. `AgentChatStartRequest` captures target workspace identity. The coordinator checks destination availability before preparing and at the ledger acceptance fence, and rejects a changed target workspace. The request uses the approved destination even if the target moves while asynchronous library capture is pending.
2. The existing coordinator checks current runtime enablement, native dispatch eligibility/backend, captured bridge and runtime owner before acceptance. A pre-cutoff change returns Not started, keeps the draft, and settles only the owned uncommitted attempt/reservation through its existing cleanup. Both durable acceptance fences, provider-worker drain, shared capacity, source ownership cutoff and manual withdrawal remain in their existing owners.
3. The approval card describes session remembering as later requests in the same mode and destination, including supplied opening prompts and instructions without another card. Explicit nonblank instructions are labeled as an override and retain their entire body. Existing markup-disabled Statics, buttons and exact request IDs remain intact.
4. The canonical resolver notice travels in the trusted preview payload and through the existing `restore_persisted_session` -> `create_session` notice field/callback. Remembered requests receive the ordinary notice without another card. No notice body is written to budget or handoff metadata. The existing user guide explains body authority and visible Persona fallback.

ADR required: no new ADR. Existing ADR: `backlog/decisions/211-console-chat-destinations-and-bounded-starts.md`. These repairs directly enforce its approved admission and disclosure contracts; no schema, new resolver, authority or lifecycle owner is introduced.

## RED before production edits

Interpreter for every command: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` (P).
Artifact directory: `.superpowers/sdd/2026-10-02-console-chat-destinations-and-starts` (S).
Commands ran from the worktree above, with scoped escalated access for its writes/cache.

```sh
TLDW_TEST_GC_EVERY=1 "$P" -m pytest \
  Tests/Chat/test_console_chat_start.py Tests/Chat/test_chat_create_confirm_card.py \
  -q --tb=short \
  -k 'unavailable_destination_refuses or destination_availability_is_rechecked or runtime_update_during_readiness or missing_persona_notice or mounted_card_discloses' \
  --basetemp="$S/pytest-final-fix-red" > "$S/final-fix-red.log" 2>&1
```

Exit 1: **11 failed, 5 passed, 116 deselected in 31.91s**. Real WorkspaceDB/registry archival remained creatable before/after approval. Archived native destinations started; removed native destinations crossed acceptance then required review. Actual `update_agent_runtime(enabled=False, bridge=existing)` while readiness waited still started. Both ordinary and remembered missing-Persona targets had empty notice fields/callbacks. Mounted draft/start cards lacked explicit-override wording. Successful unchanged-destination/runtime controls reached one actual bridge dispatch through the scripted gateway. Production was unmodified for this RED.

## Focused GREEN and corrections

The same command with `--basetemp="$S/pytest-final-fix-green3" > "$S/final-fix-green3.log" 2>&1` exited 0: **16 passed, 116 deselected in 17.58s**.

The preserved intermediate `final-fix-green.log` exposed two incorrect attribute assumptions: the session has `runtime_backend`, not `remote_active`, and Static has no public `markup` field. Removed the redundant session check and relied on the mounted literal bracket/body assertion for markup behavior. `final-fix-green2.log` then had one test-only timing failure: a finished task's done callback had not run when the task-set assertion executed. Yielding one event-loop turn after gathering the owned task made that cleanup assertion deterministic. No production guard was weakened to fit a fake.

The refusal cases assert the saved draft/state, zero gateway calls, zero generation spend, no saved messages/checkpoint, preserved destination and released exact automatic claim. Missing-Persona cases use the real resolver and ordinary store callback with/without a remembered grant. Card tests mount the actual widget with full multiline bodies and literal markup-like text. Covering runtime/backend/owner/destination tests also pass in the owner run below.

## Static evidence

Before modifying old Python owners:

```sh
"$P" scripts/terminal_qualification/format_ratchet.py snapshot \
  --base 459e666970ef9e7aa4e148705cddafb9621a72d1 \
  --output "$S/final-fix-format.json" \
  --path tldw_chatbook/Chat/console_chat_controller.py \
  --path tldw_chatbook/Chat/console_chat_store.py \
  --path tldw_chatbook/Widgets/Chat_Widgets/chat_create_confirm_card.py \
  --path Tests/Chat/test_chat_create_confirm_card.py
```

`$P $S/final-fix-static.py` records every command/output in `final-fix-static.log`: scoped Ruff E9/F63/F7/F82 for all seven amended Python owners, full format check on the three new Task2 Python files, all five affected old-owner baselines (`task2-format.json`, `task2-fix1-format.json`, `task2-fix2-format.json`, `final-fix-format.json`, `final-fix-format-schema-test.json`), and `git diff --check`. The final run (`final-fix-static-final.log`) passed every check. The schema-test baseline was captured against FIX_BASE before its assertion was changed. Postcommit verification follows below; inherited whole-file formatting was not rewritten.

## Coordination and limitations

Production was frozen after focused GREEN for the controller-owned normal isolated Console/local-provider check described in `final-fix-live-case.md`. The controller confirmed app closure before the final targeted pytest group began. Live evidence is controller-owned and must be read separately from deterministic tests; no live archive/runtime-race or Persona injection is claimed.

The controller owns Backlog task/plan/QA changes. They are excluded from this fix commit. No original checkout edits, full sweep, dependency install, merge, push, PR or task closure is performed. Existing unrelated reasoning_replay lambda baseline failures and prior aggregate FD/escape diagnostics remain qualified by earlier reports; no all-suite or resource-cleanup claim is made. Overlapping test selections will not be summed.


## Controller-owned real Console qualification

The controller reports normal isolated boot11/local-provider verification passed. The actual model's `new_chat` card displayed the complete later same-mode/destination supplied opening prompt/instructions grant explanation and `System prompt (explicit instructions override)` with its full body. The actual **Allow for this session** action created `LIVE_FINAL_APPROVAL` as a casual/global conversation with NULL workspace, pending draft and exact supplied instructions. The target had zero messages, checkpoints and native attempts; the total existing native attempts remained six. The source stayed active. Owned PTY 94565/client 88841 exited 0 before final owner pytest. Captures, receipt and real-config hash are maintained by the controller in the canonical QA directory, not this implementation commit. This does not claim live Persona injection or archive/runtime race coverage.

## Final owner verification

```sh
TLDW_TEST_GC_EVERY=1 "$P" -m pytest \
  Tests/Chat/test_console_chat_start.py \
  Tests/Chat/test_chat_create_confirm_card.py \
  Tests/Chat/test_console_chat_create_confirm.py \
  Tests/Chat/test_console_chat_create_integration.py \
  Tests/Agents/test_agent_chat_create_tools.py \
  Tests/Chat/test_console_chat_fork.py \
  Tests/Chat/test_console_chat_fork_persistence.py \
  Tests/Chat/test_console_chat_store.py \
  Tests/Workspaces/test_workspace_assistant_defaults.py \
  Tests/UI/test_design_token_governance.py \
  -q --tb=short --basetemp="$S/pytest-final-fix-owners" \
  > "$S/final-fix-owners.log" 2>&1
```

The broad targeted owner group returned **905 passed, 4 failed, 1 warning in 355.24s**, exit 1. Two project-instruction fixtures changed the target workspace without establishing a real registry or updating the captured request. They now use real WorkspaceDB/registry bindings and consistent SQL/session/request identity. The old schema assertion expected only the original three fields; it now includes the approved destination/mode fields. The fourth failure is the unrelated inherited fork fixture described below. The broad run warned of **+258 open file descriptors** despite per-test GC; this is not a resource-cleanup pass.

After those fixture corrections:

```sh
TLDW_TEST_GC_EVERY=1 "$P" -m pytest \
  Tests/Chat/test_console_chat_start.py Tests/Agents/test_agent_chat_create_tools.py \
  -q --tb=short -k 'native_project_decision_refuses or new_chat_schema_shape' \
  --basetemp="$S/pytest-final-fix-fixtures" > "$S/final-fix-fixtures.log" 2>&1
TLDW_TEST_GC_EVERY=1 "$P" -m pytest \
  Tests/Chat/test_console_chat_start.py Tests/Chat/test_chat_create_confirm_card.py \
  Tests/Agents/test_agent_chat_create_tools.py -q --tb=short \
  --basetemp="$S/pytest-final-fix-final" > "$S/final-fix-final.log" 2>&1
```

Respectively: **3 passed, 116 deselected in 2.42s** and **140 passed in 80.11s**, both exit 0. The full three-file result preceded the final destination-capture refinement below; its changed path received the focused follow-up. These overlapping counts are not summed.

## Approved destination across library capture

Self-review found an asynchronous capture boundary inside `_start_created_chat`: reading `target.workspace_id` after library capture could silently accept a move made while capture waited. The request now uses `approved["workspace_id"]` (or the global sentinel). This is part of finding 1's exact-destination contract.

```sh
TLDW_TEST_GC_EVERY=1 "$P" -m pytest Tests/Chat/test_console_chat_start.py \
  -q --tb=short -k created_start_keeps_approved_destination \
  --basetemp="$S/pytest-final-fix-capture-red" > "$S/final-fix-capture-red.log" 2>&1
```

Before the one-line production refinement: **1 failed, 115 deselected in 2.32s**, exit 1. The real `_start_created_chat` path returned Started after target/SQL destination changed during held library capture. After the fix, the first selected follow-up was **13 passed, 103 deselected in 11.84s** (`final-fix-capture-green.log`). Adding an unchanged-destination positive control produced the final command:

```sh
TLDW_TEST_GC_EVERY=1 "$P" -m pytest Tests/Chat/test_console_chat_start.py \
  -q --tb=short \
  -k 'created_start_keeps_approved_destination or destination_availability_is_rechecked or preparation_is_frozen or workspace_destination_resolves or creation_without_owner_loop' \
  --basetemp="$S/pytest-final-fix-capture-final" > "$S/final-fix-capture-final.log" 2>&1
```

**14 passed, 103 deselected in 13.91s**, exit 0. A moved target stays pending, spends zero generation allowance and sends no messages; the unchanged control starts with one gateway call. This refinement changed only native destination capture after the controller's boot11; approval-card source remained unchanged.

## Inherited fork fixture: exact baseline evidence

`Tests/Chat/test_console_chat_fork.py::test_configuration_and_leaf_writers_block_fork_through_live_publication[active_leaf]` supplies a blocking `_persist_active_leaf` fake that returns None. The existing store requires a truthy persistence result and raises `RuntimeError("Conversation cursor change was refused.")`. This test and the relevant guard were left unchanged.

The exact FIX_BASE tracked `tldw_chatbook`, `Tests`, and `pyproject.toml` were extracted to `$S/final-fix-baseline`. All **6,615 tracked blobs** were verified against Git object hashes. Reproduction from that extracted directory:

```sh
TLDW_TEST_GC_EVERY=1 "$P" -m pytest \
  Tests/Chat/test_console_chat_fork.py Tests/Agents/test_agent_chat_create_tools.py \
  -q --tb=short -k 'configuration_and_leaf_writers_block or new_chat_schema_shape' \
  --basetemp="<absolute S>/pytest-final-fix-baseline"
```

`final-fix-baseline.log`: **2 failed, 1 passed, 237 deselected in 0.81s**, exit 1. The failures are this same fork fake and the old schema assertion (the latter is repaired in this wave); the system-prompt control passes. Audit artifacts:

- `final-fix-baseline-probe.py`: exact extraction, blob verification and pytest command. The initial archive pipe ended with a setup assertion after extraction, so no result is claimed from that setup attempt. The preserved script uses an archive file; the extracted content was independently verified before the recorded baseline test.
- `final-fix-baseline-manifest.json`: base revision, extraction paths, all file/mode/blob hashes and verified count.
- `final-fix-baseline-manifest.log`: successful `--verify-only` run.
- `final-fix-overall-origin.txt`: controller-preserved AST source comparison confirms both the entire store `set_active_leaf` and entire failing test function are identical between overall BASE and current HEAD.
- `final-fix-overall-base-origin.log`: immutable overall BASE `9ba96ebb626dd010f14d093b0e8d40f37c715d32` source excerpts. The guard already exists at store lines 14013–14014 and the None-returning fake at fork-test lines 1830–1832. This establishes origin before the overall feature branch, beyond the exact FIX_BASE runtime reproduction.

The fork fixture remains deferred with controller agreement. The broad group is not reported as all-green, and no full-suite success is claimed.

## Self-review

Reviewed the fix diff against all four findings and ADR-211. Destination checks use exact captured scope and registry identity without changing generic history/fork persistence. Runtime checks occur again after awaiting draft drain and before AgentRunsDB acceptance, with no await between the final source/target/capacity checks and that cutoff. Existing cleanup owns reservation refunds and physical claim release. The two durable stores and accepted-source independence are untouched. The notice is the canonical resolver's own text, displayed only through trusted preview and the existing store field/callback. Both grant caches, denial ceilings, body caps and request/cancel bindings are untouched. All three native-request test factories supply the newly required captured workspace. There are no new styling literals, keys, settings, dependencies or schema changes.


## Commit and postcommit closure

Commit: **9cb68f456ee749ea2fc9209ba1edc73f22f15e73** (`fix(console): recheck chat start authority and disclose approval scope`). Only the eight owned production/test/guide files were committed, with `git -c gc.auto=0 commit`: 558 insertions, 19 deletions. Controller-owned Backlog task and untracked QA evidence remain outside the commit; the index is clear.

Postcommit command:

```sh
"$P" "$S/final-fix-static.py" HEAD > "$S/final-fix-static-postcommit.log" 2>&1
```

Exit 0. All seven-file scoped Ruff checks, all three new-file full format checks, all five affected formatter ratchets against committed HEAD and `git diff --check` passed. No further production edits followed the final 14-pass capture/admission selection. The complete amended-owner group had 140 passes before that single-line refinement and focused follow-up. The live disclosure check is controller-owned boot11. The inherited fork fixture failure and aggregate +258 FD warning remain qualified, not hidden by the successful selected checks. Exactly one controller-arranged scoped re-review remains; no task closure was performed.
