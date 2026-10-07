from pathlib import Path
import hashlib,json,re,subprocess
scratch=Path('.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration')
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert head=='df81b3e65c2c2a0b221c85962658acc96fc567d2'
assert not subprocess.check_output(['git','status','--porcelain'])
report=scratch/'task-1-report.md'
assert '## Task 1 independent-review fix round 1' not in report.read_text()
records=[]
for path in sorted(scratch.glob('review1-*.json')):
 data=json.loads(path.read_text())
 if isinstance(data,dict) and 'argv' in data and 'returncode' in data:records.append((path,data))
paths=json.loads((scratch/'review1-owned-source-hashes.json').read_text())
for item in paths:assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
text='''

## Task 1 independent-review fix round 1

**Status:** DONE_WITH_CONCERNS. Reviewed source was `8fbec5a3a3943003b357e02e619291876e096b90`; parent documentation commit before this fix was `384658dcb15b219cb57676e9d5baf5d323050d11`; owned fix commit is `df81b3e65c2c2a0b221c85962658acc96fc567d2` (`Reject mixed native replay and preserve child chat drafts`). Original BASE remains `ec8eda1d39a5d8ae8ed043b4270da173b95f6652`; original reviewed branch remains `53065745187aaf4d47fc4357bb9096aa28fd5673`. Tracked working tree is clean. Exactly eight owned paths changed. No helper/reviewer, app, full sweep, root-doc edit, merge, push or PR occurred.

### Disposition of independent findings

1. Common `_validate_acceptance` now rejects native plus `ContinuationReceipt` before checkpoint replay returns. Both actual SQLite regression owners retain valid exact replay, then attempt the otherwise-identical mixed replay and require typed refusal. They prove the original checkpoint and message rows stay exact, two messages remain, one checkpoint remains, and zero hook receipts are inserted. This validates refusal; it does not claim an extra provider send or hook scheduling happened in RED.
2. TASK32531 child `new_chat` remains advertised and now completes genuine prepared draft creation with an actual `CurrentRunActor("subagent", child, parent)` and native AgentRuns parent/child rows. Preparation derives kind/parent from that trusted actor and rechecks the durable row's kind, running status, parent identity and conversation against the source session incarnation/workspace. Child requests only retain same_workspace/draft; casual destination and start are refused. Fresh destination defaults/routing and exact prepared approval remain shared owners. The real controller card discloses the child identity and parent; creation persists the exact draft/source child id without transcript messages, native start attempts, source focus movement or composer loss.
3. Child approvals always return `remember=False`, do not create standing grants, do not ride an existing primary grant and cannot enter closure memoization. Two actual closure invocations obtain two distinct approval ids. A completed parent and absent primary slot still permit the surviving child; an unrelated next primary turn's Stop is excluded from its approval binding. The original parent turn's cancel Event is captured in the prepared record and continues to refuse execution after its active session slots disappear.
4. The existing `_revoke_chat_create_rounds` owner now invalidates matching prepared records and stores its run revocation fence. Preparation checks this fence under the same record lock; approval registration checks token/record identity and the fence under that lock. Actual controls cover deny, revoke during confirmation, revoke after approval while the native run row still reads running, revoke between validation and registration, and later preparation. No stale card is published at the registration race, and no token/round survives refusal. Terminal child, changed actor parent and changed source incarnation also refuse execution.
5. Removed duplicate asyncio, datetime/timezone, inspect, re and time imports from workspace.py; retained json in the standard-library block. No runtime behavior was added for this cleanup.

### RED/GREEN and mutation sensitivity

- `review1-red`: **7 failed, 2 passed, 110 deselected**. Both mixed replay tests failed `DID NOT RAISE`; the real child closure failed `creation_preparation_refused`; genuine preparation/currentness/confirmation cases failed `source_unavailable`. The two child casual/start refusal cases already passed. This is the expected pre-fix behavior, not fixture errors.
- `review1-focused-green`: **9 passed, 110 deselected** after the initial fixes.
- `review1-revoke-red`: **1 failed**. Revoke after approval left a prepared token executable while the actual child row remained running. `review1-child-cleanup-green`: **11 passed, 46 deselected** after cancellation-owner repair, including survivor and standing-grant controls.
- `review1-amended-owners`: **254 passed, 1 warning**, 207.29s, complete receipt repository, v76 migration, creation tool schema/disclosure, confirmation, integration and native-start modules. These overlapping owners are not summed with the later run. This run used the initial child cleanup fix; only formatting changed during it. The final three child cancellation controls below were added and repaired afterward.
- `review1-final-cancel-red`: **3 failed**. Revocation between validation and registration still published a card; stopped original-parent preparation did not raise; the captured parent Stop was lost after session slot removal and execution succeeded. The repaired complete confirmation/integration owners in `review1-final-create-owners` report **60 passed**, 43.62s, no warnings or exclusions. This is the amended-child-owner evidence after those final behavior changes. A final formatting-only correction to the newly added confirmation block followed; there were no further behavior changes.
- The tests catch removal of common native/hook validation even with an existing valid checkpoint, substitution of primary identity for the actual child actor, acceptance of stale native child rows/parent bindings/incarnations, remembered child grants or closure memoization, accidental child start/destination authority, forgotten token cleanup/revocation fences, captured Stop loss and late card registration. These are observable mutation sensitivities, not a claim that every hypothetical mutation was executed.

### Static qualification and limits

`review1-fatal-ruff-final` and committed-source qualification pass `E9,F63,F7,F82`; this is fatal syntax/name checking, not every Ruff style rule. A missing conjunction in the final cancellation edit was caught by Ruff parsing (`review1-format-final`, rc1) before any covering test ran; it was corrected and the final complete creation owners passed. Changed ranges were formatted, preserving inherited formatter debt. Ratchet failures and precise adjacent corrections are retained; `review1-format-ratchet-qualified` and `review1-committed-format` pass against the frozen-dev baseline. Existing Black unavailability is unchanged; no install was attempted.

`review1-committed-source-static` verifies all64 previously owned Python sources equal immutable HEAD bytes and pass fatal Ruff; every owned stage path matches HEAD and all three retired unshipped migration names remain absent. Detailed literal inner argv/results/source hashes are in `review1-committed-head-detail.json`; executable source is `review1_committed_static.py`. `review1-committed-whitespace` passes the exact `384658d..df81b3e` range. `review1-owned-source-hashes.json` holds the eight pre-commit/current matching hashes. Final working tree is clean.

### Concerns and exclusions

The amended254-owner run retains the unsuppressed warning: open descriptors grew272 (start14,end286,limit200), emitted from Tests/conftest.py:593. No causal attribution, limit increase, warning suppression or FD cleanup is claimed. Earlier focused runs also retain pytest rm_rf warnings while its default basetemp attempted cleanup of old unrelated pytest garbage; subsequent checks use a fresh owned /private/tmp basetemp rather than changing warning policy or touching unrelated processes. No unrelated process was inspected in detail or stopped.

The unchanged TASK32873 strict timer XFAIL and TASK15743 historical archaeology SKIP remain exactly as previously reported and unqualified; this round's selections contain no skip/xfail. Original all21 qualification was not rerun in this fix round. Parent owns latest-dev integration (now including test-profile guard and telemetry changes), current all21/live/final review/publication qualification and the previously recorded backlog collision. No latest-dev behavior is claimed qualified here.

### Automatic approval usage interruption

The first exact `review1-amended-owners` tool request was rejected before execution with: “Automatic approval review failed: You’ve hit your usage limit. Visit https://chatgpt.com/codex/settings/usage to purchase more credits or try again at Oct 9th, 2026 2:13 PM. The action was not executed because automatic approval review could not be completed. This is a review failure, not a determination that the action is unsafe. Do not bypass the approval check; resolve the error or ask the user for guidance.” There is no fabricated subprocess receipt for that unexecuted action. After the user instructed continuation, the identical command went through normal require_escalated review and produced `review1-amended-owners.json/log`; no bypass or duplicate run was used.

### Owned paths and final source hashes

'''
for item in paths:text+=f"- `{item['path']}` — `{item['sha256']}`\n"
text+='\n### Literal command receipts\n\nEach JSON below records literal argv, cwd, selected environment, return code, elapsed time and full combined stdout/stderr. Its matching .log preserves untruncated output. The excerpts are copied output; full failure startup/database logs remain in the referenced files. All managed writes/checks/commit used require_escalated.\n'
for path,data in records:
 text+=f"\n#### {path.stem}\n\nRecord: `{path.name}`; full output: `{path.with_suffix('.log').name}`.\n\n```json\n"+json.dumps({k:data[k] for k in ('argv','cwd','env','returncode','elapsed_s')},indent=2)+'\n```\n'
 output=data['output'];lines=output.splitlines()
 chosen=[l for l in lines if re.match(r'^(E\s|FAILED |PASSED |XFAIL |SKIP |\d+ (?:failed|passed)|Recorded:|All checks|HEAD|Formatted|error:|AssertionError:|Tests/.*formatter|required)',l) or 'open file descriptors grew' in l]
 if not chosen and len(output)<4000:chosen=lines
 if chosen:text+='\n```text\n'+'\n'.join(chosen[-28:])+'\n```\n'
 else:text+='\nOutput is retained in the full log (formatter-difference evidence or silent success).\n'
text+='\n### Preserved executable probe/formatter sources\n\nThe direct scratch copies below preserve the actual temporary helper source used in this round. They include the pre-correction final-production edit that produced the recorded parse error. Formatter/validation/commit argv are preserved in the records above; no historical receipt was overwritten. The inline adjacent formatter command uses `python -` and its actual per-range argv/env/output are preserved in `review1-format-adjacent-detail.json`.\n\n'
for p in sorted(scratch.glob('source-review1-*.py')):text+=f'- `{p.name}` — `{hashlib.sha256(p.read_bytes()).hexdigest()}`\n'
report.write_text(report.read_text()+text)
(scratch/'source-review1-console-task1-round1-report.py').write_bytes(Path(__file__).read_bytes())
print(f'Appended review fix report: HEAD{head}, {len(paths)} paths, {len(records)} literal command receipts. Working tree clean.')
