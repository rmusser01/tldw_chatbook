# Final independent whole-branch review

## Scope and method

**Base:** `a5600366381f19d99dbd0dada655113316ede9f7`  
**Head:** `78ff106faca1626faf74bb86029764475568df92`  
**Branch:** `codex/console-chat-starts-dev`  
**Verdict:** **Needs fixes.** Two Important source findings remain. The controller completed the pre-fix live matrix at the reviewed HEAD; repairs still require their own scoped review and final-source live qualification.

This is the required independent broad review of the completed range. I reviewed it myself, with no delegates, test reruns, product probes, app launches, source changes, Git mutations, or publication. The only write is this report. The reviewer template is `/Users/macbook-dev/.agents/skills/requesting-code-review/code-reviewer.md`.

I read the integration plan, including its Global Constraints verbatim, the destinations/starts design, ADR-211, TASK34202/TASK34203/TASK34203.1/TASK34203.2, applicable AGENTS guidance, relevant routing/fork ADRs and testing/live lessons. The production review covered preparation and approval, native acceptance and worker custody, source/child authority, shared allowance accounting, queue/hook/retry/Stop integration, persistence and recovery, both migrations and schema catalogs, UI draft behavior, and recorded baseline repairs. Historical QA was treated as evidence with its original scope, not current test execution.

### Package integrity

The complete package was used, with the navigation supplement only as an aid. Independently checked:

- Complete package: 4,309,438 bytes; SHA-256 `7de5696f65eb9d6ee1fe7b1c0d6be9adce707cfe82749c68deba280ba4a6b497`.
- Navigation supplement: 846,073 bytes; SHA-256 `e4d4f20f08b0c231d2e0df53dd38f61f7c4f5130ba44351a1151cb854eb67676`.
- The complete package’s diff suffix exactly equals `git diff --unified=10 BASE HEAD` for this range. The package contains its separate header/manifest material and binary notices.
- All 68 qualified Python source hashes in the final merge record match both the current files and the reviewed HEAD blobs. The latest upstream merge has no qualified product-source delta. This preserves the recorded command receipts’ source identity; it does not make unrun tests pass.

## Strengths

- Preparation captures the actual destination, fresh destination defaults, resolved routing and source custody before approval. The executor requires its exact live record, saves the destination first, and returns creation and launch outcomes separately.
- Native starts have a dedicated bounded owner and durable acceptance receipt. They preserve machine provenance, avoid manual command/hook authority, and retain physical worker ownership across cancellation. Source authority is checked before acceptance without turning the source into the accepted target’s Stop owner.
- Shared automatic allowance accounting keeps conversation-local records while aggregating reservations and clocks through an immutable canonical root. Admission, settlement, wake and child-launch paths participate in the same accounting boundary.
- The implementation advances AgentRuns 21→22 and ChaChaNotes 75→76 rather than replacing shipped migrations. Native receipt validation includes the existing root-fork and hook-continuation constraints; the prior mixed-replay issue has been repaired before this review.
- The evidence records preserve failed runs, warnings and exclusions. The latest telemetry changes restore actual idle-timer rearming under runtime custody, and the bootstrap fixture repair closes the selected fixture owner. These changes have focused behavioral coverage rather than merely removing assertions.

## Issues

### Critical (Must Fix)

None found in the reviewed range.

### Important (Should Fix)

#### I1. A primary session approval suppresses a later child’s mandatory confirmation

**Primary location:** [console_agent_bridge.py:11226](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/tldw_chatbook/Chat/console_agent_bridge.py:11226).

The bridge skips confirmation whenever `_grant_scope` is in its closure-local `remembered` set. The same injected `new_chat` closure is provided to both primary and child tool catalogs at [agent_service.py:8510](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/tldw_chatbook/Agents/agent_service.py:8510). The prepared grant tuple contains the source incarnation, tool, scope, workspace and mode, but no actor kind ([console_chat_controller.py:19260](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/tldw_chatbook/Chat/console_chat_controller.py:19260)).

Concrete sequence:

1. A primary creates a chat and the user chooses to remember approval for its destination/mode.
2. A genuine child later calls the same closure for that destination/mode.
3. Preparation correctly creates an **unapproved** child record, because controller session grants explicitly exclude children ([controller:19298](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/tldw_chatbook/Chat/console_chat_controller.py:19298)).
4. The bridge’s primary-populated memo skips the child’s confirmation, then execution returns `approval_required` ([controller:19369](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/tldw_chatbook/Chat/console_chat_controller.py:19369)).

This fails closed, but breaks the preserved child `new_chat` capability precisely after a primary session approval. The comment that every child request confirms anew is not enforced by this branch.

**Fix:** Bypass bridge memoization for a trusted prepared child actor, or put the remembering decision entirely behind the controller’s actor-aware approval boundary. Preserve remembered primary approvals and fresh per-request child cards. Add a regression through **one shared bridge closure**: primary remembers, then a genuine child makes two requests and receives two independent confirmations. Existing child tests either start with an empty bridge memo or call controller preparation/confirmation directly, so they do not cover this sequence.

#### I2. Native acceptance does not consume an already-mounted target composer

**Primary location:** [console_chat_controller.py:12779](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/tldw_chatbook/Chat/console_chat_controller.py:12779).

A user can open the created target while its native start is waiting for readiness, leaving the original opening prompt visible without editing it. Session activation does not invalidate the native request’s conversation context epoch or pending v2 handoff. When acceptance subsequently succeeds, `publish_agent_handoff_consumed` retires the persistence writer and marks the handoff consumed, but does not notify or consume the visible composer ([console_chat_store.py:9818](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/tldw_chatbook/Chat/console_chat_store.py:9818)). The common publication step clears only `live_session.draft` by string equality ([controller:13068](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/tldw_chatbook/Chat/console_chat_controller.py:13068)); the submission-accepted UI callback is restricted to manual origin ([controller:13135](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/tldw_chatbook/Chat/console_chat_controller.py:13135)).

The unchanged opening prompt therefore remains in the mounted target composer. Its next same-session synchronization copies that stale text back into the store ([session.py:4857](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/tldw_chatbook/UI/Console_Modules/session.py:4857)); switching away also saves it. The prompt already submitted by the native start is presented again as unsent and can be manually submitted a second time. The durable handoff remains consumed: this finding concerns live composer/store resurrection, not automatic replay after restart.

**Fix:** Publish a target-bound acceptance receipt that consumes the exact mounted composer revision, while preserving the source composer and any later target edits. Do not use an unconditional composer clear or text equality as the ownership check. Add a mounted test that holds native readiness, opens the unedited target, releases readiness, then checks the composer and in-memory draft are empty, durable handoff is consumed, exactly one machine-origin prompt exists, and source state is intact. Include a later-edit survival case. The existing visible Stop test opens the target after native acceptance; the prepared manual-Send test withdraws the native start, so neither covers this ordering.

### Minor (Nice to Have)

#### M1. Test cleanup remains explicitly unqualified

The final four-owner receipt records `611 passed, 2 warnings`, including an unawaited `Timer._run_timer` reported at [chat_screen.py:24890](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/tldw_chatbook/UI/Screens/chat_screen.py:24890) and FD growth of 452 (14→466, threshold 200) reported by [Tests/conftest.py:600](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/Tests/conftest.py:600). The warning’s reporting test/location does not establish allocation ownership or a production regression. Keep these unsuppressed and explicitly deferred; the passing behavioral results do not support a general timer/FD cleanup claim. A future owner-level investigation should establish attribution before changing behavior.

## Plan and spec alignment

The implementation substantially follows the plan and ADR-211: independent draft defaults, workspace/casual destination mapping, complete approval payloads, source custody, saved target before launch, revisioned v2 handoff persistence, native machine-origin acceptance, bounded shared automatic allowance, and migration/recovery integration are present. Current-dev hook, maintenance and provider boundaries have been carried forward rather than replaced. The recorded baseline repairs are explained and have scoped evidence.

I1 is a compatibility gap in the explicitly preserved child approval contract. I2 misses the exact draft-consumption contract when the user opens the target before acceptance. Both are implementation defects within the approved design; fixing them does not require a new architecture decision. The repair-source live gate and publication/task closeout remain controller responsibilities.

## Verification evidence and limits

I inspected saved command JSON/log pairs and checked their recorded output agreement and return codes; I did not rerun them. Counts overlap and must not be summed.

| Evidence | Recorded result | Qualification |
| --- | --- | --- |
| `latest-final-amended-telemetry-owners` | 611 passed, 2 warnings | Four complete amended gateway/egress/cost/spend owners; `TLDW_TEST_GC_EVERY=1`; warnings retained above |
| `latest-final-baseline21` | 21 passed | Exact 21 original baseline nodes |
| `latest-final-bootstrap-create-owners` | 115 passed, 2 skipped | Real-profile bootstrap guard plus full confirmation/integration owners; unchanged unavailable `os.setxattr`/`os.removexattr` platforms are unqualified |
| `review1-amended-owners` | 254 passed, 1 warning | Complete amended repository/v76/tool/confirmation/integration/start owners; FD warning retained |
| `review1-final-create-owners` | 60 passed | Full confirmation/integration owners after prior fix |
| `native-provenance-green` | 129 passed | Recorded provenance scope |
| `routing-recovery-controls-green` | 92 passed | Recorded routing/recovery controls |
| `recovery-core-artifact-green` | 67 passed | Recorded recovery/artifact scope |
| `launch-owner-current` | 11 passed, 1 warning | Complete repaired launch owner; FD warning retained |

The earlier `final-seam-controls` receipt is red (14 failed, 90 passed), and earlier broader groups also contain recorded failures. Later scoped repairs/owner results must be cited to qualify those repaired behaviors; the earlier files remain diagnostic history. The unchanged dev strict XFAIL for the bare-screen timer harness and historical diagnostic archaeology SKIP remain unqualified scenarios. Historical QA files are not substituted for current-dev results.

The controller preflight records successful derived guards. The final source-static evidence is scoped to fatal Ruff rules, compilation, source identity and formatter ratchets; it is not blanket lint/format/security success. No full-suite, OS-power-loss or complete resource-cleanup claim is supported.

### Live evidence inspected at report time

The isolated PTY/provider evidence is under `/private/tmp/console-dev-live-01a0fa6c/evidence/`. I read `boot1-four-modes-receipts.json`, `boot1-edited-cleared-receipts.json`, `boot1-exit.json`, and the boot2 edited/cleared draft captures.

- Workspace and casual drafts each have zero messages and a pending handoff.
- Workspace and casual starts each have exactly one `agent_chat_start` USER message and one real-provider ASSISTANT response; both native attempts are completed, with matching consumed handoff `accepted_attempt_id` and zero remaining checkpoints.
- The casual prompt’s `/settings @everyone` prefix was persisted as literal message content.
- The workspace draft edit at revision 3 and casual clear at revision 2 are recorded durably; boot2 UI captures show the edited text and empty composer respectively.
- Boot1 exited with code 0 at the reviewed HEAD. The recorded real-config hash remains `15c6cb224a6a51c7de5c3f716fbe9dfaef7b7ca05cf9daa7e3ebe247df9310da`.

The controller additionally reports source chat/workspace/composer sentinel preservation and complete approval cards. Before this report was finalized, I inspected `pre-fix-live-qualification.json` and the SDD `controller-live-pre-fix-qualification.json`/`.log` pair: verifier return code 0 and exact stdout/log agreement, qualified at the reviewed HEAD. That record confirms the automatic-disabled refusal leaves a draft without an attempt/provider send; an enabled restart/reopen does not replay it; explicit manual Send then completes with normal provenance. All three app boots exited 0 and their exact app PIDs were absent. The real user config is recorded unchanged. The controller reports its unchanged size/mtime as well as SHA. Initial unsuccessful focus/row-selection automation captures remain preserved and were not counted as successful cases.

The complete pre-fix live matrix is therefore qualified within that record’s stated limits. Held preparation/physical drain is qualified by automated mounted controls, not PTY observation. These live cases do not cover either finding’s specific interleaving. Any repair must receive scoped qualification at its successor source.

## Recommendations

1. Fix I1 and I2 in one bounded wave, with the actual shared closure and mounted preacceptance target ordering covered by regressions.
2. Have the scoped reviewer verify those repairs and their evidence. Preserve all existing admission, source-lifetime, revision, machine-origin and shared-budget boundaries.
3. Carry the complete pre-fix live record forward as historical evidence, qualify the final repair source with bounded mounted controls and actual live successor evidence, and retain the warning, skip and XFAIL qualifications in QA/PR closeout. No broad rerun is requested by this review.

## Assessment

**Ready to merge? No — with fixes and final live closeout.**

The core durable-start, migration and shared-allowance design is implemented with substantial scoped evidence. The primary-to-child approval sequence and already-visible target composer consumption still violate required behavior and should be repaired before publication readiness is claimed.
