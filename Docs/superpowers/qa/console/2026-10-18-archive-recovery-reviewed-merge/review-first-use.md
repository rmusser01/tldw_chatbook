# Scoped startup and first-use review addendum — PR3162

Date: 2026-10-18. Same single read-only reviewer, applying the requesting-code-review reviewer contract and Ponytail full. This reviews the startup repair and affected lifecycle after the published head's hosted Perf Guard failure; it does not duplicate or replace the full feature review.

## Reviewed identity

- Feature base: `5eb9a37b34a1b0af3e91ad360c0b997497ee7894`.
- Follow-up review base/published head: `566fcbe6ea56629fa8aedce64494a119cb40b327`.
- Frozen full feature source: **100 files**, SHA256 `eec2090dd0ec897f1f9742badd96f6f16b9c30076a3c65c959d87382a1041511`.
- Follow-up source delta from the published head: **11 files**, SHA256 `b4ab5f52c2645e38e2367ea4c90e1b686260d26544bcfcf293deeb16bc887579`.
- The reviewer independently reproduced both hashes. The snapshot algorithm is the original report's sorted `path + NUL + file-SHA256 + newline` algorithm, restricted to changed `.py`, `.tcss`, and `.tsv` files under `tldw_chatbook/`, `Tests/`, and `scripts/`. The follow-up uses the published head as its diff base. The eventual immutable commit must reproduce these hashes before publication.
- Original `review.md` remains unchanged, SHA256 `6f100af0e9ab7931ecce3dc28fa8716e4fe6f61d3b4a5e63f6602e91cdcc38d1`.

Scope includes the seven changed production files, four changed tests, and connected boot, received-input custody, queued-input proof production, hook binding/clearing, mounted controls and native dispatch callers. No source, index, branch, HEAD or historical artifact was changed by the reviewer. Only review artifacts and isolated temporary probes were written; no new agent, live/model request or full-suite run was used.

## Strengths

- The repair follows Textual's actual attachment precondition and rechecks it across awaited initialization. It neither suppresses live exceptions globally nor treats a latched mount flag as proof of an attached control.
- The import repair retains reserved names in the already resident catalog while deferring the concrete provider and recovery graph. Exact provider type, app-issued authority, persona/Library gates and final answerability remain mandatory.
- First-use preparation happens after synchronous received-input capture and app-owned custody, for manual and genuine queued turns, followed by custody revalidation. The new async preparation slot participates in the existing attach/detach inventory. Per-view serialization covers its concurrency with ordinary native sync; no extra timer or worker was added.
- Lazy creation wires the existing controller, including the user-only queue/edit proof producer. A failed optional projection leaves recovery admission closed and does not manufacture proof, acknowledgement or confirmation.

## Issues

### Critical

None found in the scoped delta.

### Important — all addressed

1. **Queryable roots could be detached during startup teardown.** `tldw_chatbook/UI/Console_Modules/archive_recovery.py:128`. Hosted Perf Guard failed its profile-warming boot at `root.mount(surface)`; its printed first-boot count of 1086 was not the warm census. Installed Textual checks `is_attached`, while `is_mounted` remains true after removal. The initial mount-only guard therefore did not close the hosted failure. The final guard checks both states. The pending-on-mount test separately verifies conservative initialization after full mounting; Textual itself permits child mounting during an attached parent's `on_mount`.

2. **Exit across the awaited mount still allowed later child mounting and false answerability.** `tldw_chatbook/UI/Console_Modules/archive_recovery.py:134`, `:147`, `:100`. The reviewer independently reproduced exit immediately after the first awaited mount causing a second MountError on the controls surface. A separate real mounted-control probe returned answerable=True while the screen was detached. Post-await attachment checks and actual attachment requirements for the screen, card and decision controls now close both paths. The unchanged probes subsequently returned no mount error and answerable=False during exit. Real mounted regressions cover entry-time exit, exit across mount, pending initialization and two gated concurrent sync calls.

3. **Recovery added twelve definite first-paint residents.** `tldw_chatbook/Agents/tool_catalog.py:100`, `tldw_chatbook/Chat/console_runtime.py:2907`, `tldw_chatbook/UI/Screens/chat_screen.py:9326`, `tldw_chatbook/UI/Console_Modules/prompt_queue.py:935`. Same-probe immutable-dev/current measurements after the mount guard were 1030/1043; the implementer's separate current run measured 1042. Twelve added modules belonged to recovery; the thirteenth in the reviewer's run was an existing character-search resident. The final change defers the recovery graph through lazy view hooks, controller/runtime wiring and the existing post-ready native sync. An explicit post-custody preparation hook guarantees first-send controls without awaiting before input capture; the user-only queue/edit fallback uses the actual app owner. New absence pins cover all twelve modules. The final isolated current guard passed at 1031 with the existing 1031 limit unchanged.

### Minor

No unresolved scoped finding.

## Evidence and limits

Raw evidence is in `/private/tmp/archive-recovery-pr3162-merge-20261018`.

- `perf-117820112656.log` and `boot-census.log` retain the hosted and mount-only-guard failures. `mount-exit-red.log`, `mount-yield-red.log`, `mount-concurrent-red.log`, `cold-view-red.log` and `queue-cold-red.log` retain the focused controls. The early `mount-retired-red.log` failed on an incomplete fake rather than the hosted error; it is not counted as a valid MountError control.
- Reviewer-owned real Textual probes: `reviewer-mount-yield-probe.log` → `reviewer-mount-yield-green.log`; `reviewer-answerability-exit-probe.log` → `reviewer-answerability-exit-green.log`. Both unchanged GREEN probes exited zero after the fixes.
- `reviewer-boot-attribution.json` and retained complete child outputs identify the twelve new recovery residents. Three additional serial warm boots of immutable dev on the same warmed isolated profile measured **1030, 1033, 1030**, all exit zero. The 1033 boot added only existing `Chat.console_automatic_submission`, `Chat.console_change_review_read` and `DB.character_conversation_search`. This establishes wider baseline variation than the guard's documented +/-1; it does not justify a budget increase or unrelated cost shedding.
- `attached-boot-green.log`: 33 card/catalog/other passes and one 1032 census failure, with the recovery graph absent. Its extra existing residents were a subset of the baseline variation above. It is retained as a failed batch, not reported as wholly green.
- `first-use-boot-alone.log`: **4 passed**, including both complete boots, a 1031/1031 warm census and all explicit recovery absence checks. The warning correctly reports zero headroom; no limit or diagnostic snapshot was widened.
- `cold-lifecycle-green.log`: **5 passed, 51 deselected**, covering Direct/RAG cold controls, received cold controls, preparation failure and first queue/edit initialization of the existing controller. The earlier broad first-use run used an incorrect test hook map; it is superseded by this corrected subset and is not a passing claim.
- `first-use-static-final.json`: no introduced per-file Ruff findings across all eleven changed source files; inherited findings remain attributed. `first-use-worker.log`: no new worker/publication-contract violation.
- At report freeze, `first-use-dispatch-owned.log` and `first-use-native.log` were still running. Their closed outcomes must be recorded in the implementer's qualification note before publication. This report does not substitute partial progress for those final outcomes.

The census measures assignment-time residency, not latency or a steady-state import count. This scoped review does not claim a boot-time speedup or repair the pre-existing background census variation.

## Assessment

**Local code ready: yes; no unresolved scoped code finding remains.** Publication/merge remains conditional on successful closure of the pending dispatch/ownership/lifetime and native interaction qualification, immutable-commit hash verification, current-head required PR CI, branch currency and zero unresolved review threads. The human-authorized local-review substitution for Qodo applies to PR3162; it does not waive those remaining gates. The root implementer retains all remote and queue actions.
