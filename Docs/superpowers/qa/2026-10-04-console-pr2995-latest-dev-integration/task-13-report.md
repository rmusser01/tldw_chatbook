# Task 13 newest-dev integration

## Result

**DONE_WITH_CONCERNS.** Qualified source: `9445ed93ed430ba0e3660ad2008621f6a809bd0e`. Worktree and index are clean; all owned qualification processes have completed. Independent scoped spec and quality review belongs to root before publication.

One conflict-free rebase replayed75 commits from `862c2eaa80bb744b71cb53a68952f8451b3b327c` onto `8f83422dde2a5b95da648882b8f7e09da5b41f08`, yielding `eee9d80a3e4571533fa9071dc0629a4d1c407024`. All six incoming commits and17 paths are retained. Recovery ref `codex/pr2995-task13-recovery-862c2eaa80` and a verified private bundle pin the original checkpoint; see `recovery.json`. No fetch, second rebase, GC or prune was performed.

The only new source commit is `9445ed93ed430ba0e3660ad2008621f6a809bd0e`: the two incoming controller formatter corrections (`rows = tuple` spacing and wrapped `saved_message_id` call), plus the controller ratchet lowered29367→29301. It changes two files,5 insertions/3 deletions. Controller AST is unchanged by formatting. The50-line slack rule is unchanged. Screen remains25192lines/759methods under25218/759; store22344, interrupt6479 and compaction4185 cap rows remain unchanged.

ADR required: no. Existing ADR097 semantic trace, ADR097 boot ratchets, ADR219 and ADR220 govern the incoming repair and preserved ownership. The upstream TASK33621.2 task, its completed criteria, user guides and residuals are retained. No new storage, permissions, profile policy, authority or generic dependency mechanism was introduced.

## Exact preservation

`source-union.json` independently reconstructs all29864 paths from the reviewed checkpoint and pinned dev. All ten incoming production files matched the preflight candidate SHA256 values before the narrow formatter overlay. `final-source-union.json` proves the only subsequent tree differences are the authorized two-file overlay, and every qualification source/test hash matches the clean committed tree.

`production-method-union.json` records exact body/AST/spans for each declaration in the ten affected owners. Every unaffected declaration keeps reviewed AST; each nonshared incoming declaration matches upstream AST. `four-shared-method-preservation.json` reverses the incoming delta from each integrated method and restores its complete reviewed body byte-for-byte:

- `ConsoleChatController._submit_draft_body`: only paused-preparation refusal copy changes; literal machine input, dual receipts, manual withdrawal before busy gate, staged exclusions, capture/preparation, coordinator and source/acceptance fences remain.
- `ConsolePromptQueueUIController.presentation_for`: adds live oldest-recovery reason to ordinary derivation while preserving final prepared Send and accepted Running overrides.
- `build_console_controllers`: adds the named live oldest-recovery callback; required recovery dependencies, Session-bound trace callbacks, current settings and hooks remain.
- `ChatScreen._build_console_workbench_state`: adds blocked-turn projection while preserving prepared-start Send and image-edit gates.

The incoming helper preserves aggregate categories, memory/tool mapping and omitted-image admission. Refusal reason travels through the terminal exception/recovery entry outside lifetime custody; exception args remain fixed, and diagnostic output remains content-free. Upstream archive refusal and oldest recovery shelf behavior are retained. No existing source/test assertions or fixtures were weakened or repaired. Task12's exact fixture adapter, source formatting overlays and copy-only stale-task note carry through the full union.

`historical-qa-carry.json` preserves all12337 Task12 final QA blobs:12336 remain exact original whole files. The authorized `Docs/QA/task-31245/switcher-reuse-and-mode-follow-up.md` exception retains the original7982-byte prefix and exact upstream4417-byte/70-line append. `initial-safe-artifact-pins.json` and `safe-artifact-carry.json` reverify518 prior safe files, including root Task12 handoff, Task9/12 phases and original35 loading receipts. None was rewritten.

## Qualification

The exact preflight selectors ran once with shared Python3.12, actual worktree `PYTHONPATH`, canonical unchanged `Tests/conftest.py` private profiles and the existing300-second timeout. **57passed,1warning in265.73seconds** (parent277.63seconds); no failures, errors or skips. Actual expansion is53 behavior cases plus four ratchets: the native literal/dual-receipt selector has three parameters, explaining the two cases omitted by the55-case preflight estimate. Selection and assertions are unchanged. `selected-argv.json` records actual argv and source/test hashes before execution; `selected.log`, `selected.xml`, `selected-result.json` and `actual-case-outcomes.json` retain complete output, case identities, stable hashes and result.

Scoped fatal Ruff passes for all15 changed Python paths. Required formatter check passes for seven already-formatted touched sources/ratchet. `formatter-inheritance.json` additionally measures each changed Python source/test through the installed Ruff0.16.6 formatter on stdin: every remaining full-file edit is exactly attributable to reviewed or incoming formatting; every formatting projection preserves AST. No whole-owner formatting was applied. Changed-source whitespace passes. Actual argv/output/results are retained separately for each check.

**Concern: unsuppressed session-end FD warning.** `Tests/conftest.py:609` reported start14/end383, growth369 over limit200. Pytest displays it under the last screen ratchet node because it is a session-end check; no per-case cause has been established. This is a passing cohort with a resource warning, not a clean resource-usage result. No suppression, raised limit, speculative fixture change or replay was performed. Root can scope any needed attribution separately.

## Loading, diagnostics and runtime carry

The35 loading passes at `a6887fdb8931adb9d1b99a376d63d9b9b66a5bb5` remain historical, with their exact logs, XML, manifests, source hashes and original warning attribution. No fresh loading pass is claimed or run. `loading-source-carry.json` maps all7689 previously pinned source hashes; only the authorized incoming paths/downward row differ. Original headroom warnings and measurements remain attached to their original source, including ready1033/1033 and preimport557/557.

`loading-authority-detail.json` proves the eager imported module sets of all nine existing production owners are identical. The new helper imports only inside the actual voice/durable request builders; its three project dependencies already occur in the reviewed controller closure. The diagnostic import is failure-local. New screen/recovery imported symbols come from existing resident modules. Nineteen constructor/native-authority/custody declarations retain exact AST; unchanged guard/profile/app/route/CSS/Session sources are pinned. The shared wiring reverse-delta proof preserves its required named recovery accessors and Session callback identity. Incoming mounted UI tests freshly qualify changed header/run chip/inspector/callout/shelf projections, including second-send refusal and recovery actions.

`import-worker-diagnostic-carry.json` scans only ten incoming owners. Existing diagnostic/sink/path projections and W001/W002 plus worker/screen-call ASTs are exact; the new pure helper adds none. The new keyed `record_send_stage` call uses the existing sink and is exercised by the incoming real-controller log-category test. No diagnostic inventory refresh was required or performed. No new boot CSS, service/worker construction or marginal preimport target was introduced.

## Residuals and handoff

Keep upstream documented residuals: startup trace-maintenance GC can race a revision before reservation; speculative voice capture still lacks omitted-media admission; Temporary-chat recovery can retain a duplicate after Send without capture; Chats-list blocked marker still follows run status; the incoming mounted file is absent from the upstream UI fast-lane census. Task13 does not repair or reclassify them. Known private relocation compatibility limits remain as recorded upstream/ADR220.

Self-review checked full source union, incoming and shared methods, exact two-file overlay, unchanged native/Close/acceptance/lock authority, actual test outcome/selection, inherited formatting, source carry, all prior evidence pins and QA exception. The manifest excludes private profiles, config bodies, databases, caches and recovery bundle. An initial sandbox write denial occurred before the evidence directory existed; the authorized escalated retry succeeded. An inspection-only metadata summary hit a list-vs-dict error; no source/test execution or evidence was altered by it.

Root owns independent Task13 spec/quality review, QA/export/publication/PR replies, current-head Qodo/checks, fresh ancestry and normal merge. No children, human-checkout edits, installs, full sweeps, new skip/XFAIL, source-only fresh boot or warning exemptions occurred. Tracked/index/HEAD/ref ownership is relinquished with this clean handoff.
