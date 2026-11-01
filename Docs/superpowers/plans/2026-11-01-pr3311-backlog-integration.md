# PR3311 current-dev integration

Integrate TASK-33620.1, preserving its emergency-stop indication and ordinary manual-send gate, current exact-input custody, typed attachment blocks and attach-reconciliation ordering.

ADR required: no new ADR.
ADR path: backlog/decisions/230-console-send-preparation-and-io-ownership.md; backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md.
Reason: repair regressions introduced by asynchronous stop writes using the existing captured-command and physical native-retirement contracts. No storage format, cross-process lock, background queue or new execution authority.

1. Rebase onto the latest reviewed merge immediately before publication. Preserve current-dev changes in every overlap; do not restore older whole-file versions.
2. Reproduce the independent review's two Important findings: a held physical write erases later/retyped/other-chat input, and competing set/clear operations can publish out of order. Exercise the real command handoff and production sentinel writer in a private profile.
3. On success, consume only the command's captured draft using the resident command-draft helper; capture before any await for direct callers and preserve failed-write input. Retain exact composer/generation identity across navigation.
4. Admit at most one Console stop mutation per app at a time, shared by command and chip Clear. Visibly refuse concurrent mutations, retaining their drafts. Keep the admitted native Future and admission occupied until its physical operation finishes despite repeated caller cancellation; publish the actual outcome before releasing admission. No cross-process serialization claim.
5. Run focused stop/view/command/queue and neighboring control tests, and compare failures with the exact base. Regenerate and review the diagnostic inventory under ADR228 if new diagnostic boundaries require it. No full sweep or hosted CI.
6. Record current evidence separately from historical live claims. Retain existing follow-ups .1-.7 explicitly: other submission gateways, held Send-button command route, narrow reason layout, heartbeat delay and typed-reason presentation are not silently certified by this integration.
7. Obtain independent requesting-code-review signoff after repairs and tests. Refresh live non-draft/head/threads, force-with-lease the reviewed result, merge with the already authorized narrow CI bypass, verify actual merge and exact protection restoration. Drafts and PR3267/3297 remain untouched; no release work.
