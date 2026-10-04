# PR #2995 rebase, review and merge follow-up

This directory preserves the verification records for the user-authorized rebase, review repairs and merge follow-up. Historical QA in `../2026-10-03-console-chat-starts-dev-integration/` retains its original revisions and bytes.

## Revisions and scope

- Remote draft head before rebase: `ce918c467898eb24feff216bced45b779bf310c6`.
- Initial rebased base: `f843ca811f01da6c39d903b6cd7328d68d50416f`, including dev `01a2020981c6197e5cd9945e5287567ad977edfe`.
- Task1 source repair: `07e7c23cb58b39928af3393d3e447e172d804e86`.
- Final docs-only dev integration: `bc7c2dde9febfd93cbc3bc115c8cbc61c3ee752b`, including dev `0001eba40419859ce39ed4952f0f8df7b40639bd`. Task1 production/test bytes are identical after this rebase; see `docs-dev-rebase-receipt.json`.

Task1 repairs accepted-original handoff consumption after caret/selection navigation and defers failure-only compaction-copy imports. It preserves authored revisions, widget identity and unchanged startup limits. Confirmed stale covering fixtures retain exact draft, recovery and provider-delivery assertions. The detailed report records intermediate failures and immutable-base comparisons.

Final branch review found two small issues: duplicate grant clearing before close-ticket validation, and inaccurate child-tool documentation. Task2 repairs them at source `0d2e581ba96ef00adb02e2c800cd449582f8a629`: four focused controls fail before repair, then the complete shutdown owner passes36tests without pytest warnings. Fatal Ruff, formatter ratchets and whitespace checks pass; exact owned source fingerprints match tested bytes. See `task-2-report.md`; its scoped re-review is recorded in `task-2-review.md`. Current-head GitHub checks and Qodo review remain publication gates; this archive alone does not establish merged state.

## Evidence

| Record | Scope |
| --- | --- |
| `task-1-report.md` | Exact RED/GREEN commands, source commit, fixture comparisons and static checks |
| `task-1-review.md` | Independent spec and quality approval; retained qualifications |
| `final-branch-review.md` | Whole-branch production/test/schema/authority review |
| `task-2-report.md`, `task-2-review.md` | Final close-ticket/doc fix and scoped review |
| `baseline21.json`, `baseline21.log` | Original21baseline failure nodes pass on stable repair bytes |
| `ownership-green.log` | Complete owner78passed, one inherited strict XFAIL, five unsuppressed warnings |
| `behavior-green.log` | Affected start/compaction/RAG capture/system-prompt tracing181passed |
| `shared-child-closure-green.json`, `.log` | Genuine-child shared bridge confirmation:one separately qualified pass at bc7c2dde9f |
| `derived-checks.json` | All11derived preflight guards exit0 in recorded scope |
| `latest-dev-backlog-checks.json` | Both guards affected by docs-only dev qualified again |
| `rebase-receipt.json`, `docs-dev-rebase-receipt.json` | Dev ancestry and historical QA/source identity |
| `rebase-task-id-sweep.json` | Ref/history/worktree task-ID ownership sweep |
| `manifest.json`, `publication-audit.json` | Exact archived bytes, exclusions and scoped publication audit |

Counts overlap; do not sum them. Whole-file composer/startup diagnostic runs retain their failed expectations; later focused receipts qualify the repaired nodes. One setup-directory error in the separate child test is retained separately from its passing run. The original report links identify the disposable source locations; the corresponding basename files are archived here.

## Limits and exclusions

- Existing timer-owner strict XFAIL, test-process FD-growth and inherited invalid-escape SyntaxWarnings remain disclosed. These records do not locate or repair a production leak. No warning suppression or new skip/XFAIL marker was added.
- The older skill-await draft test isolates hook review. Separate real hook tests qualify unchanged/stale admission; combined skill-await plus hook refusal is an explicit coverage limitation.
- UI-ready modules1033/1033 and screen-preimport modules556/556 have no headroom. No budget increase or failing-snapshot refresh occurred.
- Historical live PTY/provider records retain their original source revisions. Current mounted tests qualify the authored handoff repair; no full repository sweep or general live-user qualification is claimed.
- Only selected top-level reports, receipts, logs and helpers are exported. Private test profiles, user configuration, databases, caches and immutable base checkout directories are excluded. Review package hashes are recorded; their immutable ranges permit reproduction without copying the large historical QA diff.
