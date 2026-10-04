**I1 — Preserve denial when registration fails on both attempts and durable settlement fails — ADDRESSED.**

## Finding verdict

- `tldw_chatbook/DB/automatic_work.py:66` establishes the existing canonical resolved database key before the ledger can grant authority. Memory stores keep a private UUID on the DB object, shared by rebuilt ledgers and isolated from independent memory stores (`:73`). The getter returns the captured identity (`:80`). There is no lexical fallback, second registry or new root policy.
- `tldw_chatbook/Chat/console_chat_start.py:378` invokes the pure insertion fallback when ordinary registration raises. `tldw_chatbook/DB/automatic_work.py:91` uses the captured identity, original map/lock, exact owner/attempt and chain, and a fresh snapshot token. Neither insertion nor subsequent lookup/clear requires filesystem observation. Both original registration sites retain this denial (`console_chat_start.py:460`, `:591`). Failed durable settlement therefore leaves a same-store/current-owner refusal after physical exit and exact capacity release.
- Initial resolution failure precedes capacity claiming and active-item insertion (`console_chat_start.py:299`, `:309`). Actual native-start and constructor controls assert zero authority writes/claims and an intact draft (`Tests/Chat/test_console_chat_start.py:3855`; `Tests/DB/test_automatic_chat_starts.py:822`). File aliases and memory UUID sharing/isolation, same-owner recovery retention, replacement-owner recovery, and exact clear are covered at `Tests/DB/test_automatic_chat_starts.py:751`.
- The eight native cases at `Tests/Chat/test_console_chat_start.py:3714` exercise commit/provider workers, canonical-resolution/persistent-registration failures and original/replacement owners. The real SQLite trigger rejects durable review settlement (`:3764`). Repeated cancellation retains custody until worker exit; replacement item/token ownership survives (`:3797`–`:3827`). Accepted state, one charged generation, active durable root, healthy alias-handle refusal and unrelated-root admission are asserted after cleanup (`:3828`–`:3839`). Provider cases explicitly clear the local receipt flag after real acceptance and provider entry; this is an adversarial cleanup control, not a normal dispatch claim.
- Recovery still snapshots entries under its transaction and removes only the identical entry afterward (`automatic_work.py:1254`–`:1264`). Fresh insertion preserves its token fence. Both physical drains, settlement/absence guards, outcome fallback, exact active-item removal and captured-token release remain unchanged (`console_chat_start.py:461`–`:662`).

## New breakage in the fix diff

None.

## Evidence checks

- Reviewed the supplied fix package once for `9c0aa539712a2e2659a2b355d531228408932a83` → `eb7cb871390616de22d28c7a6679bbc9202841d7`, then read the relevant complete source boundaries. The intervening implementation BASE `41155ce71ed84e6cb5d9c720a6d1a2f736c35e53` is recorded as metadata-only. No Git command, source/index/HEAD/branch mutation or subagent dispatch occurred.
- Independently hashed all 46 R1 manifest entries and all 119 publication-manifest entries: no mismatch. Verified all 69 original artifacts through the 68 unchanged children plus exact before-I1 report snapshot; the current report retains that snapshot as its exact prefix.
- All five current source hashes match the actual GREEN argv receipt. Reversed the owned patch in memory and matched all five reviewed-base source hashes. Both original test files are exact prefixes. Independent AST comparison confirms the only changed production definitions are coordinator `_restrict_cleanup`, ledger constructor/getter/ordinary registrar, and the added pure insertion helper. All other methods and imports remain exact; the fork file remains byte-exact.
- Parsed actual XML, logs and exit receipts. Initial RED: 10 failures/11 cases; corrected RED: 9 failures/9 cases. Focused RED: 10 failures/10 cases, including all eight native cases failing specifically because the peer did not raise `AutomaticWorkRefused`; initial native identity refusal also failed. The earlier alias/setup and overly broad injection failures remain separately attributed.
- The recorded GREEN command uses shared Python 3.12, worktree PYTHONPATH and the eight exact selectors in `green-argv.json`: four new selectors expand to 12 cases; four authorized existing selectors expand to 15 cases. XML and log confirm **27 passed in 39.78s**, exit 0, with no errors/skips/xfails or pytest warnings. Fatal Ruff reports “All checks passed!”, formatting reports “5 files already formatted”, and whitespace exits 0. Source pins match these checks. No tests were rerun for this review; no uncovered source doubt required a new probe.
- Independently hashed 18,523 source/test/workflow/QA paths against the frozen initial tree map: only the four authorized source/test paths differ, with no missing paths. This preserves the Task14, import/loading, worker/route/Close and cap guards. The qualification ZIP remains 63,166,118 bytes with SHA256 `acecbdfb6ddbafe6df679c39f9f137f2bfb49a1f87b62556e1345021f8384c84`; no decompression/export/replay occurred. Original 61-case receipts and historical 35-loading evidence remain at their original source identities.
- Inspected closed-process and clean-source handoff receipts: all recorded executions have exits, native exits are asserted in GREEN, and final source pins match the committed handoff. External publication, current-head Qodo, CI/PerfGuard, latest-dev ancestry and merge remain root-owned gates.

## Out-of-scope observations

None newly identified. Previously recorded loading/headroom/FD concerns remain inherited and unchanged; this scoped review makes no new loading or resource measurement claim.

## Verdict

**Fix round: All findings addressed, no new Critical/Important breakage.** I1 is closed for this scoped correction gate.
