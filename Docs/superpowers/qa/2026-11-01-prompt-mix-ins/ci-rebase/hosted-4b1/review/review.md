# PR3346 bounded approval-readiness review

**Verdict: CLEAR. Critical: 0. Important: 0. Minor: 0.**

Reviewed only the hosted wrong-family UAT follow-up against published base
`4b1e070c972320e1353b6bac7e185035db125571`, TASK38544 AC28 and the existing
ADR103 evidence contract. Prior feature/native reviews were not repeated.

## Strengths

The correction waits at the exact native lifecycle boundary. Production
`ChatApprovalCard.set_batch` creates the header as a pending child of its row,
then synchronously registers the row through unawaited `mount`. Installed
Textual inserts/attaches that row before starting its message pump;
`Widget._compose` awaits its children's mount events before
`MessagePump._pre_process` finishes Compose/Mount and sets `is_mounted`.
The queried parent can therefore exist while its header is absent.

`mounted_approval_rows` returns the entire queried list only after every row
mounts. It never selects by cardinality, family, tool name or header content.
Wrong or empty committed headers still reach the original content assertion;
multiple committed rows still reach the exactly-one-row assertion. No arbitrary
sleep, extra polling layer or timeout change was added.

The permanent real-App/card control observes two queryable parents with zero
headers before yielding, then the same two complete rows after native mounting.
Its deliberately wrong nonempty header remains ready. The original factored
raw-query predicate fails that same permanent control at the intended pre-mount
assertion (1.16s); the corrected predicate passes (0.89s). This deterministic
system-event proof establishes the mechanism, not the hosted timestamp/order.

## Source and authority checks

Independent full-module AST comparison normalizes only the added helper and
one delegated return. The complete original UAT module then matches 4b1,
including all **23 original assertions**, every action/decorator, exact
request-count 3 and argument checks, fast Approve Once, receipt authority and
all original deadlines. Removing only the new control makes the support module
AST identical too.

Only these two non-documentation files changed:

- `Tests/QA/test_project_chat_uat.py`: SHA256 `bff4182da1994ceb691a644f7c458fd3fc321d64bf58df51be60670eac985681`.
- `Tests/QA/test_project_chat_uat_support.py`: SHA256 `8f22db83a9ebf3e3c5f2921894f838b1e745b181f07d6fa77c2f48c4c5a93329`.

Runtime, existing UI/Architecture/CI/Performance tests and size ceilings,
workflows, scripts, dependencies, census and exact four-case validator are
unchanged. The support file already has whole-file Derived selection; no new
census entry or workflow/validator change is necessary. Existing ADR103 applies;
no new architecture decision is introduced.

## Evidence reviewed

The final original three-file CI cohort records **74 passed in 156.49s**, including
all four exact UAT variants and the new control. I independently inspected its
JUnit: 74 cases, no failure/error/skip, and the original four UAT names exactly.
All four receipts are complete with run/chat/assistant identities and clean
model/network fields. Wrong-family completed in 41.642s, used actual
`list_directory` Approve Once and ended at request_count 6; its unchanged
pre-click assertion still required 3. The unchanged receipt checker exited 0.

I independently verified all **45 explicitly inventoried producer artifacts**
(1,921,471 bytes) and all **3026 frozen source hashes** against the final checkout.
The producer's lint comparison retains exactly three inherited findings with
zero added diagnostics; new blocks are formatted. The initial sandbox
PermissionError is setup-only. The original unmodified full-app local node
passed in 33.02s and is not represented as a natural race reproduction.

## Issues and assessment

Critical: none. Important: none. Minor: none. No actual review issue remains
unaddressed in this bounded change.

**Ready to merge: Yes for these exact test-only bytes.** Fresh published-head
protected CI and thread closure remain root-owned merge requirements; this
review claims neither hosted green nor completed merge. The separately observed
Settings performance miss has no established cause and prompted no source or
budget change here.

Skipped: no reviewer app/test/native/performance run, full feature re-audit or
full suite. Remaining risk: hosted scheduling is not replayed locally; final
published-head CI must confirm integration.
