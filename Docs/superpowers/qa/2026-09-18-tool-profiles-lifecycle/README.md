# TASK-32778 — Tool Profiles action and loading ownership

Queued row actions now belong to the actual control that was pressed. A refresh
cannot substitute another profile or a newer revision. Settings observes
app-owned initialization without cancelling it when the screen refreshes or
leaves; another waiting consumer and reopened Settings receive the result.

## Targeted evidence

78 distinct cases pass. No full suite or provider requests were run.

| Scope | Result | Evidence |
| --- | --- | --- |
| Queued Export/Edit/Bind/Remove across replacement and revision changes | 8 passed | [Targeted log](targeted.txt) |
| Settings refresh/removal, async/threaded initialization, second observer and reopen | 4 passed | [Targeted log](targeted.txt) |
| Cold workspace provisioning | 10 passed | [Targeted log](targeted.txt) |
| Design-token and component governance | 26 passed | [Targeted log](targeted.txt) |
| Existing Tool Profiles presentation and workflows | 30 qualified: 28 initial passes and 2 corrected viewport passes | [Initial summary](existing-initial-summary.txt), [viewport follow-up](workflow-scroll.txt) |

The corrected pre-fix regression run failed all ten original cases
([failure excerpt](red-summary.txt)). The final matrix adds real threaded
composition and a second app-owned observer. The first attempted fixture waited
for every app worker, including its deliberately held dependency; that run was
interrupted and does not count as loading evidence.

Fifteen existing workflow cases initially failed at profile selection before
their assertions. They now use the repository's private-profile process helper.
Two then attempted clicks at row 37 in a 35-row viewport; they now scroll the
real target into view first. Every original assertion is unchanged, verified
by AST comparison. No service calls replaced the clicks.

Scoped Ruff introduces no diagnostics. Existing Settings/panel/test baselines
remain 114/1/3; new tests pass. The changed Settings method and small files pass
formatting. Backlog uniqueness, Windows-compatible paths and diff checks pass.
Independent review found no introduced blocker. Exact source hashes and counts
are in [the verification receipt](verification.json).

This qualifies mounted lifecycle behavior. No visual values changed and no new
native screenshots were taken. Complete management journeys, export recovery,
and compact/wide dark/light presentation remain in
[the Tool Profiles review ledger](../../reports/2026-09-18-tool-profiles-review.md).
ADR-107 applies; publication and permission authority are unchanged.
