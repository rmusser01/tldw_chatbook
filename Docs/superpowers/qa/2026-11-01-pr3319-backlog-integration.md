# PR3319 backlog integration evidence

Original source 44bcf4541b8b893120ccb8ffc5464d81e1acb172 rebased onto confirmed dev b5aef686c5ee190c30297ec6150ef839e6158827 as 84c98d5189feec28d386a36167e07d990464456d, followed by the reviewed repairs below. Non-draft PR only. Hosted CI and size limits were explicitly waived by the user; focused local tests and independent requesting-code-review remain required. No release/main work.

## Repairs and integration

Enter sends the canonical draft; Ctrl+E expands collapsed tokens. Raw-command paste/file insertions remain literal. Delayed/banked refusal restores raw authority only into an empty current draft and only when the restored segments are literal; independently authored later text is kept without inheriting shell authority. Oversized raw insertions are rejected before rendering, edit serial, undo or draft mutation using the executor's existing 16 KiB UTF-8 limit, with visible feedback and no truncation. Boundary-sized multibyte commands remain accepted.

The in-place Arm route checks the visible session and supplied Send-readiness guard. Repeated requests and unavailable/failed confirmation starts give feedback while retaining the draft and existing runtime permission gates. Existing ADR-093/031/230 govern these repairs; no new architectural decision.

Current-dev acknowledgement geometry, captured input, native retirement and trace/stop recovery are preserved. The census retains the exact 1285-entry dev prefix and appends 5 source selections plus the new repair file, totaling 1291. Size budgets were not raised.

The two different TASK-37800 claimants were disambiguated: the older Console task keeps 37800; the younger CI-preflight task becomes 38413 with its In Progress status and historical evidence preserved. The namespace audit checked historical task paths, existing worktrees and 318 remote refs before allocation. Only CI-specific references changed; the historical .test-tmp/task37800-final-2 command path remains unchanged. The workflow diff contains comments only.

## Focused evidence

Private receipts live in the sibling pr3298-resync/r-* directories and record command, XML, log, exit and before/after hashes. Mounted runners were serialized. All listed completed runs reported unchanged source hashes.

- r-f1e2fda8: causal RED: 15 failures/2 passing boundary controls across 17 new tests: restored trust, pre-render insertion bounds and real-screen Arm guards/feedback. An earlier fixture-setup error receipt is not causal RED evidence.
- r-c429b7b3: 39 passed/3 failed. All 17 new repairs and 5 original mounted raw/Arm cases passed. Three raw-routing fixtures lacked the current received_inputs field; adding None restored their intended pending-send shape without relaxing assertions.
- r-6f2d69be: pytest reported 37 passed/21 failed/2 teardown errors. Routing, Enter and expansion assertions passed; two expansion cases failed native-executor retirement. Settings cases refused profile admission before behavior assertions because the module lacked the existing bootstrap_profile marker.
- r-96f45f1e: the first Settings journey failed while waiting for an unrelated, normally cancelled prompt-summary worker. The save helper now waits only the actual settings-save-raw-cli group and preserves worker failures/cancellation.
- r-2c3959f5 and r-a3c38aac: first Settings journey reached hover with the Arm control outside the viewport. The test explicitly scrolls the real control into view; focus, hover and before/after geometry assertions remain.
- r-24be2d79: 10 Settings cases passed before the navigation case's synthetic three-second save hold expired. Teardown also exposed a startup backfill that escaped the factory's construction-only no-DB patch. The test now retains that same no-DB seam through mount, releases its held save in finally covering initial startup/wait, and awaits the actual save worker. Real navigation/save/handoff assertions are retained.
- r-ea1ecc6d: the navigation regression passed with 0 failures/errors/skips and clean teardown after that fixture correction.

- r-1ae145aa: 16 passed/1 failed, 0 teardown errors. All remaining Settings cases and both earlier expansion teardown cases passed. The deferred-capture preservation check asserted before the second durable acceptance: the log records its durable commit only during teardown. A bounded outcome/turn-retirement wait now replaces that early assumption; exact sent payload and retained draft assertions are unchanged.

- r-d75b88e0: 90 passed, 0 failures/errors/skips, including deferred-capture, three focus/keyboard controls and both census/task-ID contract files. Source hashes stayed unchanged.

Reconciliation of all receipts gives 191 unique targeted cases with a final passing result; no prior failure or teardown error remains without a later passing result.

This is accumulated focused evidence, not one clean aggregate sweep. Existing dependency deprecation warnings remain visible.

## Review and static checks

Independent /root/pr3319_integration_review reviewed the full integrated range and repairs using superpowers:requesting-code-review, with no remaining Critical/Important findings by inspection. Follow-up review approved namespace provenance, fixture shape, profile ownership, targeted save waits, explicit save release and the on-screen hover correction. The final execution receipts now qualify the reviewed changes; final commit review is recorded by the publication coordinator.

Diagnostic inventory verified 716 owners, 1065 TASK492 calls, 56 TASK31551 calls, 7791 TASK494 calls, 16 sink files; subsequent edits are test/docs only. Census passes 1291 entries and the task-ID/path check passes 6085 files. Scoped Ruff on 24 changed Python files: 56 base findings, 56 current, 0 introduced; the latest fixture edits were rechecked explicitly. git diff --check passes.

## Limits

No hosted CI, full-suite or cross-platform qualification was performed. Real mounted-screen tests do not establish that an external terminal or editor forwards Ctrl+E. Its keyboard fallback remains TASK-38410; narrow/run-together chip follow-ups remain TASK-38411/37802, and locked/unarmed red styling remains TASK-37801. This PR does not claim to resolve those separate issues.
