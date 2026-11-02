# Hosted f0a idle-refreeze test follow-up

Capture head: `f0a856d1b914c668a6d8ad28baafd91adb9b43a6`, based on dev
`88af9f937908b46e8c5f9f7fddd9e8748c917d46`.

Fresh CI passed 39 of 40 test/preflight jobs, including the previously failing
UI24 and UI26 shards. Perf Guard, both Tool Recovery Python versions, and all
three Eval Recovery platforms passed on their first attempts. UI18 alone failed;
the required aggregate consequently failed. The retained workflow snapshots
qualify that head, not an uncompleted follow-up run.

The idle-heap test waited for cycle collection, then immediately asserted the
`first_send_preimport` maintenance reason. A controlled native ordering reproduced
the hosted assertion: an already-armed `run_settled` timer collected the cycle
before the requested warmup timer completed. The probe delayed only the second
original callback submission; it did not fake collection, timers, busy state or
owner retirement. No production defect was demonstrated.

The final existing ten-second wait now observes both original outcomes. The exact
reason assertion, actual received-custody and no-maintenance checks, physical
retirement, real GC/weakref collection and all timeout arguments remain intact.
There are no production, census, size-pin or performance-threshold changes.

Validation:

- Original observational run: passed in 16.59 seconds.
- Controlled old predicate: reproduced the exact assertion in 8.87 seconds.
- Same controlled ordering with the corrected predicate: passed in 8.27 seconds.
- Ordinary exact native test without the probe: passed in 5.59 seconds.
- Original assertions and timeout arguments are AST-identical; Ruff remains
  zero before and after. Existing unrelated formatter debt is unchanged.
- Independent review has no actionable findings. No full-file or full-suite
  result is inferred from these focused runs.

[Implementer report](idle-refreeze/report.md) and
[independent review](hosted-and-review/mixins-pr3346-idle-refreeze-review.md)
explain the trace and its limits. `artifact-map.json` retains 33 explicit artifact
references, including the original and final probes, source snapshots,
RED/GREEN traces and hosted results. Original relative paths are retained;
text wrappers and deterministic gzip preserve whitespace-sensitive evidence.
Two retained copies of the hosted log redact three occurrences each of a
constructed credential fixture from pytest duration node IDs. GitHub push
protection rejected the literal fixture; no bypass was used. The fixture's source
construction, original-log hashes and redacted hashes are recorded in
`redaction-proof.json`; original raw logs remain local. All other retained
artifacts match their original bytes.

Existing ADR-198/230 and the original owner semantics apply; this test-only repair
requires no new ADR. TASK-38544 and the testing-evidence lesson record the incident.
Fresh hosted confirmation is tracked on draft PR #3346. Qodo remains bypassed;
auto-merge remains off.
