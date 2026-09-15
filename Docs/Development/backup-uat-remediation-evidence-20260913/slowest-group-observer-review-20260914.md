# Slowest completed admission group — independent review

APPROVED. No actionable finding. Base ec139a0241; exact final hashes in /private/tmp/uat-slowest-group-independent-hashes.json. No product/source edits, network, platform run or agents used.

The delta retains one reference to the already-created completed group detail. Selection uses the existing completion lock after all final wall/CPU/error fields are written; output deep-copies it under that same lock. The additional bound is one completed record beside the rolling16, not a history list. It introduces no new call wrapper, filesystem/native observation, timer, lock, deadline or metadata field inside that record. Existing caller basename/function/line, numeric topology, thread ID and bounded target-call counts are retained; no root values, private data, paths or exception text are newly exposed.

Strict greater-than preserves the first completed record on equal elapsed duration. Concurrent completion order cannot replace a larger record with a smaller one because comparison and replacement are under the same lock. Nested groups retain their own local detail and thread-local restoration; the completed record is not relabeled as active. Before any completion, slowest_group is null. Failed original calls still complete their record with errors=1 and re-raise the same exception; successful results are untouched.

Independent full diagnostic suite:9 passed in0.26s, /private/tmp/uat-slowest-group-independent.log. An exact whole-module AST comparison against ec139a0241, after removing only the new two-case test function, confirms every original import/function/trailing assertion is restored unchanged. Thus the reported intermediate insertion NameErrors were repaired without weakening original tests.

Two additional deterministic private probes pass in2.58s (/private/tmp/uat-slowest-group-private.log): nested calls plus equal-duration tie retain the correct original completed record; reversed real-thread completion order selects the largest elapsed call. Probe source: /private/tmp/test_slowest_group_review.py. No sleeps or platform emulation are used for scheduling; original clock/observer state is restored and threads are joined.

Limit: only the slowest COMPLETED call is added; a process killed during its slowest still-active call can still rely only on existing rolling active details. Inclusive wall/thread/process CPU timings are not additive and do not identify a filesystem path or imply a performance fix. Existing remote runs retain their old diagnostics; no retroactive attribution or Windows acceptance is claimed.

Final naming-only update verified: reversing path_marker→secret solely in the new test reproduces the exact previously reviewed file SHA256 a6572d83… . Original full test AST remains identical to base. Helper unchanged; final test hash refreshed. Approval retained without redundant rerun. Parent reports final9 pass and no nonassert Bandit finding.
