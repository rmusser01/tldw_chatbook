# Mounted Interrupt loop-gate correction

Base: `61e28695bc4fa9a62e8f14771a529591576d96ce`. Branch:
`codex/sqlite-interrupt-ownership`. Production hooks, cancellation and lifetime
policy are unchanged.

The original mounted node passed its body but failed executor teardown at the
unchanged 300-second bound: **1 passed / 1 error in 300.63s**. The blocked
executor stack remained in the real bridge's `future.result()` call.

The provider runs on the bridge's own event-loop thread. The test directly set
both cross-thread asyncio gates: provider-to-test `entered`, then test-to-provider
`release`. A call-through probe preserved each original `Event.set()` and only
woke its owning loop. It observed both wrong-loop calls and completed the
original node: **1 passed in 10.27s**. The probe's late scratch/closed-owner
logging remains in its raw log; it is not a warning-free production teardown
certificate.

The correction schedules both sets with the owning loop's stdlib
`call_soon_threadsafe`. Original marker, Interrupt, sealed-session and cleanup
assertions and deadlines remain. No worker is cancelled as a test substitute.
The uninstrumented node completed including executor shutdown: **1 passed in
8.52s**. The complete affected module then passed **13 tests in 83.83s**, with
no pytest warnings. Whole-file Ruff and format checks pass; both Backlog guards
and whitespace checks pass. Self-review confirms only the test gate changed.

Commands used the existing interpreter with `PYTHONDONTWRITEBYTECODE=1`,
`pytest -q -p no:cacheprovider` and separate private `--basetemp` directories.
No full sweep or warning suppression ran. A prior Ruff cache-permission failure
is retained; successful static checks used `--no-cache`.

Raw logs and the transparent probe remain local; SHA-256:

| Receipt | SHA-256 |
| --- | --- |
| `/private/tmp/interrupt-owner-baseline-fZSzCN/mounted-baseline.log` | `5e05ddeba2ee9778da42197ec91ee6670771d73addf9273ee7b8dbde7f01e32c` |
| `/private/tmp/interrupt-owner-probe-vyoabj/loop_gate_probe.py` | `adde0fce7acff176485d172932fc65a52485234320da54558bcb259a9522d613` |
| `/private/tmp/interrupt-owner-probe-vyoabj/wakeup-control.log` | `34515b4f3a6e1a8c81f4e6194e760d029073c0ecd0b1b15c4f11466dc61911c2` |
| `/private/tmp/interrupt-owner-green-qK8aAO/mounted-green.log` | `6e4cdf16c931ff573cdb85c66cdd3ab384d93e4b322f7087334a5b26a9699d08` |
| `/private/tmp/interrupt-owner-module-CBZEVU/interrupt-module.log` | `1a2ab376e849db37cb6507eb33e513dfd6f4a55213fcafb6ff4f6c2bc76e2d1e` |
| `Tests/Chat/test_hooks_v2_teardown.py` | `3f3cd243dd1664c9b661f4e6dd88dcf29c9e0a70410b1ef0de0a3f53236c2cd1` |

This resolves this reproduced mounted Interrupt test teardown, not every
production executor lifetime or aggregate FD warning. The historical incoming
failure remains evidence, superseded by this fresh scoped causal repair.
No other sessions were contacted or interrupted.

The later strict AgentRuns retirement check exposed physical worker ownership
beyond the corrected test gates. Those causal REDs, whole-module failures and
the separately approved production repair are retained in TASK34661's QA, not
silently credited to this gate change. Final original module verification passed
13 tests with the native census (92.21s) and 13 uninstrumented (101.00s), without
pytest warnings. This test's source hash and original assertions/deadlines are
unchanged; whole-file Ruff/format pass. Independent review/integration and the
retained incoming static failures remain outstanding; task stays In Progress.
No source commit, push, merge or other-session control.
