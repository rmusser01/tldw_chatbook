# macOS recovery diagnostic repair — 2026-10-29

PR3286 remains draft; AC24 remains open. No generation requests or product
children were used: cumulative actual457/31; the approved single60/3 qualification
is untouched and remains conditional on successful current-head normal CI.

## Observed boundary and limits

On73112b77977b64979fda79c341907807dd5ecfda, normal PR
[Eval recovery macOS job123703198977](https://github.com/rmusser01/tldw_chatbook/actions/runs/40946517600/job/123703198977)
passed80 tests and failed the retained-eval rebackup journey. Its restore child
returned-11 (SIGSEGV). The last captured output was the repeating10-second
`faulthandler.dump_traceback_later` dump in `_allocate_native_descriptor`.
That diagnostic tick is not the unchanged45-second parent deadline.

The exact original test passed unchanged locally (76.68s total). Two bounded
three-second A/B probes, including dynamic code churn and the real native open
helper, did not reproduce SIGSEGV with either diagnostic implementation.
No native C stack was available. The precise hosted crash cause is therefore
unproved. [CPython issue140815](https://github.com/python/cpython/issues/140815)
documents a race in the timed C stack dumper; it supports removing that risky
diagnostic path, not attribution of this particular hosted crash.

## Narrow change

The test's assembled restore child now uses the existing bounded Python-owned
`observe_threads` helper at the same10-second sampling interval, with
`stop_observer` in `finally`. Native fatal reporting remains enabled; diagnostic
cleanup preserves an active restore exception. No production code, storage or
permission guard, assertion, or timeout changed. An AST comparison confirms all
51 statements of the product restore body are identical.

CI runs the two new controls and preserves only the existing three stdout-log
leaves plus `restored/home/restore-stacks.json` and `restore-stacks-fatal.log`
on failure across all three operating systems. The actual tempfile root is
resolved after Windows private-temp setup; no pytest fixture root is moved.

## Verification

- Both new controls failed against the original C watchdog, then passed after
  replacement. They execute the actual assembled child, forbid the C timer,
  retain real bounded metadata, stop the observer and preserve primary failures
  with and without a diagnostic cleanup failure.
- Ten focused tests passed in84.62s: the two controls, relevant existing observer
  controls, and the original full retained-eval journey (83.27s).
- Workflow YAML, actual locator execution, preservation of prior environment
  entries, all-OS failure gating and exact diagnostic glob matching passed a
  temporary offline self-check. Changed Python Ruff lint/format and whitespace
  checks pass.
- [Scoped review](review.md) found missing artifact retention; the workflow
  change above addresses that finding. Hosted CI remains the required evidence
  for this platform crash; local green tests alone do not establish its cause.

Raw diagnostic logs and controls remain under
`/private/tmp/task35582-macos-watchdog-diagnosis`,
`/private/tmp/task35582-macos-diagnostics-{red,green}.log`, and the failed job
receipt/log in `/private/tmp/task35582-v1-source-final60`. They are not copied
into the PR. No full suite, CI rerun, new preflight or live qualification was run
for this repair. No new ADR is needed for test diagnostics; existing ADR281 and
recovery ownership contracts remain unchanged.
