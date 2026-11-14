# Console activation allocation continuation — 2026-10-21

This work stays in the replacement checkout on
`codex/console-latency-continuation`. Other sessions, the original branch,
dirty main and existing evidence are untouched. TASK31966 remains In Progress;
the five failing original activation intervals are not waived.

## Isolation at clean b8ecfdcae9

Fresh source-bound scale: 10,000 conversations / 250,004 messages / 10,000
characters, index ready, integrity OK, cleanup handles0. Keyword300 passes,
P95 104.643125ms. Neither is UI qualification.

All following controls use the unchanged original driver/assertions/limits but
temporary runtime instrumentation or behavior, so none is qualification:

- Mounted-row reuse: six activation interval failures, max80.188ms;
  117,731 Strip / 827,007 FIFO creations. Rejected as insufficient benefit for
  its generation/focus complexity; no row-reuse production change.
- Generic equal-content Static control: five activation failures, max105.676ms;
  112,201 Strip / 788,319 FIFO creations. Of2,740 observed Static writes,
  2,036 repeat equal string/Text content. Setup and approval writers contribute
  repeatedly even with their same copy. This motivates the local correction
  below, not a generic primitive override.
- Lazy native FIFO allocation:21 paired gen2 events;116,511 Strip /32,412 FIFO
  creations, post-Runner unfrozen FIFO3,802. Native capacities/lookup/eviction
  unchanged, no collector policy edit. Still NON-GREEN: one preparation and four
  activations fail, max70.308ms. No shipping dependency/runtime approval or fix
  is implied; shared environment files were never edited.
- Native identity-cache control: six activation failures, max98.152ms;
  119,871 Strip /841,991 FIFO creations. Avoiding a divide identity self-cycle
  alone is not sufficient. The redundant adjust-length guard matches an already
  installed native early return; no benefit is attributed to it.

These are separate instrumented populations, not exclusive cost subtraction,
proof of a leak, unique GC attribution or evidence that every original failure
is GC. Thresholds700/10/10 and collector enablement remain unchanged. Early
import-order and namespace-recursion experiments are setup-invalid and retained,
not credited REDs. Corrected control sources are retained; do not claim exact
as-run source preservation for the overwritten invalid prototypes.

## Bounded app-owned correction

Use current mounted content to skip only equal setup-step/staged-notice and
approval title/deadline/summary writes. No new cache or whole-state early return.
Visibility, class repair, countdown/timer retirement, batch generations, controls
and approval authority still run. Production AST outside those four writers is
unchanged. No appearance, timing, GC, profile or ownership policy changes.

- Native RED:4 failed1.37s. GREEN:4 passed1.29s.
- Initial affected run:8 failed/55 passed23.66s. Two tests read old CSS source
  and numeric literals; six mounts fail config-source admission before their
  assertions. Replace obsolete CSS assumptions with real width/full wrapped-copy
  checks; use the existing per-node bootstrap fixture for those six mounts,
  preserving their bodies/getters/admission.
- Final affected selection:81 passed32.97s, no pytest warnings.
- Original live countdown:1 passed2.46s; sandbox pytest-cache warning retained.
- New test Ruff/format clean. Production inherited Ruff14/6 and touched original
  tests/script3/2/1/1 unchanged, no additions; not full-file lint-clean claims.
- Ten artifact guards pass. Canvas initially cannot acquire a pinned input in
  the sandbox; the unchanged network-enabled guard then reproduces all6 outputs.
  Census155/floor155; new member appended without reshuffling existing members.

New full raw logs/frames/corpus remain LOCAL under
`/private/tmp/console-latency-continuation-fT3VrM`. Only hand-audited outcomes and
hashes are published. The RED log retains late shared-temp pytest housekeeping
warnings; subsequent tests use new task-owned basetemp paths, not shared cleanup.

Fresh clean-commit original scale/Keyword/UI qualification is next. No latency
clearance, native/Windows/participant completion, full-suite pass, integration,
push or merge is claimed. Existing unrelated baseline dispositions remain open.
