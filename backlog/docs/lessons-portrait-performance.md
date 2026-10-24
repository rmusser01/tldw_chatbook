# Lessons: portrait performance storage

## Caught nested write errors need a local savepoint (TASK-36542, 2026-10-23)

**Incident.** The first atomic Apply passed ordinary rollback tests. A caller-held
transaction that caught an injected frame-insert failure then committed revision 2
with only part of the replacement, instead of retaining revision 1. The existing
`TransactionContextManager` leaves rollback to the owner of an outer or borrowed
transaction, so its nested context alone did not isolate the failed operation.

**What to do.** Wrap each multi-statement performance Apply/Remove in a SQLite
savepoint inside the existing transaction. Test a caller catching failure and
successfully committing its outer transaction, then read the record again. The
TASK-36542 regressions retain all prior frames and the prior revision after both a
failed Apply and a failed Remove; an uncaught outer failure is a separate control.

## Count gate windows instead of waiting for rounded milliseconds (TASK-36543, 2026-10-23)

**Incident.** The first silence gate accumulated 40 ms before closure. Two nominal
20 ms windows can each be slightly shorter at fractional sample-rate boundaries;
the regression with two 19.999 ms quiet windows returned small instead of rest.
That postponed closure to a third window despite the two-raw-window contract.

**What to do.** Count consecutive below-threshold raw windows for the silence gate;
use each window's actual duration only for exponential smoothing. Separately test
fractional-rate source positions and two slightly-short quiet windows. The corrected
TASK-36543 filter closes on the second quiet window without accumulated-time drift.

## Retire output identity before releasing in-flight observations (TASK-36544, 2026-10-23)

**Incident.** Stop cleared the latest mouth, but a notification already holding an
open pose published afterward with the same injected clock timestamp. Timestamp
ordering alone accepted it and reopened the mouth after Stop. Repeated old-sink
cleanup could also clear a replacement sharing the request's presentation channel.

**What to do.** Allocate an output epoch on successful sink admission and one
retirement epoch on termination. Retain that retirement marker for duplicate
cleanup; in-flight poses retain their older epoch. Serialize non-realtime analysis
and metadata release, while keeping Stop independent of the publisher lock. Test
the equal-timestamp race with barriers, rather than relying on elapsed time.

## A prepared loud frame can outlive its speech sample (TASK-36546, 2026-10-23)

**Incident.** A barrier-held mosaic render started while the owned mouth was open.
After the channel changed to silence, the old render completed and painted blue
(open) pixels over the resting portrait. Selecting the right role at timer entry
was insufficient; slow rendering created another publication boundary.

**What to do.** Recheck source/context and the current qualified mouth immediately
before adopting rendered pixels. Only record a pose as displayed when the paint
actually commits. Keep current/next images reserved until their render workers
finish, then close discarded results. The TASK-36546 barrier regression verifies
actual fitted pixels, not only the earlier role selection.

## Reused Console fixtures need their defining constructor namespace (TASK-36546, 2026-10-23)

**Incident.** Importing the existing mounted avatar fixture into the performance
test initially failed with an uncaptured-runtime refusal. The ownership fixture
looked at the importing module, while the real app constructor ran in the fixture's
defining module. An automatic unused-import fix also removed its explicit ownership
fixture import, making a later standalone run fail before the behavioral assertion.

**What to do.** Import fixture dependencies explicitly with their pytest re-export
noqa, and expose the reused fixture's defining namespace through the ownership
helper's existing `_character_screen` seam. Keep its bootstrap profile convention.
Do not reset runtime or recovery registries to conceal the refusal. The new
performance and existing geometry files then run independently with actual owner
retirement. Under NO_COLOR, compare shade glyphs instead of absent background
colours; a HEAD-source probe reproduced the old colour-only assertion failure.

## Distinguish the Console owner from its mounted harness (TASK-36547, 2026-10-23)

**Incident.** The first Personas-to-Console handoff test pushed Personas onto the
TldwCli returned by the reused Console fixture, but the Pilot belonged to the
fixture's separate `ConsoleHarness`. The Pilot remained on ChatScreen. Pushing on
`pilot.app` exercised the real mounted Console session and its invalidation path;
loading the Roleplay sheet on that host and retaining the Personas screen allowed
Apply/Remove to verify red prepared pixels and the original yellow still.

**What to do.** Read a reused fixture's yielded owner/host contract. Use its Pilot's
actual App for mounted navigation and workers, while preserving the captured owner
as the screen's application service source. Do not replace runtime ownership or
recovery registries to make a UI test mount.
