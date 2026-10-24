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
