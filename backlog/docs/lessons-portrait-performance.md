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
