# Windows shutdown reconciliation follow-up

The normal CI run on33b5afc3a3 passed the repaired UI shard, Windows eval
recovery and Console P0. Windows GGUF evidence still failed: inventory[True]
timed out at180s in Console reconciliation. The observed inventory[False]
passed, but closing took142.37s, with136.45s on the owning thread. Its trace
reached repeated post-refresh reconciliation while timer cancellation was
inactive for most of that interval. The earlier timer-starvation control did
not establish this first failure's cause.

`LifecycleMixin._shutdown` sets Textual's exit fence before draining durable
owners. Layout timers then stop updating geometry, while queued post-refresh
callbacks can still run. The bounded section retried against that frozen
geometry. Its shared scheduled callback now clears its existing pending flags
and calls reconciliation only while Textual's public `is_attached` is true.
Shutdown drain ordering, timers, native admission and all live layout behavior
remain unchanged. No timeout, skip or workflow retry was added.

Two regressions queue geometry work while attached, then hold the actual
LifecycleMixin owner drain after exit begins. Wrapped and native bodies both
executed the forbidden callback before the fix; afterward the queued callback
retires without entering reconciliation. This reproduces the causal boundary
on macOS, not the Windows142s duration.

Validation:59 focused lifecycle/layout/scoped-allocation tests pass, including
all covered/native/hidden/unmount controls. The original inventory[True] passes
once in5.47s with its private profile and unchanged180s deadline. New-test lint
and formatting, changed product-range formatting and whitespace pass; three
pre-existing product-file Ruff findings are unchanged. A scoped independent
review found no actionable issue. Native Windows confirmation remains required
from normal CI on the next commit.

ADR required:no; this is a lifecycle bug fix using the existing attachment
contract. No provider generation or product child was launched. Actual cumulative
usage remains386requests/26children; the single approved60-request/3-child
qualification remains wholly unused and conditional on successful CI. AC24 is
open and PR3286 draft. [Retained logs, patch and review](shutdown-retention.json).
