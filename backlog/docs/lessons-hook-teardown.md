# Lessons: hook teardown verification

## Exercise trusted projection beside a real standalone hook (TASK-33648, 2026-10-06)

**Incident.** The initial SessionEnd disposal fix passed its standalone close,
disposal, grant-change and cancellation controls. Independent source review then
found that installing a native plugin enables the shared event projector, which
returns a copied event even for a standalone handler. Comparing the authority
callback with the original event by object identity consequently suppressed a
valid granted SessionEnd. A real installed/trusted/activated native plugin plus
a saved standalone grant reproduced both disposal paths as missing process
markers (two behavioral failures, no errors).

**What to do.** Cover supported mixed owners through their real projector. Keep
original host-event/execution provenance, but authenticate the exact trusted
projected callback through its existing live delivery; copied labels alone grant
nothing. Replace inherited per-delivery context and reset it on every exit, and
retain public replay, current-grant and physical-cleanup checks. The corrected
62-case selection passes; its broader mounted-executor teardown timeout remains
non-green and separately reproduces on the complete unchanged base. Evidence:
`Docs/superpowers/qa/2026-10-06-task33648-session-end/`.


## Join the original producer and distinguish pending proof (TASK-38531, 2026-11-02)

**Incident.** Native v2 MCP integration initially retained a fresh nested host for
both planning and notification of the same ToolHookRun event. One original host
never returned and actual close stalled. Reusing only the exact original event
object/live delivery fixes the leak; closed-capacity SessionEnd and final native
worker/host tests exercise real close and release. A separate cancellation test
held the actual bridge's return after its MCP request finished. Marking the entire
owner cleanup-unproven at logical delivery completion permanently poisoned later
physical drain, even after the original bridge finally returned.

**What to do.** Keep known pending MCP work on its original stopping operation
until the original worker/bridge/request reaper supplies physical terminal proof.
A timeout, cancelled Future or parsed result is not that proof. Failed required
plugin settlement must likewise retain protection until the exact original
request settlement succeeds; do not clear it with a logical result or fabricate
completion. Cross-task/thread workers must enter through worker isolation and
reacquire the exact original admission sources; copied contextvars are not a
worker activation. The final tests hold actual bridge, request and release
callbacks and verify original storage remains owned until their return.


The same Task38531 affected run exposed a second real leak: a SessionEnd guard's
turn-context checkpoint correctly refused its missing input owner, but MCP
producer failure bypassed ToolHookRun.settle and left one original host retained.
Always enter the original settlement/discard path exactly once, including failure,
without waiving its required checkpoint or physical-write failures. A further
second-Send control distinguished accepted context lifetime from its spent input
registration: only positively retired original custody may omit that registration
for passive reads; a logical retired flag with negative native drain must still
refuse. Provider/schema revocation and closed dispatch remain separate negatives.
