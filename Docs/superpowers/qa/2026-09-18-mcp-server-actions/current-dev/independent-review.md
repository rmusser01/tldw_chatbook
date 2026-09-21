# Independent review — 2026-09-21

Production reviewer `mcp_server_action_review` reviewed server canvas target
ownership, synchronous mode retirement, async rendering and deterministic tests.
The review found two P2 issues, both reproduced before repair:

1. A delayed departure worker could retire a fresh Delete press published by a
   newer sync. Captured retirement revisions now prevent that refresh from
   replacing newer controls. Both before/after-release delivery cases pass.
2. An accepted Confirm's pruning could overlap a departure rebuild, publishing
   duplicate button IDs before dispatching deletion. A post-prune revision check
   suppresses stale publication while the handler still dispatches its captured
   target. The actual-pruning overlap regression passes.

Final production review: no remaining blockers. The reviewer inspected the fix
and regression and did not run tests or launch the app.

Native runner reviewer `mcp_compact_review` independently checked shared CLI and
private-profile admission, collision refusal, canonical IDs, owned cleanup,
restored callbacks, released held work, real handler/persistence assertions,
paint/hit/clip qualification, source provenance and supported terminal entry.
No launch blockers found; `git diff --check` passed. The reviewer did not launch
the app or alter files. The primary agent subsequently ran and inspected all
four native journeys and verified clean shutdown.

## Qodo mount-completion finding

Qodo flagged the await of `mount_all()` as another possible publication race.
The production reviewer independently traced installed Textual 8.2.8:
`Widget.mount_all()` synchronously calls `mount()`, which invokes
`App._register()` / `_register_child()` and adds children to the parent before
returning `AwaitMount`. A later prune therefore sees those children.
`AwaitMount.__await__()` waits for mount events, refreshes layout and updates
mouse-over; it does not register controls. No yield separates the revision
validation from registration. Completing an older await can only schedule the
guarded current-Keep lookup and dispatch an accepted confirmation's captured key.
The reviewer found no reachable counterexample and advised against adding a lock.

Two new cases call the real `mount_all()` synchronously, then hold its consumer
before or after awaiting completion while a mode round trip replaces controls.
Both pass: original controls detach, replacement IDs are unique and mapped to
the current target, and the accepted deletion dispatches exactly once. The
reviewer inspected these tests and found no blockers. The before-await case does
not claim to hold child mount events incomplete; it holds the consumer's wait.

This follow-up changes tests and documentation only. The native production and
runner hashes remain identical to all 16 reviewed captures.
