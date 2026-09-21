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
