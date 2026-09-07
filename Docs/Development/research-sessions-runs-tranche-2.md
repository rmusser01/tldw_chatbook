# Research Sessions And Runs Tranche

Date: 2026-04-22 (updated 2026-08-15)

Source spec: `Docs/Parity/2026-04-21-capability-matrix.md`

Status: first-slice landed; live streaming landed since; local autonomous
execution still deferred (now tracked as task-16322).

## Landed Scope

- typed server research-runs schemas and API-client methods for create, list, detail, pause, resume, cancel, bundle retrieval, artifact retrieval, and checkpoint patch-and-approve
- local SQLite-backed research run and artifact store for standalone Chatbook operation
- local research service for create, list, detail, pause, resume, cancel, artifact, and bundle operations
- server research service wrapper over the tldw_server research-runs contract
- source-aware research scope service with local/server routing and runtime-policy enforcement
- ~~dedicated Research screen and source-switched Research window~~ (the
  screen registration was removed by task-255; see the 2026-08 update below)
- ~~app bootstrap and navigation wiring so Research is a first-class
  destination~~ (removed with the screen; `Research_Window` /
  `Research_Modules` remain in-tree but are not reachable from navigation)
- server event stream consumption and live status updates
  (`Docs/superpowers/plans/2026-04-23-research-live-events.md`, landed
  2026-04: `observe_run_events` SSE in `server_research_service.py`,
  event streaming through the scope service, and the "Watch Events" UI in
  `Research_Window`)

## Explicitly Deferred

- local autonomous research execution engine — now tracked as
  **task-16322** (a launched local run currently only writes DB rows; the
  `web_deep_search` pipeline is not yet connected to the run lifecycle)
- rich artifact and bundle inspection UX
- mounted checkpoint review and patch-and-approve UI (local checkpoint
  approval still uses a placeholder id; see `Research_Window.py`)
- local/server sync, mirror, or mixed-view behavior
- research provider administration, which remains tracked separately under `Research Search / Provider Surfaces`

## Verification

Focused verification was run against the Research Sessions slice AT THE
TIME (2026-04; kept as a historical record — two of the referenced files
have since been deleted):

```bash
python3 -m pytest \
  Tests/tldw_api/test_research_runs_client.py \
  Tests/Research_Interop/test_research_scope_service.py \
  Tests/UI/test_research_screen.py \
  Tests/UI/test_screen_navigation.py -q
```

Result:

- `26 passed in 3.12s`

Additional syntax verification (historical; `DB/Research_DB.py` and
`UI/Screens/research_screen.py` were later deleted — the former
`Research_DB` store is now `Research_Interop/local_research_service.py`,
and the screen removal is recorded in task-255):

```bash
PYTHONPYCACHEPREFIX=/tmp/tldw-research-pycache python3 -m compileall \
  tldw_chatbook/Research_Interop \
  tldw_chatbook/DB/Research_DB.py \
  tldw_chatbook/tldw_api/research_runs_schemas.py \
  tldw_chatbook/UI/Research_Modules \
  tldw_chatbook/UI/Research_Window.py \
  tldw_chatbook/UI/Screens/research_screen.py
```

Result:

- compileall completed without syntax errors

## 2026-08 Update

- **task-255** removed the orphan `research` screen registration;
  `Constants.TAB_RESEARCH` now aliases to the Library. The deferred
  decision on `Research_Window.py` / `Research_Modules/` (keep and wire,
  vs delete) is explicitly deferred to **task-16322**, whose local
  execution engine would give the window something local to observe.
- The **`web_deep_search` agent tool** (task-1356, gated by
  `[tools] web_deep_search_enabled`) exposes the
  `generate_and_search` + `analyze_and_aggregate` pipeline to agents with
  deadline/cancellation handling and byte caps. It is NOT connected to
  the run lifecycle — that connection is part of task-16322.
- **task-16331** added citation verification to the pipeline: `[n]`
  markers are resolved against evidence ids, quoted spans are checked
  against scraped originals (verbatim-first ladder), and the counts ship
  in the tool's honesty footer
  (`Web_Scraping/deep_search_citations.py`).
- **task-16332** collapsed the duplicated research service wiring in
  `app.py` to the single `_wire_research_services` path.

## Outcome

Chatbook now has a credible standalone-first Research Sessions crosswalk:

- `local` mode is backed by Chatbook-owned persisted run and artifact records
- `server` mode operates against tldw_server research runs without copying them into local authority
- the run/event/artifact service seams and SSE observation are in place;
  the missing pieces are the local execution engine (task-16322), the UX
  decision that follows it, and richer bundle/checkpoint inspection.

The remaining work for this domain is execution depth and observation fidelity, not first-slice CRUD/control alignment.
