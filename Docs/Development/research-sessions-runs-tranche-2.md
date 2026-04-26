# Research Sessions And Runs Tranche

Date: 2026-04-22

Source spec: `Docs/Parity/2026-04-21-capability-matrix.md`

Status: first-slice landed, with live streaming and local autonomous execution still deferred.

## Landed Scope

- typed server research-runs schemas and API-client methods for create, list, detail, pause, resume, cancel, bundle retrieval, artifact retrieval, and checkpoint patch-and-approve
- local SQLite-backed research run and artifact store for standalone Chatbook operation
- local research service for create, list, detail, pause, resume, cancel, artifact, and bundle operations
- server research service wrapper over the tldw_server research-runs contract
- source-aware research scope service with local/server routing and runtime-policy enforcement
- dedicated Research screen and source-switched Research window for local/server run browsing, creation, and basic control
- app bootstrap and navigation wiring so Research is a first-class destination

## Explicitly Deferred

- server event stream consumption and live status updates from `/events/stream`
- local autonomous research execution engine
- rich artifact and bundle inspection UX
- mounted checkpoint review and patch-and-approve UI
- local/server sync, mirror, or mixed-view behavior
- research provider administration, which remains tracked separately under `Research Search / Provider Surfaces`

## Verification

Focused verification was run against the Research Sessions slice with:

```bash
python3 -m pytest \
  Tests/tldw_api/test_research_runs_client.py \
  Tests/Research_Interop/test_research_scope_service.py \
  Tests/UI/test_research_screen.py \
  Tests/UI/test_screen_navigation.py -q
```

Result:

- `26 passed in 3.12s`

Additional syntax verification:

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

## Outcome

Chatbook now has a credible standalone-first Research Sessions crosswalk:

- `local` mode is backed by Chatbook-owned persisted run and artifact records
- `server` mode operates against tldw_server research runs without copying them into local authority
- the TUI exposes Research as a first-class destination instead of burying it under evaluations or search

The remaining work for this domain is execution depth and observation fidelity, not first-slice CRUD/control alignment.
