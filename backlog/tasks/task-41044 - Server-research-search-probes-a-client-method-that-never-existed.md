---
id: TASK-41044
title: "Server research search probes a client method that never existed"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- cleanup
- research
priority: low
references:
- tldw_chatbook/Research_Interop/server_research_search_service.py
- Tests/Architecture/test_server_service_client_protocols.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Research_Interop/server_research_search_service.py:150` reads `getattr(client, "run_research_websearch", None)` and falls back to `research_websearch`. `TLDWAPIClient` has never had `run_research_websearch`, so the fallback always runs.

The probe exists only for test fakes (`Tests/Research_Interop/test_research_search_scope_service.py`). It is also the one `ABSENT_PROBES` entry in `Tests/Architecture/test_server_service_client_protocols.py`.

Found by TASK-36953.1 and endorsed by its review; still present on dev ecad404bd4 (2026-11-15).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The service calls `research_websearch` directly
- [ ] #2 Neither the test fakes nor `ABSENT_PROBES` name `run_research_websearch`
- [ ] #3 The research search and client-protocol tests pass
<!-- AC:END -->
