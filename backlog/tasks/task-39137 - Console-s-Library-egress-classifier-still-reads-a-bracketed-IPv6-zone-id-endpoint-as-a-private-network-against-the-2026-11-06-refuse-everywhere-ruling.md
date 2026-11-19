---
id: TASK-39137
title: "Console's Library egress classifier still reads a bracketed IPv6 zone-id endpoint as a private network, against the 2026-11-06 refuse-everywhere ruling"
status: To Do
created_date: 2026-11-07 02:03
labels:
- follow-up
- console
- providers
- library
priority: low
references:
- tldw_chatbook/Chat/console_library_destination.py
- Tests/Chat/test_console_library_destination.py
dependencies:
- TASK-39041
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07. TASK-39041's notes (round-11 branch fu/r11-settings) record this module as out of scope. Line numbers are dev's.

**The ruling.** Owner decision (2026-11-06), quoted from TASK-39041's notes: "Refuse everywhere: Discover refuses it too, matching the save, the contract and TTS; the #1562 pin that preserves zone ids is rewritten on purpose. Zone ids are link-local only and no task or ADR chose to accept them." TASK-39041 applies it to Discover, the save, display and compare.

**Where it does not hold yet.**
- `resolve_console_destination` accepts a scoped literal through `ip_address(hostname)` (`_parse_endpoint_origin`, `console_library_destination.py:235`, `:272`).
- It then classifies it `PRIVATE_NETWORK` (link-local, `:358-359`), keeping the zone in the identity (`_origin_identity`, `:332`).
- A pure probe on dev:
  - `http://[fe80::1%25en0]:5001/v1` gives `PRIVATE_NETWORK`, identity `http://[fe80::1%25en0]:5001`;
  - the raw `%en0` form gives the same.
- The gateway resolves this destination for every send resolution (`console_provider_gateway.py:4598-4601`).
- `Tests/Chat/test_console_library_destination.py:83-87` pins the contradicted behaviour. No PR lane runs that file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A bracketed zone-id endpoint, `%25` or raw `%`, gets the classifier's result for an endpoint it cannot use (`UNKNOWN`, `external/unknown`), like other unparseable endpoints; a test is red on dev and runs on a PR lane
- [ ] #2 The `:83-87` pin is rewritten on purpose and named in the notes
- [ ] #3 Link-local and ULA literals without a zone are still classified `PRIVATE_NETWORK`
<!-- AC:END -->
