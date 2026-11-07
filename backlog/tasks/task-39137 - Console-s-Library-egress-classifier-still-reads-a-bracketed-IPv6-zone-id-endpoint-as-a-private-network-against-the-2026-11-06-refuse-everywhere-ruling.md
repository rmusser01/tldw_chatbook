---
id: TASK-39137
title: "Console's Library egress classifier still reads a bracketed IPv6 zone-id endpoint as a private network, against the 2026-11-06 refuse-everywhere ruling"
status: Done
assignee:
- '@claude'
created_date: 2026-11-07 02:03
updated_date: 2026-11-07 06:10
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
- [x] #1 A bracketed zone-id endpoint, `%25` or raw `%`, gets the classifier's result for an endpoint it cannot use (`UNKNOWN`, `external/unknown`), like other unparseable endpoints; a test is red on dev and runs on a PR lane
- [x] #2 The `:83-87` pin is rewritten on purpose and named in the notes
- [x] #3 Link-local and ULA literals without a zone are still classified `PRIVATE_NETWORK`
<!-- AC:END -->

## Implementation Plan

1. Tests first: rewrite the `:83-87` row of `Tests/Chat/test_console_library_destination.py` on purpose, add the raw `%` form and a zone-less link-local row; add a laned test in `Tests/Chat/test_local_server_discovery.py` (the Dreams boundary step already runs that file). Show both red on base.
2. Fix: `_parse_endpoint_origin` refuses a host the contract's `canonical_endpoint_host` refuses, so a zone id takes the unparseable path (`UNKNOWN`, `external/unknown`).

## Implementation Notes

Owner decision (2026-11-06): "Refuse everywhere: Discover refuses it too, matching the save, the contract and TTS; the #1562 pin that preserves zone ids is rewritten on purpose. Zone ids are link-local only and no task or ADR chose to accept them."

- **Fix.** `console_library_destination._parse_endpoint_origin` now returns None when `canonical_endpoint_host(hostname)` (`Provider_Contract/provider_endpoint_contract.py`, the one host rule the save, Discover, display and compare already use) refuses the host. It replaces the local `len(hostname) > 253` check, which the contract also makes. A zone id, `%25` or raw `%`, now gets `UNKNOWN` / `external/unknown`, like every other endpoint the classifier cannot use. No new module at boot: the contract module is already in `boot_import_modules.txt`.
- **AC#1.** `Tests/Chat/test_local_server_discovery.py::test_console_egress_reads_a_zone_id_as_an_endpoint_it_cannot_use` (4 cases). That file runs in the "Run Dreams and Guardian boundary contracts" step, so no workflow or lane pin changed. Red on base: both zone cases `assert <PRIVATE_NETWORK> is <UNKNOWN>`; green after.
- **AC#2.** The pin at `test_console_library_destination.py:83-87` (`test_resolved_destination_classifies_only_provable_endpoint_evidence[http://[fe80::1%25en0]:8080/v1-...]`) is rewritten on purpose to `UNKNOWN` / `external/unknown`, with a comment naming this task, and a raw `%en0` row is added. Both were red on base the same way.
- **AC#3.** Zone-less `[fe80::1]` and `[fd12:3456::9]` stay `PRIVATE_NETWORK` with their own identity (new rows in both files; the ULA and `169.254.2.3` rows already existed).
- **Runs** (run.sh, `-n 4`, load 30-39): the two files, 423 passed. Covering files that import the classifier or its doubles (7 Chat files, `Tests/Utils/test_egress.py`, two `LLM_Provider_Catalog` files, `test_console_store_continuity.py`, `test_console_native_chat_flow.py`): head 290 failed / 766 passed, `fu-base-r12` 291 failed / 765 passed. The same nodes fail on both, apart from one base-only flake (`test_console_delete_confirmation_resets_when_selection_changes`); the reds are the known plain-UI `RecoveryRequired` setup failures and pre-existing Chat reds.

Files: `tldw_chatbook/Chat/console_library_destination.py`, `Tests/Chat/test_console_library_destination.py`, `Tests/Chat/test_local_server_discovery.py`.
