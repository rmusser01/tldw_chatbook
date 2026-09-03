---
id: TASK-25906
title: 'Doctor: aggregate health-check surface'
status: Done
assignee: []
created_date: '2026-08-31 15:09'
updated_date: '2026-09-02 00:32'
labels:
  - ops
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Chatbook is local-first, so the user owns the whole stack - local inference servers, optional extras, API keys, DB integrity, config validity - and today has no way to ask what is broken. Verified on origin/dev: a named grep for doctor, healthcheck, diagnostic and self_test across tldw_chatbook, Packaging and scripts returns only a log sink (Utils/persistent_diagnostics.py), TTS abbreviation expansions, and TTS supervisor health intervals; the nearest surface is Settings > Diagnostics, which parses and reloads the TOML (UI/Screens/settings_screen.py:9030-9042). Every ingredient already exists and is separately proven - this task is aggregation and presentation, not new probing logic.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A single surface reports pass/fail per check with a reason, covering at minimum: config load status, optional-dependency availability, DB integrity, configured-provider readiness, and private-path posture
- [x] #2 Each check reuses the existing implementation rather than reimplementing it (Utils/optional_deps, DB/base_db.check_integrity, config.get_config_load_failure, config.get_detected_api_providers)
- [x] #3 Checks that require network calls are opt-in and clearly labeled, never run by default
- [x] #4 A failing check names the specific remediation where one is known, and says so honestly where none is
- [x] #5 No secret values appear in the output; provider readiness reports configured/not-configured, never the key
- [x] #6 The surface is reachable without a working config - it must still run and report when config load has failed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pure Utils/doctor.py: DoctorCheck + one check per area, each injectable\n2. Each check reuses the existing impl (optional_deps, check_integrity, get_config_load_failure, get_detected_api_providers, path posture)\n3. run_doctor isolates each check; network opt-in; format_doctor_report worst-first\n4. Surface as /doctor command; DB PRAGMA offloaded to a thread\n5. TDD incl. no-secret + runs-without-config
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Aggregation + presentation over checks that already ship; no new probing logic.

Approach (Utils/doctor.py, new):
- DoctorCheck(name, status, detail, remediation). Five checks, each pure + injectable: check_config_load (config.get_config_load_failure), check_optional_dependencies (optional_deps.DEPENDENCIES_AVAILABLE), check_database_integrity (config.get_chachanotes_db_lazy().check_integrity -- the existing PRAGMA integrity_check), check_provider_readiness (config.get_detected_api_providers -- NAMES only, never a key), check_private_path_posture (os.stat owner-only bits on the data/config dirs) -- AC#2.
- run_doctor runs each in isolation so one failure can't abort the rest (AC#6) and gates network probes behind include_network=False (none implemented; the gate exists so doctor can never phone home by default -- AC#3). format_doctor_report renders worst-status-first with remediations (AC#1/#4). Provider readiness reports configured/not-configured, never a key (AC#5).
- Surfaced as the /doctor console command (grammar + description + dispatch); the DB PRAGMA is offloaded via asyncio.to_thread so a large DB doesn't stall the loop. /doctor network opts into (future) network probes. Live-verified: real report shows config/database/private-paths PASS, optional-deps/providers WARN, 0 secrets.

Tests: Tests/Utils/test_doctor.py (7: each check pass/fail, no-secret, runs-when-config-failed, network-skipped-by-default, report format). Updated suggestions COMMANDS for /doctor.

Files: tldw_chatbook/Utils/doctor.py (new), tldw_chatbook/Chat/console_command_grammar.py, tldw_chatbook/Chat/console_command_suggestions.py, tldw_chatbook/UI/Screens/chat_screen.py, Tests/Utils/test_doctor.py, Tests/Chat/test_console_command_suggestions.py.
<!-- SECTION:NOTES:END -->
