# Console Watchlists workflow QA evidence

This directory is the redacted, disposable-profile evidence bundle for
TASK-22868. It makes three different evidence claims and does not conflate them.

## Evidence taxonomy

- **Service round trip:**
  `Tests/QA/test_console_watchlists_workflow_uat.py` drives the public Console
  agent bridge through real local Watchlists tools, SQLite services, durable
  receipts, briefing persistence, external-MCP publication, and local
  skill/framework fixtures.
- **Mounted composition/navigation:**
  `Tests/UI/test_console_watchlists_mounted_uat.py` mounts the real application,
  `ConsoleChatController`, app-owned provider composition, visible approvals,
  Watchlists, Settings, and Library. Scripted model/feed fixtures prevent public
  egress. `mounted-console-{180x50,160x42}.svg` are emitted by that run.
- **Seeded rendering fixtures:** `capture_uat.py` mounts production Textual
  screens with deterministic seeded state for focused responsive/HCI review.
  The six Console, Watchlists, and Library SVGs it writes are not evidence that
  the Console tool loop ran.

## Current results

- external MCP metadata/receipt-only boundary: green
- complete QA contract: 3 sandbox-safe tests passed; the disposable loopback
  round-trip passed separately with local-bind permission
- persisted no-preset briefing provider/model resolver: manual + scheduler green
- confirmed schedule writes remain successful and request reload when that
  persisted route is unavailable; the receipt exposes fixed configuration
  attention instead of a false storage failure
- all eleven actionable Qodo review findings are covered: coordinated briefing following is
  bounded; briefing-setting writes preserve dispatch order; subscriptions v2
  DDL ships as a migration artifact under the shared transaction owner; DB
  readiness uses real SQLite variants; redaction/capture paths are centrally
  confined; policy denial is non-retryable; and capture sandboxes are removed
  on render or cleanup failure. Follow-up corrections centrally validate the
  test identifiers, context-own the temporary SQLite connection from
  acquisition, and use one tested revision across every evidence surface
- mounted approval, durable-receipt, navigation, and briefing-consumption loop:
  green at 180x50 and 160x42
- local skill/framework and TASK-613 single-flight regressions: green
- changed Library skill/import files: 204 passed
- First Run prerequisite: 138 sandbox-safe tests passed; the only two
  sandbox-blocked loopback peer tests passed with local-bind permission
- reproducible fail-closed, repository-confined redaction checker: committed;
  final zero-match record in `redaction-scan.txt`
- independent round-three review: approved at
  `2274046883ac513aca0c3960504b945cbdef1110`, with no remaining findings
- latest observed and merged `origin/dev`:
  `b1ada0fba2cafe4aee34441926ee96e036ccef55`; reconciled code HEAD
  `25be18705ec897596e61c0cebfe20814157b6530` passed the recorded rebase and
  review-fix gates

## Files

- `evidence.json` — machine-readable scope, results, commands, hashes, and
  revision state
- `automated-transcript.txt` — body-redacted service and mounted workflow trace
- `redaction_check.py` — committed stable pattern classes plus mandatory
  out-of-band private proof input
- `redaction-scan.txt` — final fail-closed and zero-match scan record
- `capture_uat.py` — deterministic seeded rendering-fixture generator
- `mounted-console-{180x50,160x42}.svg` — actual mounted UAT Console captures
- `console-{180x50,160x42}.svg` — seeded completed receipt cards
- `watchlists-{180x50,160x42}.svg` — seeded briefing/schedule state
- `library-skill-classification-{180x50,160x42}.svg` — seeded generic-framework
  classification and recovery guidance

## Reproduce the privacy scan

Supply the real private proof value only through the environment; do not put it
in a command, shell history, or committed file:

```bash
export TASK22868_PRIVATE_SENTINEL
../../.venv/bin/python \
  Docs/superpowers/qa/console-watchlists-workflow-2026-08/redaction_check.py
```

Missing or shorter-than-16-character input exits 2. A finding exits 1. A clean
scan exits 0 and prints only the file count, never matched content.

Private fixture bodies, secrets, API keys, home-directory paths, raw permission
bodies, and the out-of-band redaction sentinel are excluded. No public network, live
user state, ATHF installation, hunt creation, or briefing-to-hunt handoff is in
scope.
