# Console Watchlists workflow QA evidence

This directory contains the redacted evidence bundle for TASK-22868.

Current state:

- deterministic Console/Watchlists QA: green, 3/3
- external MCP metadata/receipt boundary: green
- local skill/framework and single-flight regression: green
- First Run broad selection: 136 passed, two order-sensitive failures; both exact nodes pass in isolation on the current and pre-task trees
- normal 180×50 and compact 160×42 Textual captures: visually reviewed after two craft passes
- capture hashes: recorded in `evidence.json`
- comprehensive redaction scan: passed, 12 files scanned with zero matches
- refreshed `origin/dev` reconciliation and rerun: pending

Files:

- `evidence.json` — machine-readable scope, results, receipts, and revision state
- `automated-transcript.txt` — body-redacted public-seam workflow transcript
- `redaction-scan.txt` — final zero-match scan record
- `capture_uat.py` — disposable-profile production-shaped capture script
- `console-{180x50,160x42}.svg` — completed source-check and briefing receipt cards
- `watchlists-{180x50,160x42}.svg` — exact completed briefing plus stored every-24-hours schedule
- `library-skill-classification-{180x50,160x42}.svg` — generic framework classification and recovery guidance

Private fixture bodies, secrets, API keys, home-directory paths, raw permission bodies, and the briefing-only sentinel are excluded. The UAT report records the sentinel assertion without reproducing the sentinel value.
