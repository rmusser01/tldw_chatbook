# Ingest Worker Test Contract Repair

## Goal

Restore the full-suite baseline by updating one stale App test to the current
parse-worker request contract.

## Design

Production submits parse work as
`(source_path, options, (generation, job_id))`. The invalid-audio routing test
still unpacks the former two-item tuple, so all seven parameterizations fail
after the valid follow-up job is dispatched.

Update only that test to unpack the third value and assert it equals the
current pool generation and valid job ID. This retains the existing source and
provider assertions while adding evidence that the follow-up job is associated
with the correct generation. No production code, dependencies, or runtime
behavior change.

## Verification

Run the seven-case node first, then the complete App test module, Ruff and diff
hygiene, and the repository full-suite fail-fast gate.

ADR required: no

ADR path: N/A

Reason: this is a test-only correction to an existing call contract.
