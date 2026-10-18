# ADR-238: Offline authenticated Canvas build inputs

Status: Accepted
Date: 2026-10-17
Related tasks: TASK-32380, TASK-32377
Related decisions: ADR-121, ADR-124

## Context

Required CI and local preflight rebuild the immutable Mermaid profile using
eight size/hash-pinned upstream inputs. Their default path downloads those
inputs for every check. An upstream outage therefore prevents verification
of an unchanged checkout. An optional external cache does not make a fresh
checkout reproducible offline.

## Decision

Commit the eight original inputs declared by `Canvas/mermaid/inputs.json`
under repository-level `vendor/canvas-mermaid-inputs/`. They total 19,636,664
bytes. Preserve the complete original Mermaid archive, Unicode source data,
conformance data, rules and license instead of creating another source format.
These are build inputs, outside application package data; installed Canvas
continues using its existing packaged runtime closure.

Normal artifact verification reads this directory and never fetches missing
inputs or skips reproduction. An explicit `--input-dir` or the existing
`TLDW_CANVAS_MERMAID_INPUT_DIR` may select another directory, subject to the
same declaration, size, SHA-256, archive, grammar and builder checks. Missing
inputs, stale bytes and generated-output mismatches remain distinct errors.

Refreshing inputs is an explicit maintainer action: a fetch-only checker
command downloads the declared inputs to a new directory and validates each
before accepting it. It does not modify generated assets, hash pins, admission
policy or existing input directories. Updating semantic inputs still requires
a new immutable profile and ADR-124 qualification.

The existing independent vendored-asset pins also cover the trusted shell's
JS, CSS and HTML. Rebuilding generated files cannot update these pins.

## Alternatives

- A hosted cache avoids some downloads but fails on a cache miss or a fresh
  offline checkout. Keep it optional rather than a prerequisite.
- Skipping unavailable inputs makes a required integrity check green without
  proving reproduction. Reject this option.
- Extracting only selected archive members reduces repository size but creates
  a second input format and discards the original archive-inventory proof.
  Retain the already validated source format.

## Consequences and verification

The repository gains about 19 MB of third-party source inputs, without adding
an application dependency or installed runtime payload. Original licenses and
upstream URLs remain available. CI and preflight use the same default path.
Tests must exercise real offline reproduction with network access refused,
missing/corrupt input refusal, shell tamper refusal and explicit refresh
validation. All generated profile and runtime bytes must remain unchanged.
