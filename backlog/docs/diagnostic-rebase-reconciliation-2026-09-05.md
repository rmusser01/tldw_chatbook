# Diagnostic boundary reconciliation after the second dev rebase

TASK-18801; prior evidence: 322 passed / 5 failed in
`/private/tmp/tldw-rebased-diagnostic-qualification.xml`.

Compared `codex/dev-test-review-before-second-rebase-20260905` with the review
tree at `5ef60d818f`, rebased onto dev `53194eee674865bd8b4aa6daac4b1e7d97160594`.
All **584 previous owner rows are unchanged**. The exact additions are:

| Owner | Calls | Diagnostic digest |
| --- | ---: | --- |
| Audio/meeting_capture.py | 7 | be82d53091d5bd1d646c |
| Audio/meeting_owner.py | 4 | 663d7f140f078cc4d6e2 |
| Audio/meeting_session.py | 7 | f55309e92202597924a9 |
| Audio/system_audio_tap.py | 9 | b8e42bd415e2f4a6f7a3 |
| UI/Screens/meetings_screen.py | 3 | 27f964117a07593a82e2 |

All five sources, plus `LLM_Management/snapshot_store.py`, match `origin/dev`
byte-for-byte. The diagnostic scanner's `--statements ... --since` report was
read for all six files. The Meetings calls retain their upstream TASK-31551
classification: fixed messages, numeric metadata, exception text, and some
path-redacted details. **This reconciliation does not classify those exception
diagnostics as metadata-only or certify their privacy.** It preserves the
upstream domain boundary rather than excluding those rows from the hash.

The twelfth sink file is `snapshot_store.py`, with two private stream-opening
sites: `_locked` opens `catalog.lock` for a checked lock; `stage_restore` copies
validated retained snapshot bytes into a reserved working file, checking file
identity, length and digest. These are storage openings, not logger additions;
the scanner includes them in its persistent sink topology. The source contains
no diagnostic calls. Existing sinks, classification rules and owner rows are
unchanged except the added TASK-31551 rule and derived summary counts. All other
inventory sections, including path candidates, are unchanged.

An independent normalization reproduced the previous pinned hash
`ac5cd5bf7bc9d5f35d80fd71a78953ea96cd6cc60fced84b2e4f60c332bc04f1` from the saved
pre-rebase manifest. The current checked manifest equals a fresh scanner rebuild;
both normalized hashes are
`caa76e94acdbf3d61961e90bcfe307c21ed5c09bfc061d7c054ba497ff817684`.
Normalization masks only the same two summarization owners' call counts/digests
and their derived TASK-492 count; the test implementation was not changed.

The repair replaces only the two boundary hashes and adds the upstream
`task_31551_calls: 0` field to the virtualenv-exclusion fixture's exact summary.
The fixture still requires exactly its application owner and sink, excluding
both conventional and arbitrarily named nested environments. No inventory
regeneration, classification change, new exclusion, ceiling increase or negative
mutant relaxation is part of this repair.

ADR required: no. ADR path: N/A. This is governed evidence reconciliation of
existing diagnostic contracts, not a new runtime or security boundary.

## Verification

Both complete files passed **327 tests in 409.28 seconds**, including the
unchanged drift, digest-schema and unreconciled-digest negative controls.
Evidence: `/private/tmp/tldw-rebased-diagnostic-repaired.xml`. The 17 warnings
are dependency compatibility/deprecation and existing source escape warnings,
not a warning-free suite claim. Scoped Ruff and changed-range formatting pass;
the architecture file retains unrelated pre-existing formatting drift.
Independent scoped review reproduced both hashes and confirmed the unchanged
old owner rows and upstream source identities, with no remaining findings.
TASK-18801 stays In Progress because its clean-origin/dev criterion requires
upstream integration; this is verified draft-branch evidence only.
