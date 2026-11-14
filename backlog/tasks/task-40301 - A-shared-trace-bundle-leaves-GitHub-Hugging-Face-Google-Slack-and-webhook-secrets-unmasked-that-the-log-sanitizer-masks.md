---
id: TASK-40301
title: "A shared trace bundle leaves GitHub, Hugging Face, Google, Slack and webhook secrets unmasked that the log sanitizer masks"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- security
- privacy
- trace
- export
priority: high
references:
- tldw_chatbook/Chat/trajectory_export.py
- tldw_chatbook/Utils/log_sanitizer.py
- Tests/Chat/test_trajectory_export.py
dependencies:
- TASK-39149
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (security review r13s-rev1 of TASK-39149). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** Trace v2 bundles are made to be shared. The export builds from `TrajectorySnapshot`, not from stored traces that credentials-v2 already sanitized, so its own patterns are the only mask. Today a GitHub PAT, a Hugging Face token, a Google API key, a Slack token or a webhook secret in a message or tool result ships verbatim in every profile, while the same text is masked in logs.

**The gap.**
- `trajectory_export.py:300-320` (`_CREDENTIAL_VALUE_RES`, from 794128caf7) covers PEM private keys, bearer and basic credentials, `sk-`, `[rs]k_(live|test)_`, `AKIA`, URL user info and `key=`-style assignments.
- `log_sanitizer.py:104-138` (`_STANDALONE_CREDENTIALS`) also covers `AIza`, `gh[pousr]_`, `github_pat_`, `hf_`, `xox[baprs]-`, the AWS `ASIA/AGPA/AIDA/AROA` prefixes and bare JWTs (`eyJ...`). The export has none of these.
- `whsec_` (and `gsk_`) is in neither list.
- Two hand-kept lists drift; TASK-39149 fixed the last gap the same way.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An exported bundle masks every credential shape the log sanitizer masks, in previews, payload values and keys, for all three export profiles; a test is red before the fix
- [ ] #2 `whsec_` and `gsk_` are masked in both places, or the notes name them out of scope with the reason
- [ ] #3 The shapes masked today stay masked, and the notes say what happens when an older bundle holding a newly masked shape is imported
<!-- AC:END -->
