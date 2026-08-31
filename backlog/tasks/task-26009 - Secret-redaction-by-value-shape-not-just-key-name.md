---
id: TASK-26009
title: 'Secret redaction by value shape, not just key name'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:44'
updated_date: '2026-08-31 17:10'
labels:
  - security
  - mcp
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Redaction can be defeated by a value that does not sit under a recognized key. Verified on origin/dev: MCP/redaction.py:1-114 matches on key names, CLI argument names and URL query parameters only, and line 64 documents its own bypass - a secret value beginning with a dash survives. The exposure surfaces are the approval card (Widgets/Chat_Widgets/chat_approval_card.py:44) and the execution log. Hermes matches value shapes: provider key prefixes, JWTs, private-key blocks, database connection strings and bearer headers. This is a pure function with no caller changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A value matching a known secret shape is redacted regardless of the key or argument name it appears under
- [x] #2 Shapes covered include at minimum: common provider key prefixes, JWTs, PEM private-key blocks, database connection URIs and Authorization header values
- [x] #3 The documented dash-prefixed bypass at MCP/redaction.py:64 no longer applies
- [x] #4 Redaction is applied on both the display path (approval card) and the stored path (execution log) - verified separately for each
- [x] #5 False positives are bounded: ordinary prose, file paths and git SHAs are not redacted, asserted by tests
- [x] #6 Redaction remains a pure function with no new I/O or configuration dependency
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. Extends an existing pure function with additional patterns; no new dependency, storage, or seam.

1. Add value-shape patterns alongside the existing key-name matching, so a secret under an innocuous key is caught.
2. Anchor and length-bound every pattern. A 40-char git SHA, a file path and ordinary prose must not match -- over-redaction makes an approval card useless for deciding whether to approve, which is the failure mode worth guarding hardest.
3. Apply the shape check at all three existing entry points: mapping values, sequence items, and CLI arg tokens.
4. Close the documented dash-prefixed bypass by distinguishing a plausible flag name from a flag-shaped secret, keeping the "do not swallow a real flag" property the bypass existed to protect.
5. Verify AC#4's two boundaries separately rather than assuming a shared helper covers both.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added value-shape secret detection to `MCP/redaction.py` alongside the existing key-name matching, and closed the module's self-documented CLI bypass.

**The gap.** Redaction matched on key names, CLI flag names and URL query parameter names, so `{"note": "sk-live-..."}` reached the approval card untouched. `looks_like_secret_value` now matches ten credential formats regardless of the key: PEM private-key blocks, JWTs, prefixed provider keys (OpenAI/Anthropic/Stripe shape), GitHub classic and fine-grained tokens, Slack tokens, AWS access key ids, Google API keys, `Bearer`/`Basic` header values, and credentials embedded in a connection URI.

**False positives were the real design constraint.** An over-redacting approval card is useless for deciding whether to approve, so the patterns describe specific credential formats rather than "looks random". Ten innocent shapes are pinned as must-not-match, including a 40-character git SHA, absolute and relative file paths, a plain URL, an ISO timestamp, a version string and SQL. That test class is what keeps the patterns honest as they grow.

**The dash bypass (AC#3).** `redact_args` treated any token starting with `-` as a new flag, so `--api-key -9f3a...` appended the secret unredacted. That branch existed for a real reason -- `--api-key --verbose` must not swallow `--verbose` -- so it was not simply deleted. The test now distinguishes a plausible flag name (`^--?[A-Za-z][A-Za-z0-9_-]*$`) from a flag-shaped secret: `-9f3a...` is redacted, `--verbose` and `-v` still survive. Bare positional tokens are shape-checked too, since a secret can arrive with no flag at all.

**AC#4, and a finding worth recording.** The display path (`chat_approval_card._summarize_arguments`) inherits the fix through `redact_mapping`, verified by a test asserting the secret is gone AND the non-secret argument survives. The stored path needed nothing: `MCP/execution_log.build_record` is metadata-only by construction -- it documents itself as such and routes every field through `safe_metadata_token`, keeping argument NAMES and never values. That is a stronger guarantee than redaction, so the test pins the exclusion (asserting even the innocuous `/tmp/x` value is absent) rather than adding a `redact_mapping` call that would be dead code.

**Verification.** 53 tests in the new file; 212 pass across every approval-card and redaction test file. Two failures in `Tests/MCP/test_tools_resources_prompts_real_methods.py` are baseline, confirmed by re-running with the changes stashed.

One test-data correction during the work: the Google API key sample was longer than the fixed 39-character form, so an exact `{35}` quantifier plus `\b` could not match it. The quantifier is now `{35,}` rather than the sample being shortened -- a longer lookalike should not escape.

**Files:** `tldw_chatbook/MCP/redaction.py`, `Tests/MCP/test_redaction_value_shapes.py` (new).
<!-- SECTION:NOTES:END -->
