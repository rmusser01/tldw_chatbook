---
id: TASK-2526
title: tiktoken is not a declared dependency
status: To Do
assignee: []
created_date: '2026-08-06 02:22'
labels:
  - packaging
  - tokens
  - cleanup
dependencies: []
priority: low
---

## Description

`Utils/token_counter.py`'s `estimate_tokens()` (`Utils/token_counter.py:137-161`, the function PR-T2 Task 5
wired the Console token estimator through, replacing a char-ratio placeholder) prefers `tiktoken` when
available, falling back to a conservative character-count floor when it isn't. The `tiktoken` import is a bare
`try/except ImportError` (`token_counter.py:31-39`) with no corresponding entry anywhere in `pyproject.toml` —
not in the base dependencies, not in any extra (`embeddings_rag`, `websearch`, etc.).

This means a real, unmodified install of `tldw_chatbook` — not just a minimal/CI environment — very plausibly
never has `tiktoken`, making the conservative chars-floor tier the common production case rather than a rare
degraded fallback. PR-T2 Task 5's own review flagged this: the dev venv used to write and review that task's
tests had neither `tiktoken` nor a working custom tokenizer, so the new tests exercise `model`/`provider`
threading through the *floor* tier's per-message framing overhead, not real subword tokenization — the "swap
the placeholder for the real tokenizer" story is only as true as `tiktoken`'s actual availability in a given
install.

## Acceptance Criteria

- [ ] Either `tiktoken` is declared as a dependency (in base `dependencies` or in a suitable extra, e.g. bundled
      with `embeddings_rag` or a new lightweight extra) so a normal install gets real subword token counts, OR
      the user-facing token/cost estimates are documented as approximate-by-default when `tiktoken` is absent
      (e.g. in the relevant `Docs/User_Guide` page and/or a startup-time log note)
- [ ] If declared as a dependency, `Tests/` covering `estimate_tokens`/`count_tokens_tiktoken` are re-run to
      confirm the tiktoken-tier code path is actually exercised in the standard dev install, not only the floor
      tier
- [ ] No behavior change to the existing floor-tier fallback for environments that still lack `tiktoken` (e.g. a
      minimal extra was chosen that some install profiles still opt out of)
