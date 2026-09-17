# RAG answer keyboard reading — TASK-32714

2026-09-17 UTC, `feat/component-pattern-library`, based on `b4a085de18`.

The existing keyboard route reaches generated answers and citation feedback:
Tab from the query through Run and available source toggles to an evidence
card, then Page Up to read the Answer region above it and Page Down to return.
The guide now explains this route and removes its fixed “Five Tabs” claim.
No production code, bindings, widgets, styles or tokens change.

ADR required: no. Existing
[ADR-150](../../../../backlog/decisions/150-design-token-system-and-design-language.md)
and [ADR-031](../../../../backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md)
govern the unchanged UI.

## Automated evidence

- [Navigation tests](navigation-tests.txt): **24 passed in 164.42s**. Production
  CSS, 170×48 / 80×24, dark / light, 2 / 80 answer lines, and uncited / invalid
  marker / validated reference states. Each case reconstructs final compositor
  rows separately while paging up and down, requiring the entire answer and
  exact citation feedback. Query, source choices, answer identity and card focus
  persist; retrieval and generation each run once. Tab still reaches a painted
  evidence action afterward.
- [Answer-service tests](answer-service-tests.txt): **65 passed in 2.05s**.
- Ruff lint/format and whitespace checks pass for the new test and native runner.
  Independent read-only review found no code issue; its stale pending-status
  documentation finding was corrected and confirmed resolved. No whole-suite run was requested or performed.

```sh
.venv/bin/python -m pytest Tests/UI/test_library_rag_answer_navigation.py \
  -q -x --tb=short --show-capture=no
.venv/bin/python -m pytest Tests/Library/test_library_rag_answer_service.py \
  -q --tb=short --show-capture=no
```

The [initial uncited probe](probe-uncited.txt) passed eight cases. An expanded
[probe](probe-page-boundary.txt) was stopped after 7 passed / 7 failed because
it incorrectly required each wrapped sentence to fit a single frame. Compact
pages can split a sentence; all painted rows are now collected by their offset
within the answer widget. This was a test assumption, not a reproduced product
failure. Probe counts overlap the final matrix.

## Native evidence

[Result](result.json), [lifecycle](lifecycle.json), and
[capture hashes](capture-hashes.json) record the successful final profile,
`/private/tmp/tldw-32714-run-003`. The [runner](native_check.py) uses real TldwCli, an owned tmux terminal, an exclusive private profile, real Media
storage and workspace membership. It adapts RAG requests to real local keyword
retrieval and controls only the synchronous answer-provider seam. The real
answer service builds prompts and validates citation labels. Setup between
independent cells explicitly focuses and reveals the query; Tab, Page Up,
Page Down and the final Tab drive the qualified reading route. Answer reading
itself never uses programmatic scrolling.

All twelve native 80-line journeys pass: both themes, both sizes, and all
three citation states. Twelve real local keyword searches and twelve controlled
provider calls complete; reading adds none. The exact stored Media ID/title is
returned, and all answer/citation text is painted in both scroll directions. Short-answer coverage is mounted only. Controlled replies
and keyword source labels do not qualify real-provider behavior, semantic
retrieval or factual grounding. Validated citations establish reference
resolution only.

[Attempt 001](run-001.json) completed one wide dark case, then failed its setup
assumption that `/` would return to the RAG query; that key focused Library's
rail search. [Attempt 002](run-002.json) completed all three wide dark cases,
then failed its reverse-Tab setup after resizing. Neither failure occurred
inside the qualified answer-reading route. Both apps quit normally with shell
status 1; their exact PIDs were absent before owned-terminal cleanup. The final
runner resets the starting query viewport explicitly between independent cells.
Keyboard return to the query across resize is outside this verification.


| Theme/size | Uncited warning | Invalid-marker warning | Validated note | Answer tail |
| --- | --- | --- | --- | --- |
| Dark 170×48 | [View](textual-dark-170-uncited.svg) | [View](textual-dark-170-unverified.svg) | [View](textual-dark-170-validated.svg) | [View](textual-dark-170-answer-tail.svg) |
| Dark 80×24 | [View](textual-dark-80-uncited.svg) | [View](textual-dark-80-unverified.svg) | [View](textual-dark-80-validated.svg) | [View](textual-dark-80-answer-tail.svg) |
| Light 170×48 | [View](textual-light-170-uncited.svg) | [View](textual-light-170-unverified.svg) | [View](textual-light-170-validated.svg) | [View](textual-light-170-answer-tail.svg) |
| Light 80×24 | [View](textual-light-80-uncited.svg) | [View](textual-light-80-unverified.svg) | [View](textual-light-80-validated.svg) | [View](textual-light-80-answer-tail.svg) |

All sixteen captures were rendered and inspected. Warnings wrap readably in the
compact panel; neutral reference-resolution notes stay distinct from cautions,
and the answer's final line and citation marker remain readable. Paging can
split a wrapped sentence across successive viewports, as the row-coverage checks
allow. No visual change was necessary.

All ten private databases pass read-only quick_check; no conversation messages
were created and the source record is unchanged. Default config/UI/runtime file
fingerprints match across all three attempts. The successful run has no error,
critical or unhandled-exception log entries, both faulthandler logs are empty,
normal stopping is logged, app.run returns and the shell reports zero. Its exact
PID was absent before terminal cleanup. The executed runner hash matches the
stored file; SVG normalization only removes trailing whitespace.

[Independent review](review.json) has no outstanding findings. The task-ID collision
check covered 310 refs and 27 worktrees; only this worktree owns TASK-32714.
No full suite, push or merge was performed. The next bounded review is query
return and resize transitions, whose setup assumptions were not qualified here.
