# PR2734 review follow-up

Addressed all three Qodo findings on `8b34eaf4e4dcecf34d61e8d09f2b35462aaee14e`.
The owner approved the gallery and continuation. No executable application or
visual-style change was needed.

- **CLI arguments and output root:** reuse the parent PR's reviewed
  `native_runner_args.py`. It checks exact argument count, bounded tmux names,
  canonical temporary-profile containment, existing private home/config/data
  directories, config/database containment and absence of prior output before
  app startup. The existing runner already checked config/database containment;
  the new boundary additionally protects the root and old evidence. Shared
  input/path validators are used, with explicit trailing-whitespace rejection.
  tmux is discovered on PATH and main documents its CLI, outputs and exits.
- **Public method documentation:** show_permission documents every argument,
  including the identity check after locking, unconditional selection for
  expected_view=None and preservation of the reviewed fingerprint on guarded
  completion. retry_permission_action documents that None or an obsolete token
  does nothing and a matching token restores the cached reviewed view.

## Evidence

Seven malformed child-runner invocations [failed before the change](red-001.txt).
[All 28 runner cases pass](green-001.txt), covering both entry points and the
shared parser's safe-path, existing-output, missing-tmux and newline boundaries.
[Independent review](independent-review.txt) found no remaining blocker.
[AST comparison](executable-equivalence.txt) confirms the inspector's executable
tree is unchanged by the parameter documentation. [Ruff](static-analysis.json)
adds no diagnostics and all [seven derived guards pass](preflight.txt).

The approved stack was also merged locally with dev `d7537cbb65` on temporary
branch `codex/mcp-permission-stack-integration`: [all 56 targeted permission
tests](current-dev-integration.txt) and [seven derived guards](current-dev-preflight.txt)
passed. That merge required no conflict choices. Subsequent changes are the QA
boundary, its tests and method docstrings. No full sweep was run.

[Fresh native replay](native/result.json) passed four theme/size cells and twelve
captures through the real stores/service. [Lifecycle checks](lifecycle.json)
confirm exit 0, absent process, released lock, ten healthy private databases,
zero conversations/messages, unchanged defaults and matching source hashes.
The original approved gallery/receipts remain historical evidence from
`8b34eaf4e4dcecf34d61e8d09f2b35462aaee14e`; their runner hash is not rewritten.

Qodo marked all three findings resolved on `8671408d15`, with zero remaining
bugs or rule violations. PR2731's four findings are also resolved on `edc59b2094`.
PR2734 now targets dev to run its required checks in parallel, but PR2731 must
merge first; after that merge the parent changes disappear from this PR's diff.
Current-head CI and accumulated review remain merge gates. CodeRabbit skips
these base branches; its success status is not code-review clearance.
