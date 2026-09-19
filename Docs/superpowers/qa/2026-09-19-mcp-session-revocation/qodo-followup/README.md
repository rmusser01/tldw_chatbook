# PR2731 review follow-up

Reviewed Qodo's four findings on `6de5273903b11557022147d392e67e487d6dc959`.
No application code, permission policy or approved layout changed.

- **CLI validation:** exact argument count now uses argparse. Bounded tmux
  identifiers reuse `input_validation.validate_username`, with an additional
  whitespace rejection because that helper's regex accepts a final newline.
  Reusing the existing string validator avoids adding a QA-only Pydantic model
  to the production input module. Invalid commands exit 2 with usage.
- **Output root:** shared canonical-directory and path validators require an
  existing dedicated child of canonical `/tmp` (`/private/tmp` on this Mac).
  Prepared home/config/data directories must be canonical and contained. The
  existing config/database containment validator remains; it already rejected
  escaping database paths before this review. Existing native.log, launch.json
  or evidence entries, including dangling symlinks, are rejected before writes.
- **Runner contract:** main now has a return annotation and documents its CLI,
  prepared profile, output files and exit statuses.
- **Portability:** tmux is discovered on PATH before startup. Image warmup uses
  the application's supported helper; this environment's textual-image does
  not export the runner's previous private probe_terminal function.

## Evidence

[Seven malformed CLI cases failed before the fix](red-001.txt).
Independent review found the trailing-newline exception; [both reproductions
failed](red-002.txt). [All 21 boundary cases now pass](green-002.txt), covering
valid prepared profiles, outside/aliased/traversing roots, escaping home/database
paths, reused evidence, missing tmux and both trailing-newline arguments.
Pytest's old temporary-directory cleanup warnings are unrelated to these cases.

[Native replay](native/result.json) completed all four cells and eight captures
with real revocation and runtime-gate checks. [Lifecycle checks](lifecycle.json)
confirm exit 0, absent process, released lock, ten healthy private databases,
zero conversations/messages, unchanged defaults and matching application source
hashes. The replay preceded the final rejection-only newline guard; the 21-case
run verifies that final guard. Valid native arguments are unchanged.

The original approved gallery and receipts remain historical evidence from
`6de5273903b11557022147d392e67e487d6dc959`; their runner hash is intentionally
not rewritten. This directory contains the subsequent replay with the revised
runner. No full test sweep was run.

[All seven derived-artifact guards pass](preflight.txt); the initial sandbox run
could not download the pinned Mermaid input, and the network-enabled retry
completed normally. [Static checks](static-analysis.txt) and [independent
review](independent-review.txt) pass.
