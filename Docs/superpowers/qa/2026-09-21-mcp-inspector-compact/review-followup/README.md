# PR2770 validation review follow-up

The owner approved the compact inspector layout at `890f10de38` on 2026-09-21.
Qodo found that the native QA launcher bypassed the shared input-validation
boundary after its obsolete username helper was removed. The launcher now calls
`validate_tmux_identifier`, a strict Pydantic adapter in that shared module.
It preserves the existing 1–64 ASCII alphanumeric/underscore/hyphen contract,
requires a leading letter or digit, rejects coercion and anchors the whole input.
No UI, tool authority, path admission or profile behavior changes.

- [153 passing boundary/admission/fixture cases](validation.txt), including 19
  new direct cases for valid boundaries, tmux syntax, Unicode and newlines.
- [Eight passing app import checks](import.txt); existing module-census warning.
- [Eight artifact guards pass in the first run](preflight-initial.txt); the
  Mermaid check could not download its inputs under sandbox restrictions.
  [The network-enabled rerun passes](mermaid-recheck.txt), completing all nine.
- No introduced Ruff findings; new/edited ranges are formatted. The shared
  module retains its eight baseline diagnostics.
- [Source comparison](provenance.json) confirms all existing shared validation
  AST and all other captured sources are unchanged. The original native
  receipts retain their historical hashes; this follow-up does not claim a
  fresh native run or replace the approved captures.

Independent review found that compiled patterns were incompatible with the
oldest supported Pydantic. The final adapter uses a plain Rust-regex string
with absolute anchors instead, preserving the dependency floor. The strict
adapter and wrapper are the only executable changes since visual approval.
[The oldest supported Pydantic 2.4.2/core 2.10.1 passes all 19 cases](pydantic-2.4.2.json)
in an isolated Python 3.12.11 environment against the actual validator AST. The
earlier compiled-pattern failure reproduced there. Independent final review is
clear. This isolated qualification does not claim a full app run on that dependency.
No new ADR: this implements the existing shared-validation boundary.
