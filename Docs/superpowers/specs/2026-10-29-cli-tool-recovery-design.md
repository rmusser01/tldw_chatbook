# CLI tool recovery

The user authorized fixing all reproduced CLI defects and clarified that only
the 4,000-character model-result cap should be removed. Executor stream and
configured storage limits remain in force.

## Behavior

- Preserve the complete executor-provided stdout/stderr in shell tool results.
- A simple `grep`, `rg`, or `findstr` command that exits 1 without stderr is an
  empty search, not an ordinary tool failure. Genuine errors and compound or
  otherwise ambiguous shell expressions retain existing failure handling.
- Preserve exact command text through Windows `cmd /D /S /C` invocation.
- Virtual `ls` accepts `-a`, `-A`, `--all`, `--almost-all`, `-l`, and combined
  forms such as `-la`. The existing listing includes hidden entries; all flags
  preserve that behavior. Long listings add type/permissions and size to each
  admitted entry. All virtual command usage is disclosed in the model schema,
  and rejected arguments include the relevant usage.
- Windows atomic file writes reuse native retained directory-handle operations,
  including exclusive creation, CAS checks, atomic replacement and cleanup.
  POSIX and shipped stdlib worker behavior remain compatible.

## Approaches and decisions

Use existing executor/provider seams and native file primitives. Do not add a
shell dependency, generic retry override, or a second permission system.
Classifying every exit 1 as success would hide genuine errors; conservatively
recognize only simple known search commands. Raising retry thresholds would
leave the output/quoting/write defects intact. Implement documented virtual
flags rather than silently discarding unsupported flags.

These implementation details are agent choices within the requested fixes.
The reporter-specific cause remains unconfirmed; regression evidence addresses
the independently reproduced defects.

## Verification and governance

Focused red/green tests cover output preservation, search/error controls,
quoted Windows commands, virtual parser/worker parity, native writes/CAS and
confinement. No full suite without user opt-in. A fresh reviewer checks the
combined patch. No push, PR or merge is required by this request.

ADR required: yes.
ADR path: backlog/decisions/254-cli-search-outcomes-and-listing-compatibility.md.
Reason: refine command outcome classification and extend the closed worker's
optional read-only listing arguments under ADR-078/094/101.
