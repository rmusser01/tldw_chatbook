# Diagnostic path privacy and regression guard design

**Task:** TASK-19864 (folds TASK-19936 into the same implementation)  
**Date:** 2026-08-28  
**Status:** approved

## Problem

Several production diagnostics interpolate user filesystem identities directly into
their message text: absolute file paths, workspace roots, database locations, and raw
exceptions whose messages repeat those paths. These records do not reach Chatbook's
metadata-only persistent file sink, but they do reach terminal output and the in-app
Logs pane. The user-selected **Copy visible logs** action copies those message bodies.

The task's verified remediation census names five owners:

- `tldw_chatbook/Utils/file_handlers.py`
- `tldw_chatbook/DB/ChaChaNotes_DB.py`
- `tldw_chatbook/UI/Screens/change_review_screen.py`
- `tldw_chatbook/Widgets/Console/console_conversation_inspector.py`
- `tldw_chatbook/Workspaces/git_workspace.py`

That historical census is not a trustworthy implementation checklist. Some calls have
changed since it was taken, multiline formatting defeats simple grep counts, and a
conservative current-dev lexical sweep produces 281 possible candidates in 85 files.
Treating those candidates as 281 proved leaks would conflate path-like code identifiers,
already de-identified values, actionable user-facing locations, and genuine raw paths.

The implementation therefore needs two distinct outcomes: completely remediate the
five recorded owners against current `dev`, and add a repository-wide guard that makes
new path-bearing diagnostic interpolation an explicit review event without pretending
the broader legacy candidate population has already been remediated.

## Goals

- Remove raw user path, workspace-root, and database-location values from every
  diagnostic in the five recorded owners.
- Preserve useful identity and failure context with path-free metadata.
- Route genuinely actionable full paths through the existing path-redaction seam.
- Fold TASK-19936's `change_review_screen.py` debug disclosure into this work.
- Extend the existing diagnostic-inventory scanner so new path-shaped interpolation is
  detected and all findings are reported in one run.
- Keep the Logs screen's sharing disclosure accurate.

## Non-goals

- Treat the 281 lexical candidates outside the five recorded owners as 281 validated
  vulnerabilities or silently mark them reviewed-safe.
- Rewrite every path-bearing diagnostic in the package in one PR.
- Add a sink-level blanket path remover, logger wrapper, dependency, or parallel scanner.
- Change the metadata-only persistent-file admission boundary.
- Remove actionable paths from user-facing notifications or errors; the policy here is
  specifically for diagnostic records.

## Architecture

### 1. Producer-side remediation

Each diagnostic in the five owners is classified from the current source rather than
from the historical line count. The replacement keeps only the least identifying data
that answers the diagnostic question:

- extension when the file kind is enough (a basename is still user data);
- integer size, count, or path depth when only shape matters;
- `content_fingerprint(path)` when records need stable correlation without plaintext;
- `type(exc).__name__` when an exception message can contain a path;
- `redact_user_paths(path)` only when the path itself is necessary for the user to act.

No new path-identity abstraction is introduced. `content_fingerprint` and
`redact_user_paths` are the existing project seams and cover the two required outcomes.
Call-site repair also keeps terminal output, the Logs pane, and any future descriptive
consumer consistent.

When a path-bearing failure currently uses `logger.opt(exception=True)`, the repair
also removes dynamic traceback/diagnose capture and emits the exception type explicitly.
Changing only the message is insufficient: Loguru can render path-bearing exception
text and frame locals even when the message itself contains only a fingerprint.

TASK-19936's failing-normalization diagnostic keeps disclosure identity through a fixed
event label and failure class. It does not print `raw` or `str(exc)`.

### 2. Extend the existing AST scanner

`scripts/check_persistent_diagnostic_inventory.py` remains the sole production
diagnostic scanner. It gains a path-privacy projection alongside its existing owner and
sink projections.

For every recognized diagnostic call, the scanner examines dynamic message inputs from:

- f-string `FormattedValue` nodes;
- Loguru positional and keyword formatting arguments;
- `%` and `.format(...)` message construction where the value expression is visible to
  the AST; and
- multiline forms, which are ordinary AST nodes and require no special case.

An expression is path-shaped when its referenced name or attribute ends in a bounded
path vocabulary such as `path`, `root`, `directory`, `folder`, or an explicit database
path form. Substring matches inside unrelated identifiers are not enough.

The scanner recognizes these existing de-identifying forms as safe:

- `content_fingerprint(...)`;
- `redact_user_paths(...)`;
- a `Path` suffix projection when only file kind matters;
- cardinality and type-only projections such as `len(...)` and
  `type(exc).__name__`.

The scanner is deliberately conservative and syntactic. It does not claim whole-program
taint analysis or infer that an arbitrary helper sanitizes its return value.

### 3. Reviewed legacy baseline

The generated diagnostic inventory advances to schema version 3 and records the
path-privacy candidate population by stable diagnostic identity. Existing candidates
outside the five owned files carry an explicit `legacy_unreviewed` status. They remain a
visible legacy baseline, not an assertion that they are safe. A changed or newly added
candidate changes the projection and fails the same architecture gate reviewers already
run for diagnostic drift. Existing tests that validate the inventory's exact schema are
updated as part of the same change; the schema version is never widened silently.

The failure report lists the complete added/changed candidate set, grouped by file and
diagnostic identity. It never returns after the first match. Reviewers can therefore
distinguish intentional use of an approved transform from a new raw interpolation before
regenerating the pin.

The five owned files have no raw-path baseline entries after remediation. A regression in
those files is therefore immediately visible rather than grandfathered.

### 4. Logs sharing copy

**Copy all (redacted)** remains metadata-only under TASK-19555 and is unchanged.

**Copy visible logs** remains a deliberate descriptive export. Its warning continues to
say that file names and search terms may remain because legacy diagnostics outside this
task's owner set still exist. Tests pin that wording as accurate; this task must not make
a repository-wide claim from a five-owner repair.

## Failure and reporting behavior

The checker keeps its current non-zero exit contract. Inventory drift and path-privacy
drift are rendered as separate sections in one report, followed by the existing review
instructions. Every candidate is collected before rendering, so two injected unsafe
calls produce two named findings.

Syntax or source-read failures remain hard failures. They are not converted into an
empty candidate set, because a guard that silently skips an unparseable file would be a
privacy bypass.

## Testing

### Scanner tests

Focused architecture tests use synthetic source to prove:

- f-string, Loguru argument, `.format`, percent-format, and multiline raw paths are
  detected;
- safe fingerprint, redaction, suffix, length, and exception-type forms are
  accepted;
- similarly named non-path values do not trigger merely by substring;
- duplicate and multi-file violations preserve multiplicity and all appear in the
  report; and
- injecting a second unsafe call turns the guard red and reports both calls.

### Owner tests

Focused Loguru-capture tests drive distinctive absolute-path sentinels through the five
owners' reachable failure paths. Each test asserts that the expected event still appears
while the full sentinel and path-bearing exception text do not appear anywhere in the
captured record, including exception output. Source-level assertions
cover branches whose production setup would be disproportionate, but every
runtime-reachable repair uses the real logging path.

### Generated inventory

After production and test formatting is final:

1. Review inherited `dev` drift separately. At branch creation, current `dev`
   (`6b2fb1de`) already has one unpinned owner,
   `tldw_chatbook/Agents/virtual_cli_provider.py` (two diagnostics from PR #2168).
2. Review every changed statement in the five owned files.
3. Regenerate `Docs/security/production-diagnostic-inventory.json` once.
4. Run the checker and the focused architecture suite again.

The inherited `virtual_cli_provider.py` row is recorded as base drift, not attributed to
TASK-19864, and is classified before it is folded into the regenerated inventory.

## Governance

ADR required: no

ADR path: `backlog/decisions/029-local-private-data-boundary.md`

Reason: ADR-029 already establishes the metadata-only persistent boundary and the
producer-side privacy rule. This task applies that existing policy to live diagnostics
and strengthens its architecture enforcement without changing data ownership, sink
admission, storage, or service contracts.

## Rejected alternatives

### Rewrite every lexical candidate now

A current conservative grep finds 281 candidates across 85 files. That is useful as a
cross-check, not as proof that all 281 are raw user paths. Combining classification,
remediation, and behavior testing for that population would violate the task's atomic
five-owner boundary and make review less reliable.

### Add a second path-diagnostic scanner

This would duplicate logger-symbol discovery, chained Loguru recognition, source
walking, inventory serialization, and reporting. The two scanners would drift on what
counts as a production diagnostic.

### Redact at the terminal or Logs sink

A blanket sink filter either removes useful actionable context or applies differently
across terminal, in-app, clipboard, and future consumers. It also leaves the diagnostic
record itself carrying the raw path. Producer-side repair is the existing ADR-029 idiom
and gives every consumer the same safe message.
