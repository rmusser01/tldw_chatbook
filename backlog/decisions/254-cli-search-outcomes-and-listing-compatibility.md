# ADR-254: CLI search outcomes and listing compatibility

Status: Accepted
Date: 2026-10-29
Related tasks: TASK-37630, TASK-37631, TASK-37632
Amends: ADR-078, ADR-094, ADR-101

## Decision

For raw model shell calls, classify exit 1 with empty stderr as successful empty
search only when the exact executed command is an unambiguous simple invocation
of grep, rg or findstr. Preserve the exit code and output in the result. Other
nonzero exits, uncertain shell expressions and actual executor failures retain
their existing outcomes and retry limits. No user-authored command authority,
approval, timeout, cleanup or permission behavior changes.

Remove the raw provider's redundant 4,000-character concatenated-result slice.
The existing executor stream and configured storage bounds remain authoritative.

Virtual ls accepts explicit all/long compatibility flags. Its existing hidden
entry behavior remains; long mode adds permissions/type and byte size. An
optional boolean long_format field extends the closed fs_list worker operation;
absent/false preserves existing output. Parent and stdlib worker validation use
the same schema, including the shipped remote bundle. Command usage is visible
to the model. No host shell or wider folder authority is introduced.

Windows atomic writes reuse the existing native file facade to implement the
same pinned-parent/CAS contract as POSIX. Secure rename, link and exclusive
creation remain mandatory; no path-based fallback weakens the mutation boundary.
The native locked-target replacement explicitly uses FileRenameInformationEx
with POSIX rename semantics so Windows's mandatory target lock stays held
through the atomic swap. Existing general rename behavior is unchanged.
Windows creation and replacement retain the facade's private DACL policy;
POSIX mode preservation remains unchanged.

## Alternatives

Treating every exit 1 as success hides ordinary program failures. Changing the
global failure threshold permits more retries of unavailable or uncertain
tools. Arbitrary shell-command rewriting can alter execution and approval text.
Silently ignoring all virtual flags misrepresents semantics. Independent native
write code would duplicate the established Windows handle layer.

## Consequences

Empty simple searches no longer consume the ordinary same-tool failure streak.
Ambiguous expressions remain conservative; this is not a general shell parser.
Complete bounded shell diagnostics can use more model context. Existing worker
limits, exclusions, root pins and permissions remain enforced. Long listing
metadata is obtained only for already admitted directory entries.
