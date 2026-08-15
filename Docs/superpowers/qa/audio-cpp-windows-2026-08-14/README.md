# Windows audio.cpp provisioned UAT

Status: **NOT RUN — TASK-13208 remains In Progress**

This handoff validates Windows 10+ on x86 or x64 with Python 3.12+. ARM is not
claimed. The checked-in harness is generic and contains no workstation path,
username, server build, model version, or device assumption.

## What the harness consumes

The operator provides:

- an existing compatible audio.cpp server executable;
- an existing local text package directory;
- an existing clone-capable package directory;
- a clone reference WAV and bounded transcript;
- exact expected recipe/model identities; and
- the exact curated managed-artifact identity for the clone package.

The executable is reviewed and then launched **in place** by Chatbook's existing
supervisor. It is not copied, imported, installed, downloaded, or bundled. The
clone model package is copied into a disposable managed-model store through the
normal `ModelArtifactService.install()` path so that the UAT exercises a real
Model Library root and lease. Its source directory is unchanged. The local text
package remains local and is scanned in place.

## One-command run

From the repository root in Windows PowerShell 5.1+ or PowerShell 7:

```powershell
./scripts/uat_audio_cpp_windows.ps1 `
  -ServerBinary <server-executable> `
  -TextPackageRoot <local-text-package-directory> `
  -ClonePackageRoot <clone-package-directory> `
  -CloneReferenceWav <reference-wav> `
  -CloneReferenceText <reference-transcript> `
  -TextRecipeId <text-recipe-id> `
  -TextRecipeRevision <text-recipe-revision> `
  -TextPackageVariant <text-package-variant> `
  -TextModelId <text-model-id> `
  -CloneRecipeId <clone-recipe-id> `
  -CloneRecipeRevision <clone-recipe-revision> `
  -ClonePackageVariant <clone-package-variant> `
  -CloneModelId <clone-model-id> `
  -CloneArtifactId <managed-artifact-id> `
  -CloneArtifactRevision <managed-artifact-revision> `
  -CloneArtifactVariant <managed-artifact-variant> `
  -EvidenceOutput <sanitized-evidence-json>
```

Do not paste real parameter values into this file, an issue, a pull-request
comment, or a chat transcript. The evidence JSON intentionally contains only
the supported architecture, Python major/minor, fixed check names, structural
WAV facts, the human audible result, cleanup state, and final status.

## Objective journey

The Python layer validates the host tuple and exact package identities, reviews
the executable without launching it, installs the clone package into the
disposable Model Library store, and constructs Guided settings without starting
a process. Deliberate use then exercises generated configuration, health and
catalog discovery, text WAV generation, operation-scoped clone-reference
materialization and clone WAV generation, cancellation/restart recovery, forced
owned-child crash recovery, live removal blocking, final shutdown, exact
post-stop removal, and structural WAV validation.

The PowerShell layer creates disposable home/config/data/runtime roots, restores
the caller's process environment in `finally`, plays both generated WAVs with
the Windows sound API, requires an explicit `yes`/`no` audible-and-intelligible
answer, and removes the disposable root. Any unsupported, partial, failed,
inaudible, or dirty-cleanup result exits nonzero.

## Evidence required before closure

Run the command independently on each architecture being claimed. Record only:

- checked-out commit;
- Windows major/minor and x86 or x64;
- Python major/minor;
- final fixed-schema evidence JSON; and
- whether the hosted `windows-audio-cpp` CI job passed for that tuple.

If either x86 or x64 lacks a compatible provisioned server/package combination,
or any objective or audible step is unavailable, record that tuple as PARTIAL
and leave TASK-13208 In Progress. Evidence from one architecture must not be
projected onto the other.
