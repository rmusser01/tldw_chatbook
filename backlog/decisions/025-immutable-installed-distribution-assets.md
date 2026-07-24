# ADR 025: Immutable Installed Distribution Assets

Status: Accepted
Date: 2026-07-24
Related Tasks:
[TASK-545](../tasks/task-545%20-%20Verify-installed-distributions-and-immutable-packaged-assets.md)
Supersedes: N/A

## Decision

Treat an installed `tldw_chatbook` distribution as immutable application code.
Runtime-owned configuration, templates, compiled styles, and vendored license
notices must be present in the built wheel. Installed commands may read those
files but must not rebuild or write them under the installed package root.

Keep the current source-checkout CSS convenience behavior, but enter the
source-module freshness and rebuild path only when the package has an adjacent
repository `pyproject.toml`. Packaged applications and ordinary wheel installs
use the committed `tldw_cli_modular.tcss` bundle as built.

Make distribution verification behavioral as well as structural. Build the
sdist and wheel from a temporary source copy, inspect their required contents,
install the wheel into a separate target, and execute the installed entry-point
contracts with all user state redirected to a temporary private root. The
installed process must prove that `tldw_chatbook` resolved from the installation
target rather than an editable checkout.

Keep package data explicit. Include only files with a demonstrated runtime or
license obligation rather than recursively shipping every non-Python file in
the repository.

## Context

The existing tests import from the source checkout. They therefore cannot
detect missing wheel data or an installed process that accidentally falls back
to editable source.

A clean distribution probe found that the wheel omitted:

- `Config_Files/rag_pipelines.toml`, used by both RAG pipeline loaders;
- all thirteen built-in `Chunking/templates/*.json` definitions;
- `Evals/config/eval_config.yaml`, whose loaded task vocabulary is broader than
  the hard-coded fallback; and
- the Apache and MIT license files belonging to the vendored aider and
  textual-fspicker code shipped in the wheel.

The same probe installed the wheel into a separate target and ran
`tldw-cli --help`. The command exited successfully, but its startup path treated
the wheel as a mutable source tree, attempted to rebuild the generated CSS
bundle, and logged a failure because `components/stats_screen.css` is not wheel
data. Merely packaging that source fragment would still leave the application
writing into `site-packages`, which is not a safe runtime contract.

The canonical release checker also failed against freshly built artifacts.
`Packaging/MANIFEST.in` is not at the project root where setuptools reads it,
so the sdist omitted six files the checker itself requires: `CLAUDE.md`,
`CHANGELOG.md`, `MANIFEST.in`, `requirements.txt`, and related manifest
expectations.

These failures cross the tooling/runtime boundary and reject a plausible
alternative—repairing individual missing files while continuing to rebuild
inside installed packages—so the decision is recorded before implementation.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Continue source-checkout tests only | They cannot observe missing distribution data, broken console-script metadata, or source-shadowed imports. |
| Inspect the wheel archive without installing it | Archive inspection proves presence, not that entry points and runtime loaders resolve the installed copy. |
| Recursively include every repository file under `tldw_chatbook` | It would conceal ownership mistakes, increase artifacts, and risk shipping backups, development documents, or unrelated data. |
| Package every CSS source and keep rebuilding in `site-packages` | Installed package roots can be read-only and are not user-state or build-output locations. |
| Remove source-checkout CSS freshness checks entirely | The existing development convenience can remain without weakening the installed boundary. |
| Create a fresh environment and resolve every dependency from package indexes in this test | That adds network availability and dependency-index state to a deterministic artifact test. Dependency-resolution matrices can be a separate release gate if needed. |
| Add a new packaging framework or custom installer | Setuptools, `build`, `pip --target`, and the standard library already provide the required proof. |

## Consequences

The root manifest and setuptools package-data declarations become the explicit
distribution contract. The artifact checker must validate the same required
runtime files and vendored notices in both applicable artifacts.
The source distribution also retains every declared CSS build input, including
`stats_screen.css`; ordinary wheels need only the committed runtime bundle and
do not rebuild from those sources.

The generated CSS bundle must be committed and current before distribution
builds. A wheel does not self-heal an absent or stale bundle at runtime. Source
checkouts retain their existing freshness check because the repository
`pyproject.toml` identifies that environment.

The installed-wheel regression uses the current test interpreter's dependencies
while installing the project wheel with `--no-deps`. This deliberately proves
artifact integrity without proving package-index dependency resolution.

Installed help commands may still initialize existing configuration and logging
paths before argument parsing. The test contains those effects under a
temporary root; changing CLI initialization order belongs to the later
application-state decomposition.

No new application-state owner, packaging service, runtime configuration
object, or dependency is introduced beyond making the already-declared `build`
test tool available to the test environment.

## Links

- [Installed distribution integrity design](../../Docs/superpowers/specs/2026-07-24-installed-distribution-integrity-design.md)
- [Packaging checklist](../../Packaging/PACKAGING_CHECKLIST.md)
