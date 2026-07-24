# Installed Distribution Integrity Design

Date: 2026-07-24
Status: Re-reviewed; awaiting written-spec approval
ADR:
[ADR-025](../../../backlog/decisions/025-immutable-installed-distribution-assets.md)
Backlog:
[TASK-545](../../../backlog/tasks/task-545%20-%20Verify-installed-distributions-and-immutable-packaged-assets.md)

## Summary

Add a deterministic distribution gate that builds both standard Python
artifacts in a fresh tree outside the working tree, validates their owned data,
metadata, exclusions, and licenses, installs the wheel outside the repository,
and exercises the installed entry-point and resource contracts. Installed
package files are immutable; only a source tree identified by the adjacent
project metadata may rebuild the committed CSS bundle.

This task repairs the packaging defects exposed by that gate. It does not begin
the larger application-state decomposition.

## Verified Baseline

The review used a tracked-source archive in a temporary directory and the
repository virtual environment.

### Artifact construction

`python -m build --sdist --wheel --no-isolation` completed, proving the
setuptools build itself is viable without a network call.

`Packaging/check_manifest.py` then exited `1`. The newly built sdist lacked six
items expected by the checker because the repository's manifest lives at
`Packaging/MANIFEST.in`, not the project root where setuptools consumes it.
Among the reported omissions were:

- `CLAUDE.md`;
- `CHANGELOG.md`;
- `MANIFEST.in`; and
- `requirements.txt`.

The checklist currently claims this path is working, so its evidence is stale.

### Wheel data

The wheel contained Python modules, the compiled TCSS bundle, TCSS sources, and
the two configured JSON note-template files. It did not contain:

- `tldw_chatbook/Config_Files/rag_pipelines.toml`;
- any of the thirteen `tldw_chatbook/Chunking/templates/*.json` files;
- `tldw_chatbook/Evals/config/eval_config.yaml`;
- `tldw_chatbook/Third_Party/aider/LICENSE.txt`; or
- `tldw_chatbook/Third_Party/textual_fspicker/LICENSE`.

Each first-party data omission has a verified package-relative runtime reader.
The two license omissions accompany vendored code that is present in the
wheel.

`embedding_configs_examples.toml`, example pipeline files, package-local
development Markdown, SQL source files not read at runtime, and backup files
have no demonstrated wheel runtime contract and remain excluded.

### Implicit package-data leakage

Setuptools defaults `include-package-data` to true for projects configured
through `pyproject.toml`. A clean reproduction with a root `MANIFEST.in`
confirmed that `tldw_chatbook/css/components/stats_screen.css`, intended as an
sdist build input, then leaked into the wheel despite not appearing in the
explicit package-data table.

Setting `include-package-data = false` and rebuilding from a fresh tree kept the
thirteen chunking templates, RAG config, eval config, compiled bundle, and two
vendored notices while excluding `stats_screen.css` from the wheel. Reusing the
same setuptools `build/` directory initially produced a false result because a
stale copied CSS file survived there. The test must therefore own a fresh source
and build tree rather than cleaning or reusing repository build output.

The reproduced sdist also contained root test modules when their directory was
spelled or normalized as lowercase `tests/`. The manifest currently excludes
`Tests/` and `STests/` only. Both spellings must be excluded, and the checker
must verify forbidden paths rather than checking presence alone.

### Build metadata

The build emits a setuptools deprecation warning for
`license = {text = "AGPL-3.0-or-later"}`. The warning states that this form
stops being supported after 2027-02-18. A clean probe with setuptools 81
confirmed that raising the build-backend floor to 77, using the SPDX string
form, and declaring `license-files = ["LICENSE"]` removes the warning and
produces Core Metadata 2.4 with:

```text
License-Expression: AGPL-3.0-or-later
License-File: LICENSE
```

### Installed execution

Installing the wheel with `pip --target` and running from an unrelated
directory succeeded when the temporary installation was placed first on
`PYTHONPATH`. The package origin resolved inside that installation target, not
the checkout.

Both installed help entry points returned success. `tldw-serve --help` kept its
current optional-dependency boundary and needs no change.

`tldw-cli --help` initialized existing application startup code and then
attempted to rebuild `css/tldw_cli_modular.tcss` inside the installation. The
build logged:

```text
Failed to build modular CSS:
FileNotFoundError: Missing declared CSS module(s): components/stats_screen.css
```

Adding that `.css` source to the wheel would hide the immediate failure but
would preserve the unsafe write into the installed package. The repair must
stop installed rebuilds.

A source scan found the same package-writing behavior in three application
startup sites: direct `app.py` execution, `get_app()`, and
`main_cli_runner()`. Guarding only the console-script path would make the test
green while leaving the Textual Serve application factory and direct module
path unsafe. This is not merely a legacy path:
`Web_Server.serve.run_web_server()` launches `python -m tldw_chatbook.app`.
All three sites must share the same source-tree predicate.

The probe also confirmed the existing missing `openai_tts_mappings.json`
fallback remains functional. Its warning is noisy but unrelated to
distribution integrity and is not part of this task.

## Goals

- Build sdist and wheel artifacts in a temporary source copy.
- Restore a root setuptools manifest and make the release checker truthful.
- Make wheel package data explicitly opt-in instead of implicitly inheriting
  every in-package sdist manifest match.
- Package every verified runtime-owned non-Python file needed by the tested
  installed flows.
- Package license notices for the vendored code shipped in the wheel.
- Replace the time-limited legacy license declaration with current SPDX
  metadata and verify the resulting distribution metadata.
- Prove the imported package comes from the installed target.
- Prove the packaged RAG, chunking, eval, CSS, and entry-point contracts work.
- Keep installed commands from rebuilding or writing generated package assets.
- Contain help-command configuration, logs, data, and temporary files under the
  test's temporary root.
- Run without network dependency resolution.

## Non-Goals

- Decomposing `app.py` or introducing a new application-state owner.
- Reordering CLI argument parsing and startup side effects.
- Testing every optional dependency or provider.
- Resolving dependencies from PyPI in a clean environment.
- Adding a packaging framework, installer abstraction, or release service.
- Shipping example-only TOML, development documents, backups, or every
  non-Python repository file.
- Reworking PyInstaller, DMG, or Windows installer flows.
- Removing the source-checkout CSS freshness convenience.
- Repairing the unrelated OpenAI TTS mapping fallback warning.
- Releasing or uploading artifacts.

## Distribution Content Contract

### Source distribution

`MANIFEST.in` moves to the repository root. It continues to include the
release metadata already required by `Packaging/check_manifest.py` and adds
the runtime asset and vendored-license patterns below.

The sdist must contain:

- `LICENSE`;
- `README.md`;
- `CLAUDE.md`;
- `CHANGELOG.md`;
- `MANIFEST.in`;
- `pyproject.toml`;
- `requirements.txt`;
- the Python package;
- the compiled CSS bundle, `.tcss` modules, and `stats_screen.css` source input;
- the runtime configuration and template files; and
- the two vendored license files.

It must not contain `Tests/`, `tests/`, `STests/`, Python bytecode/cache files,
or OS metadata. Both test-directory spellings are excluded because
case-normalized source copies and case-sensitive CI filesystems must produce
the same contract.

### Build and license metadata

The build backend remains setuptools, with its minimum raised to the first
version that supports the selected PEP 639 metadata:

```toml
[build-system]
requires = ["setuptools>=77.0"]

[project]
license = "AGPL-3.0-or-later"
license-files = ["LICENSE"]
```

The artifact checker parses distribution metadata and requires
`License-Expression: AGPL-3.0-or-later` plus `License-File: LICENSE`. The wheel
must contain the project license under its `.dist-info/licenses/` directory in
addition to the two vendored notices under their package owners.

### Wheel

Setuptools package-data is made genuinely explicit:

```toml
[tool.setuptools]
include-package-data = false
```

Without that setting, root-manifest entries inside `tldw_chatbook/` are
implicitly eligible for the wheel and the package-data table is not an
allowlist.

| Package owner | Required data |
| --- | --- |
| `tldw_chatbook.css` | `*.tcss`, nested `*.tcss`, and the committed `tldw_cli_modular.tcss` bundle |
| `tldw_chatbook.Config_Files` | existing `*.json` and `*.md`, plus `rag_pipelines.toml` |
| `tldw_chatbook.Chunking` | `templates/*.json` |
| `tldw_chatbook.Evals` | `config/*.yaml` |
| `tldw_chatbook.Third_Party.aider` | `LICENSE.txt` |
| `tldw_chatbook.Third_Party.textual_fspicker` | `LICENSE` |

The data rules do not use a catch-all recursive glob.

The wheel must not contain `stats_screen.css`, example-only TOML, development
Markdown outside the deliberately shipped Config Files guides, root test
trees, Python bytecode/cache files, or OS metadata. The CSS source remains in
the sdist for manual source builds; installed wheels consume only the committed
runtime bundle.

The built-in chunking contract is the complete current source set:

```text
academic_paper
code_documentation
conversation
ebook_chapters
json
legal_document
paragraphs
rolling_summarize
semantic
sentences
tokens
words
xml
```

The installed eval loader must observe a YAML-only value such as
`code_execution`; that distinguishes the packaged file from its smaller
hard-coded fallback. The installed RAG configuration must parse and expose the
`plain` pipeline.

## Installed Asset Immutability

Application startup currently has three CSS bootstrap sites. Add one small
internal source-tree predicate and require direct `app.py` execution,
`get_app()`, and `main_cli_runner()` to use it before creating the CSS
directory, scanning sources, or invoking the builder:

```python
def _is_source_tree(package_root: Path) -> bool:
    return (package_root.parent / "pyproject.toml").is_file()
```

The predicate centralizes environment detection without refactoring
application state or changing each source path's existing rebuild policy. Only
a positive result enters a source-module scan, creates the package CSS
directory, or invokes `css/build_css.py`. An ordinary wheel target and packaged
application do not have an adjacent repository `pyproject.toml`, so they
consume the committed bundle without attempting self-modification.

The installed regression records a relative-path-to-SHA-256 map for every
regular file in the complete installation target before and after entry-point
help execution. It also rejects CSS rebuilding and failure messages in captured
process output and private-root log files. Content hashing avoids timestamp
false positives while inventory comparison detects new files. This makes the
boundary observable rather than relying only on source inspection.

If the committed bundle is absent from a future wheel, artifact verification
fails. Runtime regeneration is not a fallback for a bad distribution.

## Test Architecture

### Temporary source build

A module-scoped integration fixture copies the package, packaging tools,
release metadata, and root test-tree candidates required to reproduce
setuptools selection into `tmp_path_factory` storage:

- `tldw_chatbook/`;
- `Packaging/`;
- `Tests/`, `tests/`, and `STests/` when present;
- `pyproject.toml`;
- root `MANIFEST.in`;
- `README.md`;
- `LICENSE`;
- `CLAUDE.md`;
- `CHANGELOG.md`; and
- `requirements.txt`.

Python caches, `.git`, virtual environments, and existing `build/`, `dist/`,
and egg-info artifacts are excluded from the copy. The destination starts
empty, so stale setuptools output cannot make an exclusion test pass or fail.

The fixture runs:

```text
python -m build --sdist --wheel --no-isolation
```

`build` and `setuptools>=77` are added to `requirements-test.txt`; `build` is
already declared by the project's `dev` extra. `--no-isolation` prevents the
build frontend from downloading build requirements during the test while the
declared test dependency guarantees the required backend is present.

### Artifact inspection

The test runs the copied `Packaging/check_manifest.py` against the temporary
`dist/` directory and requires exit `0`. The checker requires exactly one new
sdist and one new wheel, validates required paths explicitly, checks console
script metadata and license metadata, and rejects forbidden test, cache,
example, and sdist-only wheel paths.

Archive inspection remains valuable even though the wheel is installed: it
produces a direct missing-path error before a loader silently selects a
fallback.

### Isolated wheel target

The fixture installs the wheel with:

```text
python -m pip install --no-deps --target <temporary-target> <wheel>
```

The child process runs from an unrelated temporary directory with `PYTHONPATH`
set to the target only, rather than preserving a caller value. It prints and
verifies
`Path(tldw_chatbook.__file__).resolve()` under that target. A source checkout
or another editable worktree therefore cannot make the test pass.

`--no-deps` is intentional: the parent test environment already owns required
dependencies, while this gate owns the project artifact. No network is needed.

### Private child environment

The child environment uses canonical temporary paths for:

- `HOME`;
- `USERPROFILE`, `APPDATA`, and `LOCALAPPDATA`;
- `XDG_CONFIG_HOME`;
- `XDG_DATA_HOME`;
- `TLDW_CONFIG_PATH`;
- `TMPDIR`, `TEMP`, and `TMP`; and
- `PYTHONDONTWRITEBYTECODE=1`.

`PYTHONPATH` is set separately to the installed target.

Inherited test-root ownership variables are removed. The canonical spelling is
required on macOS because `/tmp` is a symlink and the private-path boundary
rejects symlinked parent traversal.

The test accepts the existing help-command creation of configuration, logs, and
data only inside this root. Disabling bytecode output ensures ordinary imports
do not create `__pycache__` under the target. The test verifies that the
complete install-target inventory and content hashes remain unchanged.

### Behavioral assertions

The installed subprocess must prove:

1. package origin is under the temporary target;
2. `tldw_cli_modular.tcss` exists;
3. `rag_pipelines.toml` parses and contains `pipelines.plain`;
4. `ChunkingTemplateManager.get_available_templates()` returns the complete
   built-in set;
5. `EvalConfigLoader().get_task_types()` includes `code_execution`;
6. both vendored license files exist;
7. wheel metadata declares the exact `tldw-cli` and `tldw-serve` console
   targets;
8. platform-aware discovery under the target's `bin/` or `Scripts/` directory
   runs both installed `--help` commands successfully;
9. project license metadata uses the SPDX expression and names `LICENSE`;
10. the installed `get_app()` factory can construct the application without a
    CSS rebuild;
11. no installed startup path reports a CSS rebuild attempt or failure; and
12. every file and hash in the installation target is unchanged by installed
    entry-point and factory execution.

The tests use the `integration` marker so the existing integration and full
suite workflows pick them up without multiplying the build across the unit
OS/Python matrix.

## Error Handling and Diagnostics

Every subprocess captures stdout and stderr. Failures report the command,
return code, and captured output. Artifact checks name missing archive paths.
The origin assertion reports both expected target and actual module location.

The tests do not suppress the repository's existing dependency and optional
feature warnings. They reject only packaging-specific failure signals and
incorrect return codes.

Temporary directories are pytest-owned and removed normally. No build command
cleans or overwrites the working tree's `dist/`, `build/`, or egg-info paths.

## Documentation

Update `Packaging/PACKAGING_CHECKLIST.md` so it no longer claims the misplaced
manifest is healthy and documents the installed-wheel integration command.
Update `Packaging/check_manifest.py` as executable release evidence rather than
adding a second checker.

The Backlog task, ADR, design, implementation plan, and completion notes link
to one another.

## Verification Gates

Completion requires:

- red proof for the current missing resources, broken manifest, installed CSS
  rebuild, implicit wheel-data leakage, sdist test leakage, legacy license
  metadata, and missing vendored notices;
- focused installed-distribution integration tests;
- focused config, chunking, eval, every CSS bootstrap site, and packaging
  safety regressions;
- the existing eval/tool cross-task gates affected by this branch;
- Ruff on changed Python;
- Python compilation of changed Python;
- `git diff --check`; and
- a final installed artifact build from the committed source state.

## Re-review Record

The approved initial proposal covered only `Config_Files/*.toml`. A complete
runtime-data and installed-entry-point review corrected five weaknesses:

- it added missing chunking and eval resources rather than fixing one TOML
  symptom;
- it found and included the omitted vendored license obligations;
- it proved the existing root-manifest/checker contract is already broken;
- it replaced the tempting `stats_screen.css` packaging patch with the correct
  immutable installed-package boundary; and
- it replaced a static-only check with an installed origin assertion and
  behavioral loader checks.

The result still uses setuptools, `build`, pip, pytest, and the standard
library. It adds no packaging framework, state owner, or application
decomposition.

A second adversarial review then verified five additional hazards:

- root manifest matches implicitly leaked sdist-only CSS into wheels until
  `include-package-data` was explicitly disabled;
- reused setuptools build output preserved stale data, so the gate now requires
  a fresh source and build tree;
- case-normalized lowercase `tests/` entered the sdist and now has an explicit
  negative contract; and
- the legacy license-table syntax carries a dated 2027 failure boundary, so the
  same packaging change now emits and checks current SPDX metadata; and
- CSS bootstrap writes exist in three startup sites, so all three now share the
  same source-tree predicate instead of guarding only the tested console path.

The source-only CSS freshness behavior remains unchanged because it is an
already-approved development convenience and the installed-target hash gate
directly enforces the relevant immutability boundary.
