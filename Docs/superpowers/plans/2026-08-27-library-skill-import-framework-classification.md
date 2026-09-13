# Library Skill Import Framework Classification Implementation Plan

> Execution: use `superpowers:test-driven-development` for each backlog task
> and `superpowers:verification-before-completion` at each commit boundary.

**Goal:** Make every Library skill import single-flight and truthfully classify
installable skills, multi-skill repositories, non-skill frameworks, malformed
inputs, and network failures.

**Architecture:** Library keeps one authoritative import operation state and
refuses a second submit while a file, folder, zip, or URL import is already
accepted. A pure package inspector owns the shared classification vocabulary
for local directories and downloaded zip central directories. Fetching,
security policy, bounded extraction, import, and trust remain with the existing
services; classification never grants trust or executes repository code.

**Tech stack:** Python 3.11+, dataclasses/StrEnum, pathlib/zipfile, existing
SSRF-hardened remote fetch, Textual 8.x, pytest/pytest-asyncio.

**Backlog tasks:** TASK-613 → TASK-22867.

**ADR required:** no

**ADR path:** N/A

**Reason:** This closes an existing worker race and adds truthful outcome/UI
classification while preserving the current skill storage, trust, remote-fetch,
and runtime boundaries. A framework remains external content; Chatbook does not
adopt it as a product integration or execute its installer.

## TASK-613 — Single in-flight Library skill import

### Files

- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/Skills/test_skills_import.py`
- Modify: `Tests/Skills/test_skills_library_flow.py`
- Modify: `Docs/User_Guide/library/skills.md`

### Step 1: Reproduce the superseded threaded import

Add a RED mounted test with an importer blocked by two events. Start a folder
import, attempt a second submit through Enter and the Import button, and then
release the first call. Prove the current exclusive worker group can cancel the
first UI await while its `asyncio.to_thread`/worker-side mutation still lands.
Repeat the guard assertions for loose Markdown, folder, zip, and URL routes.

The expected contract is one accepted import at a time. A second invocation
does not queue, cancel, or replace the first and shows the fixed status
“An import is already in progress.”

Run:

```bash
pytest -q Tests/Skills/test_skills_import.py -k "in_flight or second_submit or superseded"
```

Expected RED: the second exclusive worker replaces the first UI await.

### Step 2: Make the import row genuinely single-flight

Add `_library_skills_import_in_flight: bool` beside the existing import draft
state. Set it synchronously in `_start_library_skills_import` before scheduling
work, run the accepted worker without exclusive replacement semantics, and
clear it in one `finally` after local/remote success or failure. Every early
return after acceptance must pass through that `finally`.

While true, disable the path input, Browse, Browse folder, Import, and Cancel;
render “Inspecting/importing…” as visible status. Handler guards must repeat the
state check because disabled widgets are not an authorization boundary. Keep
Library navigation available. Leaving the Skills canvas does not claim to
cancel the operation, and returning renders the in-flight state or the actual
completed/failed outcome from the same screen-owned operation state.

Do not add a Cancel button: the accepted threaded filesystem/network work has
no truthful cancellation guarantee. Do not infer completion from a worker
being cancelled; refresh the authoritative skills snapshot after the service
call returns.

Run:

```bash
pytest -q Tests/Skills/test_skills_import.py Tests/Skills/test_skills_library_flow.py -k "import and (in_flight or disabled or navigate or complete or failure)"
```

### Step 3: Verify and commit TASK-613

```bash
pytest -q Tests/Skills/test_skills_import.py Tests/Skills/test_skills_library_flow.py Tests/Skills/test_import_skill_directory.py Tests/Skills/test_skill_remote_fetch.py
ruff check tldw_chatbook/UI/Screens/library_screen.py
git diff --check
```

Commit boundary:

```bash
git add tldw_chatbook/UI/Screens/library_screen.py Tests/Skills/test_skills_import.py Tests/Skills/test_skills_library_flow.py Docs/User_Guide/library/skills.md backlog/tasks/task-613\ -\ Library-skills-import-in-flight-cancel-race.md
git commit -m "fix: serialize Library skill imports"
```

## TASK-22867 — Classify frameworks and multi-skill repositories

### Files

- Create: `tldw_chatbook/Skills_Interop/skill_package_inspection.py`
- Create: `Tests/Skills/test_skill_package_inspection.py`
- Create: `tldw_chatbook/UI/Library_Modules/skill_import_choice_modal.py`
- Create: `Tests/Skills/test_skill_import_choice_modal.py`
- Modify: `tldw_chatbook/Skills_Interop/skill_remote_fetch.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/Skills/test_skill_remote_fetch.py`
- Modify: `Tests/Skills/test_skills_import.py`
- Modify: `Tests/Skills/test_skills_library_flow.py`
- Modify: `Docs/User_Guide/library/skills.md`

### Step 1: Pin one package-classification vocabulary

Create local directory and in-memory zip fixtures, then write RED tests for:

```python
class SkillPackageKind(StrEnum):
    ROOT_SKILL = "root_skill"
    MULTI_SKILL_REPOSITORY = "multi_skill_repository"
    FRAMEWORK_REPOSITORY = "framework_repository"
    MALFORMED_OR_UNSUPPORTED = "malformed_or_unsupported"
    FETCH_OR_AUTH_FAILURE = "fetch_or_auth_failure"

@dataclass(frozen=True)
class SkillPackageInspection:
    kind: SkillPackageKind
    candidates: tuple[str, ...]
    message: str
    recovery_actions: tuple[str, ...]

def inspect_skill_directory(path: Path) -> SkillPackageInspection:
    """Classify one bounded local import candidate without importing it."""

def inspect_skill_zip(
    data: bytes, *, repository_source: bool
) -> SkillPackageInspection:
    """Classify one bounded archive from central-directory metadata."""
```

Both entry points share one candidate scanner and return the same outcome
model. A top-level `SKILL.md` is one root skill. Two or more accepted
subdirectories are a multi-skill repository in stable path order. A valid,
nonempty GitHub repository archive with no accepted `SKILL.md` is a framework
repository. A corrupt/empty/direct archive with no skill, unsafe candidate,
unsupported input type, or invalid local path is malformed/unsupported.
Transport, HTTP authentication, rate limit, and download failures are mapped at
the remote-fetch boundary to fetch/auth failure.

Reuse the current depth, count, size, path, symlink, and extraction caps. The
inspector reads zip central-directory metadata only and never executes or
imports repository content.

Run:

```bash
pytest -q Tests/Skills/test_skill_package_inspection.py
```

Expected RED: `re_root_skill_zip` currently reduces no-candidate and
multi-candidate repositories to message-only `RemoteSkillError` values.

### Step 2: Preserve bounded fetch/import separation

Refactor `re_root_skill_zip` and `install_skill_from_url` to consume the new
inspection result. Keep URL parsing, GitHub ref resolution, public-address
validation, redirect revalidation, authorization stripping, total deadline,
download cap, bounded extraction, and trust-pending import unchanged.

Add an optional, already-inspected `subdir` only after the user chooses an
exact candidate. Retain the bounded downloaded archive and its SHA-256 in the
single in-memory import operation until choice/cancel, and re-root/import those
same bytes; do not refetch a mutable branch after the user reviewed its
candidate list. Clear the bytes on every terminal path. The final import still
flows through
`SkillsScopeService.import_skill_file(final_bytes, mode="local",
filename=f"{final_name}.zip", content_type="application/zip",
trust_approved=False)`. A framework
classification returns without importing any bytes. Redact tokens, signed
queries, response bodies, local paths, and raw exception strings from all
presented outcomes.

Run:

```bash
pytest -q Tests/Skills/test_skill_remote_fetch.py -k "classif or candidate or framework or auth or redact or reroot"
```

### Step 3: Add generic multiple-skill selection and recovery states

Render the inline row's explicit states: idle, inspecting/importing,
multi-skill choice, not-a-skill/framework, trust review, complete, and failed
with Retry. For a multi-skill result, push `SkillImportChoiceModal` listing the
bounded candidate paths with one selected candidate and Import/Cancel. Import
exactly the chosen subdirectory; never batch-install the whole repository and
never choose silently.

For a valid framework repository with no skill, show:

“This repository is a framework, not an installable Codex skill.”

Offer only generic guidance supported by the product:

- choose a repository subdirectory that contains `SKILL.md`;
- use its project instructions when that is the intended integration;
- use the framework's external CLI outside Chatbook;
- create a separately reviewed wrapper skill.

Do not name ATHF, threat hunting, a vendor, or any repository-specific
installation command. Keep “Import skill” distinct from Library media/document
ingestion. Every successful candidate still lands in the existing trust-review
state and offers the existing Review action.

Run:

```bash
pytest -q Tests/Skills/test_skill_import_choice_modal.py Tests/Skills/test_skills_import.py Tests/Skills/test_skills_library_flow.py -k "framework or multiple or candidate or trust or retry"
```

### Step 4: Combine classification with TASK-613's single-flight contract

Add one delayed remote-inspection test and one delayed selected-candidate
import test. During both phases, all import entry controls stay disabled and a
second file/folder/zip/URL submit is refused. Navigating away and back shows
the authoritative current phase/outcome. Dismissing the candidate modal returns
to the preserved import draft; it does not cancel an already-started import
because import begins only after the explicit candidate selection.

Run:

```bash
pytest -q Tests/Skills/test_skills_import.py Tests/Skills/test_skill_import_choice_modal.py -k "in_flight or navigation or candidate"
```

### Step 5: Verify and commit TASK-22867

```bash
pytest -q Tests/Skills/test_skill_package_inspection.py Tests/Skills/test_skill_remote_fetch.py Tests/Skills/test_skill_import_choice_modal.py Tests/Skills/test_skills_import.py Tests/Skills/test_skills_library_flow.py Tests/Skills/test_import_skill_directory.py
ruff check tldw_chatbook/Skills_Interop/skill_package_inspection.py tldw_chatbook/Skills_Interop/skill_remote_fetch.py tldw_chatbook/UI/Library_Modules/skill_import_choice_modal.py tldw_chatbook/UI/Screens/library_screen.py
git diff --check
```

Commit boundary:

```bash
git add tldw_chatbook/Skills_Interop/skill_package_inspection.py tldw_chatbook/Skills_Interop/skill_remote_fetch.py tldw_chatbook/UI/Library_Modules/skill_import_choice_modal.py tldw_chatbook/UI/Screens/library_screen.py Tests/Skills/test_skill_package_inspection.py Tests/Skills/test_skill_remote_fetch.py Tests/Skills/test_skill_import_choice_modal.py Tests/Skills/test_skills_import.py Tests/Skills/test_skills_library_flow.py Docs/User_Guide/library/skills.md backlog/tasks/task-22867\ -\ Classify-framework-repositories-during-Library-skill-import.md
git commit -m "feat: classify Library skill packages"
```

## Plan-level self-review gate

- A second submit can neither replace the UI await nor create a hidden landed
  import.
- Disabled controls are backed by handler guards.
- Classification performs no import, trust grant, script execution, or project
  activation.
- Multi-skill repositories require an exact human choice.
- Framework recovery is generic and does not create a Chatbook integration.
- Every successful import remains trust-pending and visibly attributable to
  the selected skill name.
- Network and filesystem failures are bounded and redacted.
