# Personal Context Profile Chatbook Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish accurate, discoverable Chatbook user and developer documentation for the Personal Context Profile without advertising unshipped synchronization behavior.

**Architecture:** Keep the existing Settings guide as the canonical user reference, add task-oriented entry points and troubleshooting, and add one focused developer guide for Chatbook-owned implementation details. Link to the already-published server guides for server-owned behavior instead of duplicating them; preserve the reviewed distinction between Shared Core models, Sync-v2 transport, and current product limitations.

**Tech Stack:** Markdown, Backlog.md, Git, GitHub, existing Python/pytest contract checks

**Backlog task:** TASK-27019

**Design specification:** `Docs/superpowers/specs/2026-08-31-personal-context-documentation-design.md`

---

## File map

**Create**

- `Docs/Development/personal-context-profile.md` — canonical Chatbook developer guide for the profile service, encrypted repository, interviews, agent tools, context injection, and Sync-v2 client boundary.

**Modify**

- `Docs/User_Guide/settings/personal-context-profile.md` — add a quick start, workflows, shipped-behavior synchronization table, troubleshooting, and server links without duplicating the existing detailed reference.
- `Docs/User_Guide/index.md` — add the Personal Context guide to the how-to table.
- `Docs/Development/Developer_Guide.md` — add a concise pointer to the focused guide.
- `Docs/superpowers/specs/2026-08-31-personal-context-documentation-design.md` — correct review-status metadata only after the shipped-behavior correction has been reviewed and merged; do not alter the authoritative TASK-27016 record.
- `Docs/superpowers/plans/2026-09-01-personal-context-documentation-chatbook.md` — this executable plan.
- `backlog/tasks/task-27019 - Document-Personal-Context-Profile-for-Chatbook-users-and-developers.md` — plan, acceptance criteria, evidence, ADR result, and implementation notes.

**Inspect but normally do not modify**

- `Docs/User_Guide/settings.md` — already links **Data & Privacy > My Profile** to the canonical guide.
- `Docs/Development/Sync-v2-client.md` — generic client transport reference.
- `Docs/superpowers/specs/2026-08-28-unified-personal-context-profile-design.md` — accepted architecture, including future behavior that must not be presented as shipped.
- `backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md` — governing ADR.

## Cross-repository execution prerequisite

Completed before Chatbook execution. The server documentation landed through PR [#2858](https://github.com/rmusser01/tldw_server/pull/2858), merged to server `dev` as `c85fb8db6b6efc338162276a52a193fc5d2d0ce5` on 2026-09-01. GitHub Contents API verification on 2026-09-01 confirmed these stable targets:

- `https://github.com/rmusser01/tldw_server/blob/dev/Docs/User_Guides/Server/Personal_Context_Profile.md` (`cc238b007d531a491519cafcc9eeff0708d1c959`)
- `https://github.com/rmusser01/tldw_server/blob/dev/Docs/Code_Documentation/Personal_Context_Developer_Guide.md` (`eb47613706fe7979442f7a5c40e7a81a4ee478ff`)
- `https://github.com/rmusser01/tldw_server/blob/dev/Docs/API-related/Personal_Context_API.md` (`163bea3315b9a6708a62f04f632f8f477c2de355`)

The Chatbook approved specification and publication history are already on `dev`: PR #2292 published the design under authoritative TASK-27016, and PR #2294 corrected its final evidence. The stale younger TASK-26836 publication record from this branch was dropped during rebase; the older TASK-26836 Console tray record and authoritative TASK-27016 remain unchanged.

## Corrected shipped-behavior claim inventory (PR #2310)

This inventory supersedes the older continuous-sync assumptions in the first documentation pass and is the fail-closed source for Tasks 2-5:

- Reviewed first linking is the only shipped Personal Context publication path. It publishes the eligible canonical snapshot resulting from the user's approved content-free reconciliation plan. Later syncable Chatbook changes create encrypted Personal Context outbox entries, but no shipped ongoing Personal Context caller drains them. **Manual Sync** covers Notes and Chat only. Ordinary server REST changes are not published to Chatbook.
- Setup completes before the optional chained interview. Leaving **Get to know you after setup** unchecked is the setup-only opt-out and stores no interview answers. Within an interview, **Skip** skips only the current question. **Cancel** opens **Leave interview**, where **Keep draft**, **Discard draft**, and **Continue interview** determine whether the encrypted draft is retained and whether the interview exits.
- The fixed interview is local. The adaptive interview uses the default Console provider and model with tools disabled. Each request includes the audience, coverage topics, attempt number, and eligible records from the selected scope; after the first answer, it also includes every prior answered turn and raw answer text. The actual provider/model is shown only after the first provider response completes, before answer input.
- Interview draft and transcript objects are not Sync payloads. Approved answer text may become an ordinary canonical record, after which the record's controls determine first-link eligibility.
- Chatbook accepts HTTP and HTTPS home-server URLs. HTTP is unencrypted. HTTPS provides transport privacy when a valid certificate is verified through default trust or a correctly configured custom CA. Disabling verification removes server authentication and permits interception. Runtime calls honor the saved default verification, custom CA bundle, or verification-off setting. **Test Connection** uses the HTTP client's default certificate verification rather than the saved custom/off setting.
- Before approval, first-link bootstrap exchanges metadata and downloads sync-eligible server records and proposals into transient memory. The review UI and durable review state are content-free, and no local profile content uploads before approval.
- **Remove local profile** removes the canonical profile repository and canonical profile outbox. Separate Sync state, staged encrypted envelopes, and staging keys can remain; the action neither deletes the server copy nor unregisters the device. If key cleanup fails, use **Finish secure removal**. Recovery export has no shipped import/restore path.
- Chatbook does not expose **Delete everywhere**. Authenticated server purge leaves a server-local fence in `purge_pending`; Sync distribution and acknowledgement completion are not wired end to end.
- First-link version conflicts and semantic collisions are resolved in content-free review with **Keep this device** or **Keep server** lineage choices. Later version or semantic conflicts can retain generic Sync metadata, but no shipped ongoing Personal Context cycle, status surface, or dedicated Personal Context resolver is available; first-link choices are not a post-link resolver.
- The only current Console preview route is **Ctrl+Shift+P** (**View context**) > **Conversation Inspector** > outer **Next Send** > inner **Next Send** payload tab.

**Ordered-merge dependency:** the stable server guide URLs temporarily retain the older continuous-sync wording. Final cross-repository parity and link-content checks remain blocked until the already-approved server correction branch is merged; the Chatbook guide must use the corrected PR #2310 specification in the meantime.

### Task 1: Rebase and establish the shipped-behavior claim inventory

**Files:**

- Inspect: all paths in the file map
- Modify: `backlog/tasks/task-27019 - Document-Personal-Context-Profile-for-Chatbook-users-and-developers.md`

- [x] **Step 1: Rebase the isolated branch on current `dev`**

Run:

```bash
set -e -o pipefail
git fetch origin dev
git rebase origin/dev
```

Expected: the branch rebases cleanly without unrelated working-tree changes.

- [x] **Step 2: Verify Backlog ownership and read the applicable workflow lessons**

Run:

```bash
set -e -o pipefail
backlog task 27019 --plain
rg -n "TASK-27019|Document Personal Context Profile for Chatbook" \
  "backlog/tasks/task-27019 - Document-Personal-Context-Profile-for-Chatbook-users-and-developers.md"
sed -n '1,240p' backlog/docs/lessons-testing-evidence.md
sed -n '1,220p' backlog/docs/lessons-backlog-hygiene.md
```

Expected: the task resolves to this documentation file, is assigned to `@codex`, and no duplicate ID or title appears. TASK-27019 replaces this task's younger TASK-26835 claim; current `dev` retains the older 2026-09-01 14:27 TASK-26835 Console evidence task. Repeat the task-resolution check after every rebase.

Run this all-ref/all-worktree collision sweep now and after the final rebase:

```bash
set -e -o pipefail
profile_task_matches=$(
  {
    git for-each-ref --format='%(refname)' refs/heads refs/remotes |
      while IFS= read -r profile_ref; do
        if profile_ref_match=$(git grep -l -E \
          '^id: TASK-27019$|^title: Document Personal Context Profile for Chatbook users and developers$' \
          "$profile_ref" -- 'backlog/tasks/*.md' 2>/dev/null); then
          printf '%s\n' "$profile_ref_match"
        else
          profile_ref_status=$?
          test "$profile_ref_status" -eq 1 || exit "$profile_ref_status"
        fi
      done | sed 's/^[^:]*://'
    git worktree list --porcelain |
      awk '$1 == "worktree" { sub(/^worktree /, ""); print }' |
      while IFS= read -r profile_worktree; do
        if [ ! -d "$profile_worktree/backlog/tasks" ]; then
          continue
        fi
        if profile_worktree_match=$(rg -l -g '*.md' \
          '^id: TASK-27019$|^title: Document Personal Context Profile for Chatbook users and developers$' \
          "$profile_worktree/backlog/tasks" 2>/dev/null); then
          printf '%s\n' "$profile_worktree_match"
        else
          profile_worktree_status=$?
          test "$profile_worktree_status" -eq 1 || exit "$profile_worktree_status"
        fi
      done
  } | awk -F/ '{ print $NF }' | sort -u
) || {
  echo "TASK-27019 collision sweep failed"
  exit 1
}
printf '%s\n' "$profile_task_matches"
test "$profile_task_matches" = "task-27019 - Document-Personal-Context-Profile-for-Chatbook-users-and-developers.md"
```

Expected: the only unique matching task filename by either ID or title is the intended TASK-27019 record. Any scanner error fails the command instead of being converted into a no-match result.

- [x] **Step 3: Confirm merged UI and service boundaries**

Run:

```bash
set -e -o pipefail
rg -Fq 'Remove local profile' tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py
rg -Fq 'Run interview again' tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py
rg -Fq 'Get to know you' tldw_chatbook/UI/Screens/profile_interview_screen.py
rg -Fq 'Define project context after creating' tldw_chatbook/Widgets/workspace_create_modal.py
profile_common_dir=$(git rev-parse --path-format=absolute --git-common-dir)
profile_repo_root=$(dirname "$profile_common_dir")
profile_python=
for profile_python_candidate in \
  "$profile_repo_root/.venv/bin/python" \
  "${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}" \
  "$PWD/.venv/bin/python"; do
  if [ -n "$profile_python_candidate" ] && [ -x "$profile_python_candidate" ]; then
    profile_python=$profile_python_candidate
    break
  fi
done
test -n "$profile_python"
"$profile_python" - <<'PY'
import sys

if not ((3, 11) <= sys.version_info[:2] < (4, 0)):
    raise SystemExit(f"Unsupported Python: {sys.version.split()[0]}")
PY
profile_ui_files=(
  tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py
  tldw_chatbook/Widgets/Settings_Widgets/personal_context_link_modal.py
  tldw_chatbook/Widgets/Settings_Widgets/personal_context_review_modal.py
  tldw_chatbook/UI/Screens/profile_interview_screen.py
  tldw_chatbook/Widgets/workspace_create_modal.py
)
"$profile_python" - "${profile_ui_files[@]}" <<'PY'
import ast
import re
import sys
from pathlib import Path

CONTROL_CALLS = {"action", "binding", "button", "label", "menuitem"}
CONTROL_KEYWORDS = {"action", "id", "label", "name", "title"}
violations: list[str] = []


def call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id.lower()
    if isinstance(call.func, ast.Attribute):
        return call.func.attr.lower()
    return ""


sources: list[tuple[str, ast.AST]] = [
    (
        "<negative-control>",
        ast.parse(
            "def action_delete_everywhere(self):\n"
            "    yield Button('Delete everywhere', id='profile-purge')\n"
        ),
    )
]
sources.extend(
    (
        raw_path,
        ast.parse(
            Path(raw_path).read_text(encoding="utf-8"), filename=raw_path
        ),
    )
    for raw_path in sys.argv[1:]
)
for path, tree in sources:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            normalized = node.name.lower()
            if normalized.startswith(("action_", "handle_", "on_")) and re.search(
                r"(?:delete_?everywhere|purge)", normalized
            ):
                violations.append(f"{path}:{node.lineno}: purge/Delete handler")
        if isinstance(node, (ast.Name, ast.Attribute)):
            identifier = node.id if isinstance(node, ast.Name) else node.attr
            if "delete_everywhere" in identifier.lower():
                violations.append(f"{path}:{node.lineno}: Delete-everywhere identifier")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if re.fullmatch(r"\s*delete[ _-]*everywhere[.!]?\s*", node.value, re.I):
                violations.append(f"{path}:{node.lineno}: visible Delete-everywhere label")
            if node.value.startswith("#") and "purge" in node.value.lower():
                violations.append(f"{path}:{node.lineno}: purge control selector")
        if not isinstance(node, ast.Call):
            continue
        control = call_name(node) in CONTROL_CALLS
        values = list(node.args) if control else []
        values.extend(
            keyword.value
            for keyword in node.keywords
            if keyword.arg in CONTROL_KEYWORDS
        )
        for value in values:
            for child in ast.walk(value):
                if (
                    isinstance(child, ast.Constant)
                    and isinstance(child.value, str)
                    and re.search(r"(?:delete[ _-]*everywhere|purge)", child.value, re.I)
                ):
                    violations.append(f"{path}:{node.lineno}: purge/Delete control")
negative_violations = [
    violation for violation in violations if violation.startswith("<negative-control>")
]
if len(negative_violations) < 3:
    raise SystemExit("Personal Context UI negative control was not detected")
production_violations = [
    violation for violation in violations if not violation.startswith("<negative-control>")
]
if production_violations:
    raise SystemExit("\n".join(sorted(set(production_violations))))
print("Negative control passed; no Personal Context Delete-everywhere or purge UI control found.")
PY
while IFS='|' read -r profile_component profile_symbol; do
  test -f "$profile_component"
  rg -Fq "$profile_symbol" "$profile_component"
done <<'EOF'
tldw_chatbook/Personal_Context/bootstrap.py|def bootstrap_personal_context_service
tldw_chatbook/Personal_Context/key_protector.py|class ProfileKeyProtector
tldw_chatbook/Personal_Context/repository.py|class PersonalContextRepository
tldw_chatbook/Personal_Context/service.py|class PersonalContextService
tldw_chatbook/Personal_Context/context_service.py|class ProfileContextService
tldw_chatbook/Personal_Context/proposal_service.py|class ProfileProposalService
tldw_chatbook/Personal_Context/runtime_policy.py|class AgentAuthority
tldw_chatbook/Personal_Context/interview_coordinator.py|class ProfileInterviewCoordinator
tldw_chatbook/Personal_Context/interview_draft_repository.py|class InterviewDraftRepository
tldw_chatbook/Personal_Context/interview_provider.py|class InterviewQuestionProvider
tldw_chatbook/Personal_Context/link_service.py|class PersonalContextLinkService
tldw_chatbook/Personal_Context/link_key_custody.py|class PersonalContextLinkKeyCustodian
tldw_chatbook/Personal_Context/sync_outbox.py|class ProfileSyncOutbox
tldw_chatbook/Sync_Interop/personal_context_adapter.py|class PersonalContextSyncAdapter
tldw_chatbook/Sync_Interop/personal_context_dispatcher.py|class PersonalContextOutboxDispatcher
tldw_chatbook/Sync_Interop/personal_context_first_link_sync.py|class PersonalContextFirstLinkSync
tldw_chatbook/tldw_api/client.py|async def bootstrap_sync_v2_personal_context
tldw_chatbook/tldw_api/client.py|async def complete_sync_v2_personal_context_link
tldw_chatbook/Agents/profile_tool_provider.py|class ProfileToolProvider
tldw_chatbook/Chat/console_chat_controller.py|class ConsoleChatController
tldw_chatbook/Chat/console_agent_bridge.py|class ConsoleAgentBridge
tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py|class PersonalContextSettingsPanel
tldw_chatbook/Widgets/Settings_Widgets/personal_context_link_modal.py|class PersonalContextLinkModal
tldw_chatbook/Widgets/Settings_Widgets/personal_context_review_modal.py|class PersonalContextReviewModal
tldw_chatbook/UI/Screens/profile_interview_screen.py|class ProfileInterviewScreen
EOF
```

Expected: removal, interview, and service surfaces exist; no shipped Chatbook **Delete everywhere** control is found.

- [x] **Step 4: Confirm Sync-v2 domains and current gaps**

Run:

```bash
set -e -o pipefail
for profile_domain in \
  personal_context.manifest \
  personal_context.scope \
  personal_context.record \
  personal_context.proposal \
  personal_context.purge; do
  rg -Fq "\"$profile_domain\"" tldw_chatbook/Sync_Interop/personal_context_first_link_sync.py
done
rg -Fq 'Require explicit review before any canonical profile apply or upload.' \
  tldw_chatbook/Widgets/Settings_Widgets/personal_context_link_modal.py
rg -Fq 'class PersonalContextOutboxDispatcher' \
  tldw_chatbook/Sync_Interop/personal_context_dispatcher.py
rg -Fq 'async def bootstrap_sync_v2_personal_context' tldw_chatbook/tldw_api/client.py
rg -Fq 'async def complete_sync_v2_personal_context_link' tldw_chatbook/tldw_api/client.py
profile_common_dir=$(git rev-parse --path-format=absolute --git-common-dir)
profile_repo_root=$(dirname "$profile_common_dir")
profile_python=
for profile_python_candidate in \
  "$profile_repo_root/.venv/bin/python" \
  "${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}" \
  "$PWD/.venv/bin/python"; do
  if [ -n "$profile_python_candidate" ] && [ -x "$profile_python_candidate" ]; then
    profile_python=$profile_python_candidate
    break
  fi
done
test -n "$profile_python"
"$profile_python" - <<'PY'
import sys

if not ((3, 11) <= sys.version_info[:2] < (4, 0)):
    raise SystemExit(f"Unsupported Python: {sys.version.split()[0]}")
PY
"$profile_python" - <<'PY'
import ast
from pathlib import Path

ROOT = Path("tldw_chatbook")
REPOSITORY = ROOT / "Personal_Context/repository.py"
ADAPTER = ROOT / "Sync_Interop/personal_context_adapter.py"
EXPECTED = {"manifest", "scope", "record", "proposal"}
PRODUCERS = {"_insert_outbox", "commit_outbox_body"}
EXPECTED_MATERIALIZATION = [
    ("extend", "scope"),
    ("extend", "record"),
    ("extend", "record"),
    ("extend", "proposal"),
]


def name(call: ast.Call) -> str:
    return call.func.attr if isinstance(call.func, ast.Attribute) else (
        call.func.id if isinstance(call.func, ast.Name) else ""
    )


def literal_types(call: ast.Call) -> set[str]:
    values = [*call.args, *(kw.value for kw in call.keywords if kw.arg == "object_type")]
    return {
        value.value
        for value in values
        if isinstance(value, ast.Constant) and isinstance(value.value, str)
    }


def purge_calls(tree: ast.AST) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and name(node) in PRODUCERS
        and "purge" in literal_types(node)
    ]


def targets_materialization(target: ast.AST) -> bool:
    if isinstance(target, ast.Name):
        return target.id == "materialization"
    if isinstance(target, ast.Subscript):
        return targets_materialization(target.value)
    if isinstance(target, (ast.List, ast.Tuple)):
        return any(targets_materialization(item) for item in target.elts)
    return False


def materialization_sequence(function: ast.FunctionDef) -> list[tuple[str, str]]:
    assignments: list[ast.AST] = []
    for node in ast.walk(function):
        if isinstance(node, ast.AnnAssign) and targets_materialization(node.target):
            assignments.append(node)
        elif isinstance(node, ast.Assign) and any(
            targets_materialization(target) for target in node.targets
        ):
            assignments.append(node)
        elif isinstance(node, ast.AugAssign) and targets_materialization(node.target):
            assignments.append(node)
        elif isinstance(node, ast.Delete) and any(
            targets_materialization(target) for target in node.targets
        ):
            assignments.append(node)
    if (
        len(assignments) != 1
        or not isinstance(assignments[0], ast.AnnAssign)
        or not isinstance(assignments[0].value, ast.List)
        or assignments[0].value.elts
    ):
        raise ValueError("materialization initialization or reassignment changed")

    sequence: list[tuple[str, str]] = []
    calls = sorted(
        (
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "materialization"
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    for call in calls:
        method = call.func.attr
        if method not in {"extend", "append"} or len(call.args) != 1 or call.keywords:
            raise ValueError(f"unknown materialization mutator at line {call.lineno}")
        source = call.args[0]
        if method == "append":
            items = [source]
        elif isinstance(source, (ast.GeneratorExp, ast.ListComp, ast.SetComp)):
            items = [source.elt]
        elif isinstance(source, (ast.List, ast.Tuple, ast.Set)):
            items = list(source.elts)
        else:
            raise ValueError(f"dynamic materialization source at line {call.lineno}")
        for item in items:
            if (
                not isinstance(item, ast.Tuple)
                or not item.elts
                or not isinstance(item.elts[0], ast.Constant)
                or not isinstance(item.elts[0].value, str)
            ):
                raise ValueError(f"dynamic materialization domain at line {call.lineno}")
            domain = item.elts[0].value
            if domain not in EXPECTED:
                raise ValueError(f"unreviewed materialization domain: {domain}")
            sequence.append((method, domain))
    return sequence


widget_tree = ast.parse(
    "def on_click(repository):\n"
    "    repository.commit_outbox_body(object_type='purge')\n"
)
if purge_calls(widget_tree) != [2]:
    raise SystemExit("Widget-like purge caller negative control was not detected")
synthetic_function = next(
    node
    for node in ast.walk(
        ast.parse(
            "def materialize(items):\n"
            "    materialization: list[tuple] = []\n"
            "    materialization.extend(('purge', item, item, item) for item in items)\n"
        )
    )
    if isinstance(node, ast.FunctionDef)
)
try:
    materialization_sequence(synthetic_function)
except ValueError as error:
    if "purge" not in str(error):
        raise
else:
    raise SystemExit("purge materialization negative control was not rejected")

trees = {
    path: ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for path in ROOT.rglob("*.py")
}
violations: list[str] = []
literal_producers: set[str] = set()
dynamic_insertions: set[tuple[str, str, str]] = set()
direct_commit_calls: list[str] = []
for path, tree in trees.items():
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        producer = name(call)
        if producer not in PRODUCERS:
            continue
        types = literal_types(call)
        literal_producers.update(types)
        if "purge" in types:
            violations.append(f"{path}:{call.lineno}: literal purge producer")
        owner = call
        while owner in parents and not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
            owner = parents[owner]
        owner_name = owner.name if isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)) else "<module>"
        object_kw = next((kw.value for kw in call.keywords if kw.arg == "object_type"), None)
        if producer == "_insert_outbox" and object_kw is not None and not isinstance(object_kw, ast.Constant):
            dynamic_insertions.add((str(path), owner_name, ast.unparse(object_kw)))
        if producer == "commit_outbox_body":
            direct_commit_calls.append(f"{path}:{call.lineno}")

if literal_producers != EXPECTED:
    violations.append(f"literal producer domains changed: {sorted(literal_producers)}")
expected_dynamic = {
    (str(REPOSITORY), "apply_reviewed_link", "object_type"),
    (str(REPOSITORY), "commit_outbox_body", "object_type"),
}
if dynamic_insertions != expected_dynamic:
    violations.append(f"dynamic producer seams changed: {sorted(dynamic_insertions)}")
if direct_commit_calls:
    violations.append(f"commit_outbox_body gained callers: {direct_commit_calls}")

repository = trees[REPOSITORY]
reviewed_link_functions = [
    node
    for node in ast.walk(repository)
    if isinstance(node, ast.FunctionDef) and node.name == "apply_reviewed_link"
]
if len(reviewed_link_functions) != 1:
    violations.append("PersonalContextRepository.apply_reviewed_link changed or is ambiguous")
else:
    try:
        materialization = materialization_sequence(reviewed_link_functions[0])
    except ValueError as error:
        violations.append(str(error))
    else:
        if materialization != EXPECTED_MATERIALIZATION:
            violations.append(f"materialization sources changed: {materialization}")

adapter = trees[ADAPTER]
model_keys = next(
    {
        key.value
        for key in node.value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    for node in ast.walk(adapter)
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "_MODELS" for target in node.targets)
    and isinstance(node.value, ast.Dict)
)
if model_keys != EXPECTED:
    violations.append(f"adapter publishable model map changed: {sorted(model_keys)}")
if violations:
    raise SystemExit("\n".join(violations))
print(f"Negative controls passed; no production purge caller; materialization: {materialization}")
PY
if rg -n -i -e 'personal.context.*post.?link.*resolve' \
  -e 'post.?link.*personal.context.*resolve' \
  tldw_chatbook/Widgets/Settings_Widgets tldw_chatbook/UI/Screens/profile_interview_screen.py; then
  echo 'Unexpected dedicated Personal Context post-link resolver'
  exit 1
fi
```

Expected: five protocol domains and reviewed first-link behavior exist; no dedicated post-link resolver, literal purge caller anywhere under `tldw_chatbook`, production `commit_outbox_body` caller, or unreviewed first-link materialization source is found.

Executed inventory on rebased Chatbook `dev` `862bfaf9c18795f6a41bcda626ed25e66f8319d2` confirmed the named controls and component paths; all five domains; reviewed first-link reconciliation; encrypted `ProfileSyncOutbox` dispatch; API bootstrap/link completion; generic Sync conflict handling only; and outbox producers for manifest, scope, record, and proposal, with no purge producer. Merged server PR #2858 independently records that ordinary server REST edits are not published to linked clients and that purge distribution/acknowledgement remain incomplete.

- [x] **Step 5: Record the plan and ADR result in TASK-27019**

Run:

```bash
backlog task edit 27019 --plan $'1. Rebase/inventory shipped behavior.\n2. Task-oriented user guide.\n3. Focused developer guide.\n4. Discovery/server links.\n5. Final targeted contract/link/diff verification.\n6. Complete notes/open docs-only PR.\n\nADR required: no new ADR required; existing ADR applies\nADR path: backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md\nReason: Documentation only; the existing Personal Context authority, Sync, and encryption ADR applies.'
```

Expected: task remains **In Progress** with an implementation plan and ADR check.

- [x] **Step 6: Commit the plan and task metadata**

Run:

```bash
set -e -o pipefail
git add \
  Docs/superpowers/plans/2026-09-01-personal-context-documentation-chatbook.md \
  "backlog/tasks/task-27019 - Document-Personal-Context-Profile-for-Chatbook-users-and-developers.md"
git commit -m "docs: plan Chatbook Personal Context guides"
```

### Task 2: Make the user guide task-oriented and release-accurate

**Files:**

- Modify: `Docs/User_Guide/settings/personal-context-profile.md`
- Modify: `Docs/superpowers/plans/2026-09-01-personal-context-documentation-chatbook.md`
- Modify: `backlog/tasks/task-27019 - Document-Personal-Context-Profile-for-Chatbook-users-and-developers.md`

- [x] **Step 1: Correct the quick start after `Getting there`**

Wrap the section in `<!-- personal-context-quick-start:start -->` and `<!-- personal-context-quick-start:end -->`. It must contain exactly five numbered steps covering:

1. Open **F9 > Data & Privacy > My Profile**; separately note that leaving **Get to know you after setup** unchecked is available only during initial setup.
2. For manual entry, use **Add** or **Edit**, review scope, visibility, and syncability, then choose **Save**.
3. For an interview, select the scope and question style, run the interview, review every proposed row and its controls, then choose **Save only** or **Save and use with agents**.
4. Keep agent use optional and inspect the planned payload through **Ctrl+Shift+P** (**View context**) > **Conversation Inspector** > outer **Next Send** > inner **Next Send** payload tab.
5. Link a supported home server only when sharing is desired.

- [x] **Step 2: Correct `Common workflows` and current controls**

Under `## Common workflows`, use these exact subheadings so each workflow is independently verifiable without a redundant intermediate heading:

- `### Edit manually`
- `### Run or rerun an interview`
- `### Review agent proposals`
- `### Export plaintext and recovery material`
- `### Remove the local copy`
- `### Link a home server`

Cover global preferences and workspace goals/conventions across those workflows. Use current control names. The manual editor chooses a workspace scope inside **Add record**, while the interview first chooses a linked scope and mode. State that new inferred facts remain proposals; direct write only updates an existing eligible record for an explicit correction evidenced by the current persisted user message.

- [x] **Step 3: Add `What first linking publishes, and what does not sync afterward`**

Include this deliberately identical shared-contract block, with the markers retained so cross-repository parity can be checked automatically:

```markdown
<!-- shared-personal-context-contract:start -->
- `tldw_profile_core` defines the versioned canonical profile object models, exact canonical bytes, interview/tool contracts, serialization, and validation used by both peers. Sync-v2 transport envelopes are a separate contract.
- After successful reviewed first linking, Chatbook and tldw_server converge on the same canonical manifest, scope, record, proposal, and version identities and bytes for the eligible snapshot resulting from the user-approved content-free reconciliation plan.
- Sync V2 defines the `personal_context.manifest`, `personal_context.scope`, `personal_context.record`, `personal_context.proposal`, and content-free `personal_context.purge` domains. Reviewed first linking publishes the eligible snapshot resulting from the user-approved content-free reconciliation plan. Later syncable Chatbook mutations create encrypted local outbox entries, but the current shipped app does not run an ongoing Personal Context sync cycle, so those post-link changes remain queued locally. Purge production and distribution are not wired end to end.
- Each peer retains its own at-rest ciphertext and keys, local database rows, runtime permissions, conflict-review metadata, acknowledgement tracking, and other operational state.
<!-- shared-personal-context-contract:end -->
```

Follow it with this full matrix, wrapped in `<!-- personal-context-boundary-matrix:start -->` and `<!-- personal-context-boundary-matrix:end -->`. Keep its cell wording compact while retaining every category:

| Published at reviewed first link when eligible | Not published afterward or peer-local |
| --- | --- |
| Approved eligible canonical manifest | Later syncable Chatbook mutations, which remain queued locally |
| Required global and linked-workspace scopes | Ordinary server REST mutations |
| Controls-eligible record heads and tombstones; eligible proposals and canonical review state; approved interview answers after they are saved as records | Device-only or non-syncable records |
| Exact canonical IDs, versions, and bytes | Runtime agent authority, tool availability, workspace mappings, and enablement |
| — | At-rest and recovery keys; local undo, caches, ciphertext, database row IDs, conflict-review objects, acknowledgement tracking, and operational metadata |
| — | Interview draft and transcript objects; adaptive requests still send prior raw answers to the provider |

Required notes:

- **Manual Sync** sends Notes and Chat changes only; it does not drain the Personal Context outbox.
- Ordinary server REST edits are not published to linked Chatbook clients.
- `personal_context.purge` exists at the protocol boundary, but Chatbook has no producer and the server endpoint does not distribute it through Sync V2.

- [x] **Step 4: Correct interview, linking, TLS, export, and removal detail**

Document these shipped boundaries in user language:

- Fixed interviews stay local. Adaptive interviews use the default Console provider/model with tools off and send the audience, topics, attempt, eligible selected-scope records, and, after the first answer, all prior answered turns including raw answer text. The UI shows the actual provider/model only after the first response and before answer input. **Cancel** opens **Leave interview**, where **Continue interview**, **Keep draft**, and **Discard draft** control exit and retention; memory-only drafts cannot be kept.
- Link from **Settings > Overview > Advanced / Diagnostics > Switch Source / Server**, then return to **Data & Privacy > My Profile > Server sync > Link to home server**. HTTP and HTTPS are accepted, but HTTP is unencrypted. HTTPS protects transport privacy only with valid certificate verification; **Disable verification** removes server authentication and permits interception. Runtime calls honor default verification, custom CA, or verification off; **Test Connection** always uses default certificate verification.
- The current payload-preview route is **Ctrl+Shift+P** (**View context**) > **Conversation Inspector** > outer **Next Send** > inner **Next Send** payload tab.
- Before approval, bootstrap exchanges metadata and downloads eligible server records/proposals into memory. The durable review and screen remain content-free, and no local profile content uploads before approval.
- Plaintext export and recovery export are separate. Recovery export includes canonical local heads, including device-only records, but Chatbook has no shipped recovery import/restore flow.
- **Remove local profile** removes canonical profile rows and the canonical profile outbox. It can leave separate Sync state, staged encrypted envelopes, and staging keys; it does not delete the server copy or unregister the device. If key cleanup fails, use **Finish secure removal**.

Keep **Remove local profile** as the available Chatbook action. State plainly:

> Chatbook does not currently expose **Delete everywhere**. The authenticated server purge endpoint creates a server-local purge fence and remains `purge_pending`; distribution and acknowledgement completion are not wired end to end.

Do not tell users that reconnecting devices clears `purge_pending`.

- [x] **Step 5: Replace troubleshooting with the corrected shipped states**

Under `### Troubleshooting`, add a table wrapped in `<!-- personal-context-troubleshooting:start -->` and `<!-- personal-context-troubleshooting:end -->`. Use the exact header `| State | Cause | Safe next action | Current limit |`. Each of these exact eleven bold state labels must have non-empty cells for cause, safe next action, and current product limit:

1. **Profile locked**
2. **Adaptive interview privacy or provider failure**
3. **HTTP or altered TLS verification**
4. **Post-link change queued**
5. **Capability not negotiated**
6. **First-link publication interrupted**
7. **Version conflict**
8. **First-link semantic collision**
9. **Post-link semantic collision**
10. **Local removal incomplete or residual state**
11. **Purge pending**

Do not recommend **Manual Sync** or a Personal Context status screen. Explain that later local edits and ordinary server REST edits are not published by the shipped ongoing lifecycle. Treat **Version conflict** as a first-link content-free lineage choice using **Keep this device** or **Keep server**. Separate that from post-link version and semantic conflicts, which can retain generic Sync metadata but have no ongoing Personal Context cycle, status screen, or dedicated resolver.

- [x] **Step 6: Keep stable server links and concise operator guidance**

Link to:

- `https://github.com/rmusser01/tldw_server/blob/dev/Docs/User_Guides/Server/Personal_Context_Profile.md`
- `https://github.com/rmusser01/tldw_server/blob/dev/Docs/API-related/Personal_Context_API.md`

- [x] **Step 7: Run corrected claim, control, and diff guards**

Run:

```bash
set -e -o pipefail
profile_user_guide=Docs/User_Guide/settings/personal-context-profile.md
rg -n "does not currently expose|no shipped ongoing Personal Context|Manual Sync.*Notes and Chat|not published|not wired end to end|no dedicated Personal Context" \
  "$profile_user_guide"
rg -n "Fixed local questions|default Console provider|Test Connection|custom CA|Finish secure removal|no recovery import|content-free|Keep this device|Keep server|View context|Conversation Inspector" \
  "$profile_user_guide"
if rg -n -i \
  -e 'retry (manual )?sync' \
  -e '(open|inspect|use).{0,30}personal context status' \
  -e 'delete everywhere.{0,40}(available|choose|select)' \
  -e 'ordinary server REST edits are (published|synced|synchronized)' \
  "$profile_user_guide"; then
  echo "Unsupported Personal Context guidance remains"
  exit 1
fi
git diff --check -- Docs/User_Guide/settings/personal-context-profile.md
git diff -- Docs/User_Guide/settings/personal-context-profile.md
```

Expected: current limitations are explicit and the existing reference is refined rather than duplicated.

- [x] **Step 8: Record Task 2 remediation and commit the user guide**

After Steps 1-7 have run successfully, mark every Task 2 step through this commit step `[x]`. Commit that execution record with the guide.

Run:

```bash
set -e -o pipefail
git add \
  Docs/User_Guide/settings/personal-context-profile.md \
  Docs/superpowers/plans/2026-09-01-personal-context-documentation-chatbook.md \
  "backlog/tasks/task-27019 - Document-Personal-Context-Profile-for-Chatbook-users-and-developers.md"
git diff --check --cached
git commit -m "docs: clarify Chatbook Personal Context workflows"
```

### Task 3: Add the focused developer guide

**Files:**

- Create: `Docs/Development/personal-context-profile.md`

- [x] **Step 1: Write contract and ownership sections**

Cover Shared Core `0.1.0`, separate Sync-v2 envelopes, reviewed first-link snapshot convergence, the absent ongoing Personal Context sync caller, peer-local at-rest keys/ciphertext, and wrapped server-owned Sync integrity-key bootstrap. Link relatively to:

- `../superpowers/specs/2026-08-28-unified-personal-context-profile-design.md`
- `../../backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md`
- `Sync-v2-client.md`
- `https://github.com/rmusser01/tldw_server/blob/dev/Docs/Code_Documentation/Personal_Context_Developer_Guide.md`
- `https://github.com/rmusser01/tldw_server/blob/dev/Docs/API-related/Personal_Context_API.md`

Include this exact four-bullet block, including both markers and every bullet; marker presence alone is insufficient:

```markdown
<!-- shared-personal-context-contract:start -->
- `tldw_profile_core` defines the versioned canonical profile object models, exact canonical bytes, interview/tool contracts, serialization, and validation used by both peers. Sync-v2 transport envelopes are a separate contract.
- After successful reviewed first linking, Chatbook and tldw_server converge on the same canonical manifest, scope, record, proposal, and version identities and bytes for the eligible snapshot resulting from the user-approved content-free reconciliation plan.
- Sync V2 defines the `personal_context.manifest`, `personal_context.scope`, `personal_context.record`, `personal_context.proposal`, and content-free `personal_context.purge` domains. Reviewed first linking publishes the eligible snapshot resulting from the user-approved content-free reconciliation plan. Later syncable Chatbook mutations create encrypted local outbox entries, but the current shipped app does not run an ongoing Personal Context sync cycle, so those post-link changes remain queued locally. Purge production and distribution are not wired end to end.
- Each peer retains its own at-rest ciphertext and keys, local database rows, runtime permissions, conflict-review metadata, acknowledgement tracking, and other operational state.
<!-- shared-personal-context-contract:end -->
```

- [x] **Step 2: Add the component map**

Document these exact owners, using repository-root paths:

- `tldw_chatbook/Personal_Context/bootstrap.py` — `bootstrap_personal_context_service`
- `tldw_chatbook/Personal_Context/key_protector.py` — `ProfileKeyProtector`, the local at-rest key protection and recovery boundary
- `tldw_chatbook/Personal_Context/repository.py` — `PersonalContextRepository`
- `tldw_chatbook/Personal_Context/service.py` — `PersonalContextService`
- `tldw_chatbook/Personal_Context/context_service.py` — `ProfileContextService`
- `tldw_chatbook/Personal_Context/proposal_service.py` — `ProfileProposalService`
- `tldw_chatbook/Personal_Context/runtime_policy.py` — `AgentAuthority`
- `tldw_chatbook/Personal_Context/interview_coordinator.py` — reviewed interview execution
- `tldw_chatbook/Personal_Context/interview_draft_repository.py` — unfinished interview-draft storage
- `tldw_chatbook/Personal_Context/interview_provider.py` — `InterviewQuestionProvider`, the interview model-provider boundary
- `tldw_chatbook/Personal_Context/link_service.py` — `PersonalContextLinkService`
- `tldw_chatbook/Personal_Context/link_key_custody.py` — `PersonalContextLinkKeyCustodian`, the wrapping/integrity-key custody boundary
- `tldw_chatbook/Personal_Context/sync_outbox.py` — encrypted `ProfileSyncOutbox` lifecycle boundary
- `tldw_chatbook/Sync_Interop/personal_context_adapter.py` — `PersonalContextSyncAdapter`
- `tldw_chatbook/Sync_Interop/personal_context_dispatcher.py` — `PersonalContextOutboxDispatcher`
- `tldw_chatbook/Sync_Interop/personal_context_first_link_sync.py` — `PersonalContextFirstLinkSync`
- `tldw_chatbook/tldw_api/client.py` — `bootstrap_sync_v2_personal_context` and `complete_sync_v2_personal_context_link`
- `tldw_chatbook/Agents/profile_tool_provider.py` — `ProfileToolProvider`
- `tldw_chatbook/Chat/console_chat_controller.py` — Console snapshot/context injection
- `tldw_chatbook/Chat/console_agent_bridge.py` — Console agent-tool bridge
- `tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py` — `PersonalContextSettingsPanel`, Settings presentation and user actions only
- `tldw_chatbook/Widgets/Settings_Widgets/personal_context_link_modal.py` — `PersonalContextLinkModal`, reviewed linking presentation only
- `tldw_chatbook/Widgets/Settings_Widgets/personal_context_review_modal.py` — `PersonalContextReviewModal`, proposal/review presentation only
- `tldw_chatbook/UI/Screens/profile_interview_screen.py` — `ProfileInterviewScreen`, interview presentation only

State exactly: `UI, agent, and transport code must use the owning service/repository boundary and must not access profile tables directly.`

- [x] **Step 3: Document read/write lifecycles and current gaps**

Use these exact lifecycle headings and cover the named behavior under each:

- `### Manual edits and reviewed interviews` — manual edits, reviewed interview output, immutable encrypted versions, controls, expiry, tombstones, and receipts.
- `### Proposals and direct writes` — proposals, review, and the narrow explicit-correction direct-write boundary.
- `### Context injection and Next Send` — context selection, runtime authority, Console injection, and the disposable **Next Send** preview.
- `### Transactional outbox and reviewed first link` — transactional Chatbook outbox and reviewed first linking.
- `### Post-link conflicts and purge limits` — generic post-link conflict metadata and protocol-only purge without end-to-end production, distribution, or acknowledgement.

Include these exact current-limit sentences so final verification can fail closed per document:

- `Ordinary server REST edits are not currently published to linked Chatbook clients.`
- `The Personal Context purge domain is protocol-only in the current linked flow: Chatbook has no producer, and end-to-end distribution and acknowledgement are not wired.`
- `Post-link conflicts retain generic Sync metadata but have no dedicated Personal Context resolution screen.`

Repeat the full boundary matrix in developer terms, wrapped in `<!-- personal-context-boundary-matrix:start -->` and `<!-- personal-context-boundary-matrix:end -->`, so every shared and peer-local category is explicit:

| Published at reviewed first link when eligible | Not published afterward or peer-local |
| --- | --- |
| Approved eligible canonical manifest | Later syncable Chatbook mutations, which remain queued locally |
| Required global and linked-workspace scopes | Ordinary server REST mutations |
| Controls-eligible record heads and tombstones; eligible proposals and canonical review state; approved interview answers after they are saved as records | Device-only or non-syncable records |
| Exact canonical IDs, versions, and bytes | Runtime agent authority, tool availability, workspace mappings, and enablement |
| — | At-rest and recovery keys; local undo, caches, ciphertext, database row IDs, conflict-review objects, acknowledgement tracking, and operational metadata |
| — | Interview draft and transcript objects; adaptive requests still send prior raw answers to the provider |

Include this exact privacy prohibition: `Never log profile plaintext, ciphertext, wrapped keys, or raw cryptographic errors.`

- [x] **Step 4: Add the complete extension checklist and test map**

Wrap exactly these ten numbered items in `<!-- personal-context-extension-checklist:start -->` and `<!-- personal-context-extension-checklist:end -->`:

1. Decide whether the integration is a full local-first Sync peer or a server/API-only client.
2. Make shared canonical object changes in `tldw_profile_core` first; change Sync transport separately.
3. Preserve canonical identities and explicit syncability whenever the integration persists or transports canonical objects.
4. Route full peers through their owning services; route API-only clients through authenticated public server APIs, never profile tables.
5. Enforce authority, scope, expiry, visibility, and secret-rejection rules at the boundary the integration owns.
6. Keep plaintext out of logs, diagnostics, outbox metadata, and unencrypted fixtures.
7. Add parity/conformance coverage for every shared-core or Sync contract the integration implements.
8. Test only the owned surface: full peers need storage, key, service, Sync, runtime/UI, and recovery coverage; API-only clients need authentication, request/response, error, and privacy coverage.
9. Update the governing ADR for storage, ownership, encryption, Sync, or authority changes.
10. Update both documentation sets whenever the shared contract changes.

Map:

- `Tests/Packaging/test_profile_core_packaging.py`
- `Tests/Personal_Context/`
- `Tests/Agents/test_personal_context_prompt.py`
- `Tests/Chat/test_console_personal_context_snapshot.py`
- `Tests/Sync_Interop/test_personal_context_*.py`
- `Tests/UI/test_settings_personal_context.py`
- `Tests/UI/test_personal_context_*.py`
- `Tests/tldw_api/test_personal_context_sync_client.py`

- [x] **Step 5: Validate referenced paths and Markdown**

Run:

```bash
set -e -o pipefail
test -f Docs/Development/Sync-v2-client.md
test -f Docs/superpowers/specs/2026-08-28-unified-personal-context-profile-design.md
test -f backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md
test -f tldw_chatbook/Personal_Context/key_protector.py
test -f tldw_chatbook/Personal_Context/interview_coordinator.py
test -f tldw_chatbook/Chat/console_chat_controller.py
test -f tldw_chatbook/Personal_Context/service.py
test -f tldw_chatbook/Sync_Interop/personal_context_adapter.py
test -f tldw_chatbook/Agents/profile_tool_provider.py
git diff --check -- Docs/Development/personal-context-profile.md
```

- [x] **Step 6: Record Task 3 execution and commit the developer guide**

After Steps 1-5 have run successfully, mark every Task 3 step through this commit step `[x]`. Commit that execution record with the guide.

Run:

```bash
set -e -o pipefail
git add \
  Docs/Development/personal-context-profile.md \
  Docs/superpowers/plans/2026-09-01-personal-context-documentation-chatbook.md
git diff --check --cached
git commit -m "docs: add Chatbook Personal Context developer guide"
```

#### Task 3 quality-correction addendum

**Allowed files:** `Docs/Development/personal-context-profile.md`, this plan, `Docs/superpowers/specs/2026-08-31-personal-context-documentation-design.md` review-status metadata, and TASK-27019. Preserve the approved user guide, the TASK-28228 collision correction, and authoritative TASK-27016 unchanged.

**Evidence:** `PersonalContextService.remove_local_profile()` and `finish_secure_removal()`, `PersonalContextRepository.destroy_profile_content()`, `_draft_repository()` and `InterviewDraftRepository`, `KeyringProfileKeyProtector`, the keyring wrapping/link custodians, and `ProfileContextService._ordered_with_workspace_overrides()` plus `_serialize_whole_records()`.

- [x] **Step 1: Govern the review-correction scope before changing specification metadata**
- [x] **Step 2: Correct removal residuals, exact context ordering, and integrator responsibilities**
- [x] **Step 3: Mark the already-reviewed and merged specification correction accurately without reopening TASK-27016**
- [x] **Step 4: Re-run Task 3 semantic/path/symbol/shared-block/privacy/task-ID/scope checks and commit only the allowed files**

### Task 4: Add discovery links

**Files:**

- Modify: `Docs/User_Guide/index.md`
- Modify: `Docs/Development/Developer_Guide.md`
- Inspect: `Docs/User_Guide/settings.md`

- [ ] **Step 1: Add the how-to row**

```markdown
| [Set up and manage your Personal Context Profile](settings/personal-context-profile.md) | Optional interviews, global/workspace context, agent proposals, synchronization boundaries, export, and removal. |
```

- [ ] **Step 2: Add the focused developer pointer**

Near the top of `Docs/Development/Developer_Guide.md`, add this exact pointer and do not add a second architecture summary:

```markdown
For Personal Context internals and extension work, see [Personal Context Profile](personal-context-profile.md).
```

- [ ] **Step 3: Confirm Settings already links the page**

Run:

```bash
set -e -o pipefail
rg -Fq '| **Applies immediately** | Each action takes effect at once; no draft to save or revert. | Workspaces, [My Profile](settings/personal-context-profile.md) |' \
  Docs/User_Guide/settings.md
rg -Fq '| Data & Privacy | **My Profile** → [own page](settings/personal-context-profile.md) | Personal and workspace context, interviews, agent proposals, authority, export, and removal. | Applies immediately |' \
  Docs/User_Guide/settings.md
```

Expected: both existing Settings links are present independently; leave `settings.md` unchanged.

- [ ] **Step 4: Validate discovery links, record Task 4 execution, and commit**

After Steps 1-3 have run successfully, mark every Task 4 step through this commit step `[x]`. Commit that execution record with the discovery indexes.

Run:

```bash
set -e -o pipefail
rg -Fq '| [Set up and manage your Personal Context Profile](settings/personal-context-profile.md) | Optional interviews, global/workspace context, agent proposals, synchronization boundaries, export, and removal. |' \
  Docs/User_Guide/index.md
rg -Fq 'For Personal Context internals and extension work, see [Personal Context Profile](personal-context-profile.md).' \
  Docs/Development/Developer_Guide.md
rg -Fq '| **Applies immediately** | Each action takes effect at once; no draft to save or revert. | Workspaces, [My Profile](settings/personal-context-profile.md) |' \
  Docs/User_Guide/settings.md
rg -Fq '| Data & Privacy | **My Profile** → [own page](settings/personal-context-profile.md) | Personal and workspace context, interviews, agent proposals, authority, export, and removal. | Applies immediately |' \
  Docs/User_Guide/settings.md
git diff --check -- Docs/User_Guide/index.md Docs/Development/Developer_Guide.md
git add \
  Docs/User_Guide/index.md \
  Docs/Development/Developer_Guide.md \
  Docs/superpowers/plans/2026-09-01-personal-context-documentation-chatbook.md
git diff --check --cached
git commit -m "docs: link Personal Context guides"
```

### Task 5: Final rebase and verification

**Files:**

- Verify: all changed documentation

- [ ] **Step 1: Perform the final rebase before closing the task**

Run:

```bash
set -e -o pipefail
git fetch origin dev
git rebase origin/dev
test "$(git merge-base origin/dev HEAD)" = "$(git rev-parse origin/dev)"
backlog task 27019 --plain
rg -n "TASK-27019|Document Personal Context Profile for Chatbook" \
  "backlog/tasks/task-27019 - Document-Personal-Context-Profile-for-Chatbook-users-and-developers.md"
profile_task_matches=$(
  {
    git for-each-ref --format='%(refname)' refs/heads refs/remotes |
      while IFS= read -r profile_ref; do
        if profile_ref_match=$(git grep -l -E \
          '^id: TASK-27019$|^title: Document Personal Context Profile for Chatbook users and developers$' \
          "$profile_ref" -- 'backlog/tasks/*.md' 2>/dev/null); then
          printf '%s\n' "$profile_ref_match"
        else
          profile_ref_status=$?
          test "$profile_ref_status" -eq 1 || exit "$profile_ref_status"
        fi
      done | sed 's/^[^:]*://'
    git worktree list --porcelain |
      awk '$1 == "worktree" { sub(/^worktree /, ""); print }' |
      while IFS= read -r profile_worktree; do
        if [ ! -d "$profile_worktree/backlog/tasks" ]; then
          continue
        fi
        if profile_worktree_match=$(rg -l -g '*.md' \
          '^id: TASK-27019$|^title: Document Personal Context Profile for Chatbook users and developers$' \
          "$profile_worktree/backlog/tasks" 2>/dev/null); then
          printf '%s\n' "$profile_worktree_match"
        else
          profile_worktree_status=$?
          test "$profile_worktree_status" -eq 1 || exit "$profile_worktree_status"
        fi
      done
  } | awk -F/ '{ print $NF }' | sort -u
) || {
  echo "TASK-27019 collision sweep failed"
  exit 1
}
printf '%s\n' "$profile_task_matches"
test "$profile_task_matches" = "task-27019 - Document-Personal-Context-Profile-for-Chatbook-users-and-developers.md"

# The exact UI/component and Sync/purge inventories are canonical in Task 1.
backlog task 27019 --plain | rg -q 'Status:.*In Progress'
```

Immediately after this rebase/collision block, rerun the exact Task 1 Step 3 command block and then the exact Task 1 Step 4 command block, unchanged. Those two canonical blocks are the required post-rebase UI/control, per-file symbol, Sync-domain, purge-producer, dynamic-materialization, and resolver inventory; do not substitute an abbreviated copy.

Expected: the branch is based on current `origin/dev`; TASK-27019 resolves uniquely; the current controls, components, five domains, reviewed first-link, exact `scope`, `record`, `record`, `proposal` materialization sequence, outbox/dispatcher/client boundaries, and negative purge/resolver claims still match the guides. Any production-tree scanner failure or newly shipped seam stops execution for re-inventory. There must be no later rebase after the task is marked Done.

- [ ] **Step 2: Verify server docs have landed on `dev`**

Run:

```bash
set -e -o pipefail
for server_doc in \
  Docs/User_Guides/Server/Personal_Context_Profile.md \
  Docs/Code_Documentation/Personal_Context_Developer_Guide.md \
  Docs/API-related/Personal_Context_API.md; do
  test "$(gh api -X GET "repos/rmusser01/tldw_server/contents/$server_doc" \
    -f ref=dev --jq .path)" = "$server_doc"
done
test "$(gh api -X GET repos/rmusser01/tldw_server/pulls/2858 --jq .merge_commit_sha)" = \
  'c85fb8db6b6efc338162276a52a193fc5d2d0ce5'
test "$(gh api -X GET repos/rmusser01/tldw_server/pulls/2858 --jq .base.ref)" = 'dev'
test -n "$(gh api -X GET repos/rmusser01/tldw_server/pulls/2858 --jq .merged_at)"
```

Expected: each command returns file metadata, and PR #2858 remains merged into `dev` at `c85fb8db6b6efc338162276a52a193fc5d2d0ce5`.

- [ ] **Step 3: Compare the shared contract block with server `dev`**

Run:

```bash
set -e -o pipefail
profile_common_dir=$(git rev-parse --path-format=absolute --git-common-dir)
profile_repo_root=$(dirname "$profile_common_dir")
profile_python=
for profile_python_candidate in \
  "$profile_repo_root/.venv/bin/python" \
  "${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}" \
  "$PWD/.venv/bin/python"; do
  if [ -n "$profile_python_candidate" ] && [ -x "$profile_python_candidate" ]; then
    profile_python=$profile_python_candidate
    break
  fi
done
test -n "$profile_python"
"$profile_python" - <<'PY'
import sys

if not ((3, 11) <= sys.version_info[:2] < (4, 0)):
    raise SystemExit(f"Unsupported Python: {sys.version.split()[0]}")
PY
profile_parity_dir=$(mktemp -d)
trap 'rm -r "$profile_parity_dir"' EXIT
gh api -X GET repos/rmusser01/tldw_server/contents/Docs/User_Guides/Server/Personal_Context_Profile.md \
  -f ref=dev --jq .content > "$profile_parity_dir/server.b64"
"$profile_python" - \
  Docs/User_Guide/settings/personal-context-profile.md \
  Docs/Development/personal-context-profile.md \
  "$profile_parity_dir/server.b64" <<'PY'
import base64
import re
import sys
from pathlib import Path

START = "<!-- shared-personal-context-contract:start -->"
END = "<!-- shared-personal-context-contract:end -->"
sources = [
    (raw_path, Path(raw_path).read_text(encoding="utf-8"))
    for raw_path in sys.argv[1:3]
]
server_payload = "".join(Path(sys.argv[3]).read_text(encoding="utf-8").split())
sources.append(
    (
        "tldw_server dev operator guide",
        base64.b64decode(server_payload, validate=True).decode("utf-8"),
    )
)
normalized: list[tuple[str, str]] = []
for label, text in sources:
    match = re.search(re.escape(START) + r"(.*?)" + re.escape(END), text, re.DOTALL)
    if match is None:
        raise SystemExit(f"missing shared-contract block: {label}")
    bullets = [line for line in match.group(1).splitlines() if line.startswith("- ")]
    if len(bullets) != 4:
        raise SystemExit(f"expected four shared-contract bullets in {label}, found {len(bullets)}")
    normalized.append((label, " ".join(match.group(0).split())))
baseline_path, baseline = normalized[0]
if not baseline:
    raise SystemExit(f"empty shared-contract block: {baseline_path}")
for path, block in normalized[1:]:
    if not block or block != baseline:
        raise SystemExit(f"shared-contract block diverges: {baseline_path} != {path}")
print("Shared-contract four-bullet block matches across both Chatbook guides and server dev.")
PY
```

Expected: every marked block is non-empty, contains exactly four bullets, and normalizes identically across the Chatbook user guide, Chatbook developer guide, and server `dev` operator guide.

- [ ] **Step 4: Run targeted contract, Settings, Console, linking, dispatcher, and client checks**

Run:

```bash
set -e -o pipefail
profile_common_dir=$(git rev-parse --path-format=absolute --git-common-dir)
profile_repo_root=$(dirname "$profile_common_dir")
profile_python=
for profile_python_candidate in \
  "$profile_repo_root/.venv/bin/python" \
  "${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}" \
  "$PWD/.venv/bin/python"; do
  if [ -n "$profile_python_candidate" ] && [ -x "$profile_python_candidate" ]; then
    profile_python=$profile_python_candidate
    break
  fi
done
test -n "$profile_python"
"$profile_python" - <<'PY'
import sys

if not ((3, 11) <= sys.version_info[:2] < (4, 0)):
    raise SystemExit(f"Unsupported Python: {sys.version.split()[0]}")
print(f"Using checked project Python {sys.version.split()[0]}")
PY
"$profile_python" -m pytest -q \
  Tests/Packaging/test_profile_core_packaging.py \
  Tests/Sync_Interop/test_personal_context_capabilities.py \
  Tests/Sync_Interop/test_personal_context_adapter.py \
  Tests/UI/test_settings_personal_context.py \
  Tests/Chat/test_console_personal_context_snapshot.py \
  Tests/UI/test_personal_context_link_app_flow.py \
  Tests/Sync_Interop/test_personal_context_first_link.py \
  Tests/Sync_Interop/test_personal_context_first_link_sync.py \
  Tests/Sync_Interop/test_personal_context_dispatcher.py \
  Tests/tldw_api/test_personal_context_sync_client.py
```

Expected: the selected tests pass under a checked shared or active Chatbook project environment using supported Python `>=3.11,<4`. Results from an unsupported or unrelated interpreter are not completion evidence.

- [ ] **Step 5: Run claim, path, and diff guards**

Run:

```bash
set -e -o pipefail
profile_user_guide=Docs/User_Guide/settings/personal-context-profile.md
profile_developer_guide=Docs/Development/personal-context-profile.md
profile_spec=Docs/superpowers/specs/2026-08-31-personal-context-documentation-design.md
profile_plan=Docs/superpowers/plans/2026-09-01-personal-context-documentation-chatbook.md
profile_task='backlog/tasks/task-27019 - Document-Personal-Context-Profile-for-Chatbook-users-and-developers.md'
profile_common_dir=$(git rev-parse --path-format=absolute --git-common-dir)
profile_repo_root=$(dirname "$profile_common_dir")
profile_python=
for profile_python_candidate in \
  "$profile_repo_root/.venv/bin/python" \
  "${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}" \
  "$PWD/.venv/bin/python"; do
  if [ -n "$profile_python_candidate" ] && [ -x "$profile_python_candidate" ]; then
    profile_python=$profile_python_candidate
    break
  fi
done
test -n "$profile_python"
"$profile_python" - <<'PY'
import sys

if not ((3, 11) <= sys.version_info[:2] < (4, 0)):
    raise SystemExit(f"Unsupported Python: {sys.version.split()[0]}")
PY
"$profile_python" - "$profile_user_guide" "$profile_developer_guide" <<'PY'
import re
import sys
from pathlib import Path

user_path, developer_path = map(Path, sys.argv[1:])
user = user_path.read_text(encoding="utf-8")
developer = developer_path.read_text(encoding="utf-8")


def marked(text: str, name: str, document: Path) -> str:
    start = f"<!-- {name}:start -->"
    end = f"<!-- {name}:end -->"
    if text.count(start) != 1 or text.count(end) != 1:
        raise SystemExit(f"{document}: expected exactly one {name} marker pair")
    block = text.split(start, 1)[1].split(end, 1)[0]
    if not block.strip():
        raise SystemExit(f"{document}: empty {name} block")
    return block


def require_each(text: str, values: list[str], document: Path, claim: str) -> None:
    for value in values:
        if value not in text:
            raise SystemExit(f"{document}: missing {claim}: {value}")


expected_shared_bullets = [
    "- `tldw_profile_core` defines the versioned canonical profile object models, exact canonical bytes, interview/tool contracts, serialization, and validation used by both peers. Sync-v2 transport envelopes are a separate contract.",
    "- After successful reviewed first linking, Chatbook and tldw_server converge on the same canonical manifest, scope, record, proposal, and version identities and bytes for the eligible snapshot resulting from the user-approved content-free reconciliation plan.",
    "- Sync V2 defines the `personal_context.manifest`, `personal_context.scope`, `personal_context.record`, `personal_context.proposal`, and content-free `personal_context.purge` domains. Reviewed first linking publishes the eligible snapshot resulting from the user-approved content-free reconciliation plan. Later syncable Chatbook mutations create encrypted local outbox entries, but the current shipped app does not run an ongoing Personal Context sync cycle, so those post-link changes remain queued locally. Purge production and distribution are not wired end to end.",
    "- Each peer retains its own at-rest ciphertext and keys, local database rows, runtime permissions, conflict-review metadata, acknowledgement tracking, and other operational state.",
]


def shared_contract(text: str, document: Path) -> None:
    block = marked(text, "shared-personal-context-contract", document)
    bullets = [line for line in block.splitlines() if line.startswith("- ")]
    if bullets != expected_shared_bullets:
        raise SystemExit(f"{document}: shared-contract bullets are missing or divergent")


matrix_categories = [
    "canonical manifest",
    "global and linked-workspace scopes",
    "record heads and tombstones",
    "eligible proposals and canonical review state",
    "canonical IDs, versions, and bytes",
    "At-rest and recovery keys",
    "Interview draft and transcript objects",
    "Runtime agent authority",
    "Device-only or non-syncable records",
    "local undo, caches, ciphertext, database row IDs",
    "conflict-review objects, acknowledgement tracking",
]
for text, document in ((user, user_path), (developer, developer_path)):
    shared_contract(text, document)
    require_each(
        marked(text, "personal-context-boundary-matrix", document),
        matrix_categories,
        document,
        "boundary-matrix category",
    )

quick_start = marked(user, "personal-context-quick-start", user_path)
quick_steps = [
    int(number)
    for number in re.findall(r"(?m)^(\d+)\.\s+", quick_start)
]
if quick_steps != [1, 2, 3, 4, 5]:
    raise SystemExit(f"{user_path}: quick start must contain exactly steps 1-5")
require_each(
    quick_start,
    [
        "Data & Privacy",
        "My Profile",
        "Manual:",
        "Add",
        "Edit",
        "Save",
        "Interview:",
        "Question style",
        "Save only",
        "Save and use with agents",
        "visibility",
        "syncability",
        "View context",
        "Conversation Inspector",
        "Next Send",
        "home server",
    ],
    user_path,
    "quick-start claim",
)
require_each(
    user,
    [
        "## Common workflows",
        "### Edit manually",
        "### Run or rerun an interview",
        "### Review agent proposals",
        "### Export plaintext and recovery material",
        "### Remove the local copy",
        "### Link a home server",
    ],
    user_path,
    "workflow heading",
)

troubleshooting = marked(user, "personal-context-troubleshooting", user_path)
if "| State | Cause | Safe next action | Current limit |" not in troubleshooting:
    raise SystemExit(f"{user_path}: missing structured troubleshooting header")
troubleshooting_rows: list[tuple[str, list[str]]] = []
for line in troubleshooting.splitlines():
    if not line.startswith("| **"):
        continue
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    if len(cells) != 4:
        raise SystemExit(f"{user_path}: malformed troubleshooting row: {line}")
    troubleshooting_rows.append((cells[0].strip("*"), cells[1:]))
expected_states = [
    "Profile locked",
    "Adaptive interview privacy or provider failure",
    "HTTP or altered TLS verification",
    "Post-link change queued",
    "Capability not negotiated",
    "First-link publication interrupted",
    "Version conflict",
    "First-link semantic collision",
    "Post-link semantic collision",
    "Local removal incomplete or residual state",
    "Purge pending",
]
if [label for label, _fields in troubleshooting_rows] != expected_states:
    raise SystemExit(f"{user_path}: troubleshooting states must be the exact eleven labels")
for label, fields in troubleshooting_rows:
    if any(len(field) < 3 or field in {"---", "TBD", "—"} for field in fields):
        raise SystemExit(f"{user_path}: {label} needs cause, safe action, and current limit")

user_limits = [
    "Chatbook does not currently expose **Delete everywhere**.",
    "Ordinary server REST edits are not published to linked Chatbook",
    "Chatbook has no producer",
    "no ongoing Personal Context cycle",
    "no shipped recovery import or restore action",
    "does not authenticate the server",
]
developer_limits = [
    "Ordinary server REST edits are not currently published to linked Chatbook clients.",
    "The Personal Context purge domain is protocol-only in the current linked flow: Chatbook has no producer, and end-to-end distribution and acknowledgement are not wired.",
    "Post-link conflicts retain generic Sync metadata but have no dedicated Personal Context resolution screen.",
]
require_each(user, user_limits, user_path, "current limitation")
require_each(developer, developer_limits, developer_path, "current limitation")

contradictions = [
    r"Chatbook\s+(?:currently\s+)?(?:exposes|offers|provides|supports)\s+\*{0,2}Delete everywhere",
    r"ordinary server REST edits\s+(?:are\s+)?(?:published|synced|synchronized)\s+to linked Chatbook",
    r"purge[^.\n]{0,100}(?:is|are)\s+(?:fully|end[- ]to[- ]end)\s+(?:wired|distributed|acknowledged)",
    r"post-link conflicts?[^.\n]{0,100}(?:have|offer|use)\s+a dedicated[^.\n]{0,40}(?:screen|resolver|resolution)",
]
for text, document in ((user, user_path), (developer, developer_path)):
    for pattern in contradictions:
        if re.search(pattern, text, re.IGNORECASE):
            raise SystemExit(f"{document}: contradictory shipped claim: {pattern}")

extension = marked(
    developer, "personal-context-extension-checklist", developer_path
)
extension_numbers = [
    int(number) for number in re.findall(r"(?m)^(\d+)\.\s+", extension)
]
if extension_numbers != list(range(1, 11)):
    raise SystemExit(f"{developer_path}: extension checklist must be exactly 1-10")
require_each(
    extension,
    [
        "full local-first Sync peer or a server/API-only client",
        "`tldw_profile_core` first",
        "canonical identities and explicit syncability whenever",
        "API-only clients through authenticated public server APIs, never profile tables",
        "authority, scope, expiry, visibility, and secret-rejection rules at the boundary",
        "plaintext out of logs, diagnostics, outbox metadata, and unencrypted fixtures",
        "parity/conformance coverage for every shared-core or Sync contract",
        "API-only clients need authentication, request/response, error, and privacy coverage",
        "governing ADR",
        "both documentation sets",
    ],
    developer_path,
    "extension-checklist requirement",
)

component_owners = [
    ("tldw_chatbook/Personal_Context/bootstrap.py", "bootstrap_personal_context_service"),
    ("tldw_chatbook/Personal_Context/key_protector.py", "ProfileKeyProtector"),
    ("tldw_chatbook/Personal_Context/repository.py", "PersonalContextRepository"),
    ("tldw_chatbook/Personal_Context/service.py", "PersonalContextService"),
    ("tldw_chatbook/Personal_Context/context_service.py", "ProfileContextService"),
    ("tldw_chatbook/Personal_Context/proposal_service.py", "ProfileProposalService"),
    ("tldw_chatbook/Personal_Context/runtime_policy.py", "AgentAuthority"),
    ("tldw_chatbook/Personal_Context/interview_coordinator.py", "ProfileInterviewCoordinator"),
    ("tldw_chatbook/Personal_Context/interview_draft_repository.py", "InterviewDraftRepository"),
    ("tldw_chatbook/Personal_Context/interview_provider.py", "InterviewQuestionProvider"),
    ("tldw_chatbook/Personal_Context/link_service.py", "PersonalContextLinkService"),
    ("tldw_chatbook/Personal_Context/link_key_custody.py", "PersonalContextLinkKeyCustodian"),
    ("tldw_chatbook/Personal_Context/sync_outbox.py", "ProfileSyncOutbox"),
    ("tldw_chatbook/Sync_Interop/personal_context_adapter.py", "PersonalContextSyncAdapter"),
    ("tldw_chatbook/Sync_Interop/personal_context_dispatcher.py", "PersonalContextOutboxDispatcher"),
    ("tldw_chatbook/Sync_Interop/personal_context_first_link_sync.py", "PersonalContextFirstLinkSync"),
    ("tldw_chatbook/tldw_api/client.py", "bootstrap_sync_v2_personal_context"),
    ("tldw_chatbook/tldw_api/client.py", "complete_sync_v2_personal_context_link"),
    ("tldw_chatbook/Agents/profile_tool_provider.py", "ProfileToolProvider"),
    ("tldw_chatbook/Chat/console_chat_controller.py", "ConsoleChatController"),
    ("tldw_chatbook/Chat/console_agent_bridge.py", "ConsoleAgentBridge"),
    ("tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py", "PersonalContextSettingsPanel"),
    ("tldw_chatbook/Widgets/Settings_Widgets/personal_context_link_modal.py", "PersonalContextLinkModal"),
    ("tldw_chatbook/Widgets/Settings_Widgets/personal_context_review_modal.py", "PersonalContextReviewModal"),
    ("tldw_chatbook/UI/Screens/profile_interview_screen.py", "ProfileInterviewScreen"),
]
developer_lines = developer.splitlines()
for component_path, symbol in component_owners:
    if not any(component_path in line and symbol in line for line in developer_lines):
        raise SystemExit(f"{developer_path}: missing owner pair {component_path} / {symbol}")

require_each(
    developer,
    [
        "### Manual edits and reviewed interviews",
        "### Proposals and direct writes",
        "### Context injection and Next Send",
        "### Transactional outbox and reviewed first link",
        "### Post-link conflicts and purge limits",
        "tldw_chatbook_personal_context_interviews.db",
        "retained draft payloads may include raw answers and turn transcripts",
        "retries only its protected profile-key deletion",
        "workspace corrections and constraints; other keyed workspace records; global corrections and constraints; preferences and working-context records relevant to the current user text; then the remainder",
        "A higher-priority record that does not fit is skipped",
        "### Full local-first Sync peer",
        "### Server/API-only client",
        "It does not need to implement a local canonical repository",
        "Never log profile plaintext, ciphertext, wrapped keys, or raw cryptographic errors.",
        "UI, agent, and transport code must use the owning service/repository boundary and must not access profile tables directly.",
    ],
    developer_path,
    "lifecycle/privacy claim",
)
require_each(
    developer,
    [
        "Tests/Packaging/test_profile_core_packaging.py",
        "Tests/Personal_Context/",
        "Tests/Agents/test_personal_context_prompt.py",
        "Tests/Chat/test_console_personal_context_snapshot.py",
        "Tests/Sync_Interop/test_personal_context_*.py",
        "Tests/UI/test_settings_personal_context.py",
        "Tests/UI/test_personal_context_*.py",
        "Tests/tldw_api/test_personal_context_sync_client.py",
    ],
    developer_path,
    "targeted test-map entry",
)
print("Per-document semantic claims passed independently.")
PY

rg -Fqx -- '- **Status:** Approved design; shipped-behavior correction reviewed and merged' \
  "$profile_spec"
if rg -Fq 'codex/personal-context-docs-sync-truth' "$profile_developer_guide"; then
  echo 'Developer guide contains a temporary branch name'
  exit 1
fi

# No repository-wide docs-link checker governs these pages. Mirror the local-link
# existence contract used by Tests/Docs/test_console_library_controls_docs.py with
# exact target and source checks, and verify counterpart links through GitHub above.
test -f Docs/User_Guide/settings/personal-context-profile.md
test -f Docs/Development/personal-context-profile.md
test -f Docs/Development/Sync-v2-client.md
test -f Docs/superpowers/specs/2026-08-28-unified-personal-context-profile-design.md
test -f backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md
rg -Fq '(Sync-v2-client.md)' "$profile_developer_guide"
rg -Fq '(../superpowers/specs/2026-08-28-unified-personal-context-profile-design.md)' \
  "$profile_developer_guide"
rg -Fq '(../../backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md)' \
  "$profile_developer_guide"
rg -Fq 'https://github.com/rmusser01/tldw_server/blob/dev/Docs/User_Guides/Server/Personal_Context_Profile.md' \
  "$profile_user_guide"
rg -Fq 'https://github.com/rmusser01/tldw_server/blob/dev/Docs/API-related/Personal_Context_API.md' \
  "$profile_user_guide"
rg -Fq 'https://github.com/rmusser01/tldw_server/blob/dev/Docs/Code_Documentation/Personal_Context_Developer_Guide.md' \
  "$profile_developer_guide"
rg -Fq 'https://github.com/rmusser01/tldw_server/blob/dev/Docs/API-related/Personal_Context_API.md' \
  "$profile_developer_guide"
rg -Fq '| [Set up and manage your Personal Context Profile](settings/personal-context-profile.md) | Optional interviews, global/workspace context, agent proposals, synchronization boundaries, export, and removal. |' \
  Docs/User_Guide/index.md
rg -Fq 'For Personal Context internals and extension work, see [Personal Context Profile](personal-context-profile.md).' \
  Docs/Development/Developer_Guide.md
rg -Fq '| **Applies immediately** | Each action takes effect at once; no draft to save or revert. | Workspaces, [My Profile](settings/personal-context-profile.md) |' \
  Docs/User_Guide/settings.md
rg -Fq '| Data & Privacy | **My Profile** → [own page](settings/personal-context-profile.md) | Personal and workspace context, interviews, agent proposals, authority, export, and removal. | Applies immediately |' \
  Docs/User_Guide/settings.md

profile_changed_paths=$(
  {
    git diff --name-only origin/dev...HEAD
    git diff --name-only
    git diff --cached --name-only
    git ls-files --others --exclude-standard
  } | sed '/^$/d' | sort -u
)
profile_unexpected_paths=$(
  printf '%s\n' "$profile_changed_paths" | awk '
    $0 == "Docs/Development/Developer_Guide.md" { next }
    $0 == "Docs/Development/personal-context-profile.md" { next }
    $0 == "Docs/User_Guide/index.md" { next }
    $0 == "Docs/User_Guide/settings/personal-context-profile.md" { next }
    $0 == "Docs/superpowers/plans/2026-09-01-personal-context-documentation-chatbook.md" { next }
    $0 == "Docs/superpowers/specs/2026-08-31-personal-context-documentation-design.md" { next }
    $0 == "backlog/tasks/task-27019 - Document-Personal-Context-Profile-for-Chatbook-users-and-developers.md" { next }
    NF { print }
  '
)
if [ -n "$profile_unexpected_paths" ]; then
  printf 'Unexpected changed paths:\n%s\n' "$profile_unexpected_paths"
  exit 1
fi
printf 'Allowed changed paths:\n%s\n' "$profile_changed_paths"
git diff --check origin/dev...HEAD
git diff --check --cached
git status --short
git diff --stat origin/dev...HEAD
git diff --stat --cached
```

Expected: each guide independently proves its required shared-contract and current-limit claims; all seven user failure-state labels are explicit; every new discovery link and every internal/server target is checked independently; and the allowed-path assertion accepts only the two guides, two discovery indexes, plan, and TASK-27019 across committed, staged, unstaged, and untracked paths.

- [ ] **Step 6: Commit the completed Task 5 execution record**

After Steps 1-5 have run successfully, mark every Task 5 step through this commit step `[x]`. Before Task 6 begins, fail if any Task 1-5 checkbox remains open, then commit the plan so the final rebase and all verification evidence are recorded in a clean worktree.

Run:

```bash
set -e -o pipefail
profile_plan=Docs/superpowers/plans/2026-09-01-personal-context-documentation-chatbook.md
backlog task 27019 --plain | rg -q 'Status:.*In Progress'
if profile_pending_steps=$(sed -n '/^### Task 1:/,/^### Task 6:/p' "$profile_plan" | rg -n '^- \[ \]'); then
  printf 'Unexecuted Task 1-5 plan steps:\n%s\n' "$profile_pending_steps"
  exit 1
else
  profile_pending_status=$?
  test "$profile_pending_status" -eq 1 || exit "$profile_pending_status"
fi
git add "$profile_plan"
git diff --check --cached
git commit -m "docs: record Chatbook Personal Context verification"
git diff --check origin/dev...HEAD
test -z "$(git status --short)"
backlog task 27019 --plain | rg -q 'Status:.*In Progress'
```

Expected: all guide/index content and Tasks 1-5 execution checkboxes are committed, TASK-27019 remains **In Progress**, and Task 6 starts with no uncommitted plan changes.

### Task 6: Open, review, and close the Chatbook documentation PR

**Files:**

- Modify: `Docs/superpowers/plans/2026-09-01-personal-context-documentation-chatbook.md`
- Modify: `backlog/tasks/task-27019 - Document-Personal-Context-Profile-for-Chatbook-users-and-developers.md`

- [ ] **Step 1: Push and open the PR while TASK-27019 is In Progress**

Prepare `/tmp/personal-context-chatbook-pr.md` with the documentation summary, current limitations, exact Task 5 evidence, and ADR result. Do not check this plan step yet; Task 6 execution checkboxes are recorded together in Step 3 after review completes.

```bash
set -e -o pipefail
backlog task 27019 --plain | rg -q 'Status:.*In Progress'
test -z "$(git status --short)"
git push -u origin codex/personal-context-docs
profile_pr_url=$(gh pr create \
  --base dev \
  --head codex/personal-context-docs \
  --title "docs: add Personal Context user and developer guides" \
  --body-file /tmp/personal-context-chatbook-pr.md)
test -n "$profile_pr_url"
test "$(gh pr view "$profile_pr_url" --json baseRefName --jq .baseRefName)" = 'dev'
test "$(gh pr view "$profile_pr_url" --json headRefName --jq .headRefName)" = \
  'codex/personal-context-docs'
backlog task 27019 --plain | rg -q 'Status:.*In Progress'
```

Expected: the PR is open against `dev` from `codex/personal-context-docs`, and TASK-27019 is still **In Progress**.

- [ ] **Step 2: Wait for initial checks/review and address valid feedback while the task is open**

Inspect the initial review comments and required checks. Address valid feedback only while TASK-27019 is **In Progress**. If feedback requires a rebase or repository edit, make that change while the task is open, rerun Task 5 Steps 1-5, commit it, push it, and restart this step. Do not edit Task 6 checkboxes during that loop.

```bash
set -e -o pipefail
backlog task 27019 --plain | rg -q 'Status:.*In Progress'
test -z "$(git status --short)"
profile_pr_number=$(gh pr view --json number --jq .number)
test -n "$profile_pr_number"
gh pr view "$profile_pr_number" --comments
gh pr checks "$profile_pr_number" --required --watch --fail-fast
profile_review_decision=$(gh pr view "$profile_pr_number" --json reviewDecision --jq .reviewDecision)
case "$profile_review_decision" in
  ''|APPROVED) ;;
  REVIEW_REQUIRED|CHANGES_REQUESTED)
    echo "PR review is not complete: $profile_review_decision"
    exit 1
    ;;
  *)
    echo "Unknown PR review state: $profile_review_decision"
    exit 1
    ;;
esac
test "$(gh pr view "$profile_pr_number" --json baseRefName --jq .baseRefName)" = 'dev'
test "$(gh pr view "$profile_pr_number" --json headRefName --jq .headRefName)" = \
  'codex/personal-context-docs'
test "$(gh pr view "$profile_pr_number" --json headRefOid --jq .headRefOid)" = \
  "$(git rev-parse HEAD)"
test "$(gh pr view "$profile_pr_number" --json isDraft --jq .isDraft)" = 'false'
test "$(gh pr view "$profile_pr_number" --json mergeable --jq .mergeable)" = 'MERGEABLE'
profile_pr_paths=$(gh pr view "$profile_pr_number" --json files --jq '.files[].path' | sort -u)
profile_unexpected_pr_paths=$(
  printf '%s\n' "$profile_pr_paths" | awk '
    $0 == "Docs/Development/Developer_Guide.md" { next }
    $0 == "Docs/Development/personal-context-profile.md" { next }
    $0 == "Docs/User_Guide/index.md" { next }
    $0 == "Docs/User_Guide/settings/personal-context-profile.md" { next }
    $0 == "Docs/superpowers/plans/2026-09-01-personal-context-documentation-chatbook.md" { next }
    $0 == "Docs/superpowers/specs/2026-08-31-personal-context-documentation-design.md" { next }
    $0 == "backlog/tasks/task-27019 - Document-Personal-Context-Profile-for-Chatbook-users-and-developers.md" { next }
    NF { print }
  '
)
if [ -n "$profile_unexpected_pr_paths" ]; then
  printf 'Unexpected PR paths:\n%s\n' "$profile_unexpected_pr_paths"
  exit 1
fi
backlog task 27019 --plain | rg -q 'Status:.*In Progress'
```

Expected: initial required checks pass, required review is approved or the repository has no required review decision, the PR is mergeable with the exact base/head and docs-only scope, and TASK-27019 remains **In Progress**. Repeat this step after every feedback commit.

- [ ] **Step 3: Close TASK-27019 only after the PR is clean, then push final metadata**

Rerun the exact Step 2 verification block immediately before closing. Then mark Task 6 Steps 1, 2, and 3 `[x]` together in this plan and run the following, replacing bracketed evidence with the exact Task 5/PR results. This plan-and-task commit is the final repository mutation when its post-push checks pass.

```bash
set -e -o pipefail
profile_plan=Docs/superpowers/plans/2026-09-01-personal-context-documentation-chatbook.md
profile_task='backlog/tasks/task-27019 - Document-Personal-Context-Profile-for-Chatbook-users-and-developers.md'
profile_pr_url=$(gh pr view --json url --jq .url)
profile_pr_number=$(gh pr view --json number --jq .number)
test -n "$profile_pr_url"
test -n "$profile_pr_number"
profile_expected_base_oid=$(gh pr view "$profile_pr_number" --json baseRefOid --jq .baseRefOid)
profile_expected_review_decision=$(gh pr view "$profile_pr_number" --json reviewDecision --jq .reviewDecision)
test "$profile_expected_base_oid" = "$(git rev-parse origin/dev)"
test "$profile_expected_base_oid" = "$(git merge-base origin/dev HEAD)"
test "$(gh pr view "$profile_pr_number" --json mergeable --jq .mergeable)" = 'MERGEABLE'
test "$(gh pr view "$profile_pr_number" --json mergeStateStatus --jq .mergeStateStatus)" = 'CLEAN'
case "$profile_expected_review_decision" in
  ''|APPROVED) ;;
  *)
    echo "PR review is not clean: $profile_expected_review_decision"
    exit 1
    ;;
esac
backlog task 27019 --plain | rg -q 'Status:.*In Progress'
if profile_pending_task6=$(sed -n '/^### Task 6:/,$p' "$profile_plan" | rg -n '^- \[ \]'); then
  printf 'Unexecuted Task 6 plan steps:\n%s\n' "$profile_pending_task6"
  exit 1
else
  profile_pending_status=$?
  test "$profile_pending_status" -eq 1 || exit "$profile_pending_status"
fi
backlog task edit 27019 \
  --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 \
  --ref Docs/superpowers/specs/2026-08-31-personal-context-documentation-design.md \
  --ref backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md \
  --ref "$profile_pr_url" \
  --notes "Implemented the Chatbook Personal Context user and developer guides, discovery links, exact shared-contract parity block, current sync/non-sync matrix, seven structured failure states, and ten-item extension checklist. Verification: [exact Task 5 results]. PR checks/review/base/head/scope: [exact Step 2 results]. ADR required: no new ADR required; existing ADR applies. ADR path: backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md. Reason: documentation only; the existing Personal Context authority, Sync, and encryption ADR applies. Lessons learned: [record a genuine lesson with its incident, or state none]." \
  -s Done
backlog task 27019 --plain | rg -q 'Status:.*Done'
if rg -n '^- \[ \]' "$profile_task"; then
  echo 'TASK-27019 still has unchecked acceptance criteria'
  exit 1
fi
rg -Fq "$profile_pr_url" "$profile_task"
git add "$profile_plan" "$profile_task"
git diff --check --cached
test "$(git diff --cached --name-only | sort)" = \
  "$(printf '%s\n' "$profile_plan" "$profile_task" | sort)"
git commit -m "docs: close Chatbook Personal Context documentation task"
git push
profile_local_final_head=$(git rev-parse HEAD)
profile_published_final_head=
for _profile_wait in $(seq 1 30); do
  profile_published_final_head=$(gh pr view "$profile_pr_number" --json headRefOid --jq .headRefOid)
  if [ "$profile_published_final_head" = "$profile_local_final_head" ]; then
    break
  fi
  sleep 2
done
test "$profile_published_final_head" = "$profile_local_final_head"
gh pr checks "$profile_pr_number" --required --watch --fail-fast
test "$(gh pr view "$profile_pr_number" --json baseRefName --jq .baseRefName)" = 'dev'
test "$(gh pr view "$profile_pr_number" --json headRefName --jq .headRefName)" = \
  'codex/personal-context-docs'
profile_final_head_oid=$(gh pr view "$profile_pr_number" --json headRefOid --jq .headRefOid)
profile_final_base_oid=$(gh pr view "$profile_pr_number" --json baseRefOid --jq .baseRefOid)
profile_final_mergeable=$(gh pr view "$profile_pr_number" --json mergeable --jq .mergeable)
profile_final_merge_state=$(gh pr view "$profile_pr_number" --json mergeStateStatus --jq .mergeStateStatus)
profile_final_review_decision=$(gh pr view "$profile_pr_number" --json reviewDecision --jq .reviewDecision)
test "$profile_final_head_oid" = "$profile_local_final_head"
test "$profile_final_base_oid" = "$profile_expected_base_oid"
test "$profile_final_mergeable" = 'MERGEABLE'
test "$profile_final_merge_state" = 'CLEAN'
case "$profile_expected_review_decision:$profile_final_review_decision" in
  APPROVED:APPROVED|:APPROVED|:) ;;
  *)
    echo "PR review was dismissed or is no longer clean: $profile_final_review_decision"
    exit 1
    ;;
esac
profile_final_pr_paths=$(gh pr view "$profile_pr_number" --json files --jq '.files[].path' | sort -u)
profile_expected_pr_paths=$(printf '%s\n' \
  Docs/Development/Developer_Guide.md \
  Docs/Development/personal-context-profile.md \
  Docs/User_Guide/index.md \
  Docs/User_Guide/settings/personal-context-profile.md \
  Docs/superpowers/plans/2026-09-01-personal-context-documentation-chatbook.md \
  'backlog/tasks/task-27019 - Document-Personal-Context-Profile-for-Chatbook-users-and-developers.md' | \
  sort
)
if [ "$profile_final_pr_paths" != "$profile_expected_pr_paths" ]; then
  printf 'Expected final PR paths:\n%s\nActual final PR paths:\n%s\n' \
    "$profile_expected_pr_paths" "$profile_final_pr_paths"
  exit 1
fi
profile_repo_slug=$(gh repo view --json nameWithOwner --jq .nameWithOwner)
profile_thread_state=$(gh api graphql \
  -F owner="${profile_repo_slug%%/*}" \
  -F name="${profile_repo_slug#*/}" \
  -F number="$profile_pr_number" \
  -f query='query($owner: String!, $name: String!, $number: Int!) {
    repository(owner: $owner, name: $name) {
      pullRequest(number: $number) {
        reviewThreads(first: 100) {
          nodes { isResolved }
          pageInfo { hasNextPage }
        }
      }
    }
  }' \
  --jq '[.data.repository.pullRequest.reviewThreads.pageInfo.hasNextPage, ([.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved == false)] | length)] | @tsv')
IFS=$'\t' read -r profile_more_threads profile_unresolved_threads <<< "$profile_thread_state"
test "$profile_more_threads" = 'false'
test "$profile_unresolved_threads" = '0'
gh pr view "$profile_pr_number" --comments
gh api "repos/$profile_repo_slug/pulls/$profile_pr_number/reviews" --paginate \
  --jq '.[] | select(.body != "") | [.state, .commit_id, .user.login, .html_url, .body] | @tsv'
gh api "repos/$profile_repo_slug/issues/$profile_pr_number/comments" --paginate \
  --jq '.[] | [.user.login, .html_url, .body] | @tsv'
printf 'After reviewing final-head comments/reviews, type exactly "no actionable feedback": ' >&2
IFS= read -r profile_feedback_confirmation
test "$profile_feedback_confirmation" = 'no actionable feedback'
backlog task 27019 --plain | rg -q 'Status:.*Done'
test -z "$(git status --short)"
```

Expected: all ACs and executed plan steps are checked, implementation notes and the PR reference are recorded, and the final metadata head is pushed and green. The PR must still point at that exact head and the unchanged rebased `dev` OID, remain mergeable and `CLEAN`, retain its clean review state, contain only the exact documentation allowlist, have no unresolved review threads, and have no actionable final-head feedback. TASK-27019 is **Done** only while all of those assertions remain true.

If a final check fails because the base advanced, the PR is behind/conflicting, review was dismissed or requests changes, scope changed, checks are pending/failing, a review thread is unresolved, actionable feedback remains, or any repository edit is required, TASK-27019 cannot silently remain Done. Do not edit docs/code while the task is Done. The status change below must be the first mutation; commit and push it, then return to Step 2 and repeat the review/close loop:

```bash
set -e -o pipefail
backlog task edit 27019 -s "In Progress"
git add "backlog/tasks/task-27019 - Document-Personal-Context-Profile-for-Chatbook-users-and-developers.md"
git diff --check --cached
git commit -m "docs: reopen Chatbook Personal Context documentation task"
git push
```
