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

### Task 1: Rebase and establish the shipped-behavior claim inventory

**Files:**

- Inspect: all paths in the file map
- Modify: `backlog/tasks/task-27019 - Document-Personal-Context-Profile-for-Chatbook-users-and-developers.md`

- [x] **Step 1: Rebase the isolated branch on current `dev`**

Run:

```bash
git fetch origin dev
git rebase origin/dev
```

Expected: the branch rebases cleanly without unrelated working-tree changes.

- [x] **Step 2: Verify Backlog ownership and read the applicable workflow lessons**

Run:

```bash
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
rg -Fq 'Remove local profile' tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py
rg -Fq 'Run interview again' tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py
rg -Fq 'Get to know you' tldw_chatbook/UI/Screens/profile_interview_screen.py
rg -Fq 'Define project context after creating' tldw_chatbook/Widgets/workspace_create_modal.py
if rg -n -i -e 'action_delete_everywhere' -e 'delete_everywhere' \
  -e 'id="[^"]*delete[^"]*everywhere' tldw_chatbook --glob '*.py'; then
  echo 'Unexpected Delete Everywhere action/control'
  exit 1
fi
for profile_component in \
  tldw_chatbook/Personal_Context/bootstrap.py \
  tldw_chatbook/Personal_Context/key_protector.py \
  tldw_chatbook/Personal_Context/repository.py \
  tldw_chatbook/Personal_Context/service.py \
  tldw_chatbook/Personal_Context/context_service.py \
  tldw_chatbook/Personal_Context/proposal_service.py \
  tldw_chatbook/Personal_Context/runtime_policy.py \
  tldw_chatbook/Personal_Context/interview_coordinator.py \
  tldw_chatbook/Personal_Context/interview_draft_repository.py \
  tldw_chatbook/Personal_Context/interview_provider.py \
  tldw_chatbook/Personal_Context/link_service.py \
  tldw_chatbook/Personal_Context/link_key_custody.py \
  tldw_chatbook/Personal_Context/sync_outbox.py \
  tldw_chatbook/Sync_Interop/personal_context_adapter.py \
  tldw_chatbook/Sync_Interop/personal_context_dispatcher.py \
  tldw_chatbook/Sync_Interop/personal_context_first_link_sync.py \
  tldw_chatbook/Agents/profile_tool_provider.py \
  tldw_chatbook/Chat/console_chat_controller.py \
  tldw_chatbook/Chat/console_agent_bridge.py \
  tldw_chatbook/tldw_api/client.py \
  tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py \
  tldw_chatbook/Widgets/Settings_Widgets/personal_context_link_modal.py \
  tldw_chatbook/Widgets/Settings_Widgets/personal_context_review_modal.py \
  tldw_chatbook/UI/Screens/profile_interview_screen.py; do
  test -f "$profile_component"
done
rg -n '^(class (PersonalContextService|PersonalContextRepository|ProfileContextService|ProfileProposalService|PersonalContextLinkService|ProfileSyncOutbox|PersonalContextSyncAdapter|PersonalContextOutboxDispatcher|PersonalContextFirstLinkSync|ProfileToolProvider|AgentAuthority|ProfileInterviewCoordinator|InterviewDraftRepository|ConsoleChatController|ConsoleAgentBridge)|def bootstrap_personal_context_service)' \
  tldw_chatbook/Personal_Context tldw_chatbook/Sync_Interop \
  tldw_chatbook/Agents/profile_tool_provider.py \
  tldw_chatbook/Chat/console_chat_controller.py \
  tldw_chatbook/Chat/console_agent_bridge.py
```

Expected: removal, interview, and service surfaces exist; no shipped Chatbook **Delete everywhere** control is found.

- [x] **Step 4: Confirm Sync-v2 domains and current gaps**

Run:

```bash
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
if rg -n -U -P '_insert_outbox\(\s*\n\s*connection,\s*\n\s*object_type="purge"' \
  tldw_chatbook/Personal_Context/repository.py; then
  echo 'Unexpected Chatbook Personal Context purge producer'
  exit 1
fi
if rg -n -i -e 'personal.context.*post.?link.*resolve' \
  -e 'post.?link.*personal.context.*resolve' \
  tldw_chatbook/Widgets/Settings_Widgets tldw_chatbook/UI/Screens/profile_interview_screen.py; then
  echo 'Unexpected dedicated Personal Context post-link resolver'
  exit 1
fi
```

Expected: five protocol domains and reviewed first-link behavior exist; no dedicated post-link resolver or reachable Chatbook purge producer is found.

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
git add \
  Docs/superpowers/plans/2026-09-01-personal-context-documentation-chatbook.md \
  "backlog/tasks/task-27019 - Document-Personal-Context-Profile-for-Chatbook-users-and-developers.md"
git commit -m "docs: plan Chatbook Personal Context guides"
```

### Task 2: Make the user guide task-oriented and release-accurate

**Files:**

- Modify: `Docs/User_Guide/settings/personal-context-profile.md`

- [ ] **Step 1: Add `In five minutes` after `Getting there`**

The numbered flow must cover:

1. Open **F9 > Data & Privacy > My Profile**.
2. Choose manual entry or optional **Get to know you**; skipping is supported and stores no answers.
3. Review every proposed value and its visibility/syncability controls.
4. Save, optionally enable agent use, and inspect **Context > Next Send**.
5. Link a supported home server only when sharing is desired.

- [ ] **Step 2: Add `Common workflows`**

Cover global preferences, workspace goals/conventions, agent proposal review, rerunning global/workspace interviews, reviewed first linking, plaintext/recovery export, and local-copy removal. Use current control names. State that new inferred facts remain proposals; direct write only updates an existing eligible record for an explicit correction evidenced by the current persisted user message.

- [ ] **Step 3: Add `What currently synchronizes`**

Include this deliberately identical shared-contract block, with the markers retained so cross-repository parity can be checked automatically:

```markdown
<!-- shared-personal-context-contract:start -->
- `tldw_profile_core` defines the versioned canonical profile object models, exact canonical bytes, interview/tool contracts, serialization, and validation used by both peers. Sync-v2 transport envelopes are a separate contract.
- After a successful reviewed link, Chatbook and tldw_server converge on the same canonical manifest, scope, record, proposal, and version identities and bytes for eligible shared objects.
- Sync V2 defines the `personal_context.manifest`, `personal_context.scope`, `personal_context.record`, `personal_context.proposal`, and content-free `personal_context.purge` domains. The current linked flow publishes eligible Chatbook-originated manifest, scope, record, and proposal changes; purge production and distribution are not wired end to end.
- Each peer retains its own at-rest ciphertext and keys, local database rows, runtime permissions, conflict-review metadata, acknowledgement tracking, and other operational state.
<!-- shared-personal-context-contract:end -->
```

Follow it with this full matrix:

| Shared through the current linked flow when eligible | Remains peer-local or is not currently published |
| --- | --- |
| Canonical manifest after successful reviewed linking | Peer-local at-rest encryption and recovery keys |
| Required global and linked-workspace scope objects | Raw interview answers and unfinished drafts |
| Records and tombstones whose controls permit synchronization | Runtime agent authority grants and tool availability |
| Eligible proposals and their canonical review state | Device-only records or records marked non-syncable |
| Exact canonical object identities, versions, and bytes for eligible shared objects | Local undo history, caches, ciphertext, database row identities, and other operational metadata |
| — | Conflict-review objects and acknowledgement tracking |

Required notes:

- Ordinary server REST edits are not currently published to linked Chatbook clients.
- `personal_context.purge` exists at the protocol boundary, but Chatbook has no producer and the server endpoint does not distribute it through Sync V2.

- [ ] **Step 4: Correct deletion wording**

Keep **Remove local profile** as the available Chatbook action. State plainly:

> Chatbook does not currently expose **Delete everywhere**. The authenticated server purge endpoint creates a server-local purge fence and remains `purge_pending`; distribution and acknowledgement completion are not wired end to end.

Do not tell users that reconnecting devices clears `purge_pending`.

- [ ] **Step 5: Add troubleshooting**

Use these exact seven failure-state labels and give a cause, safe next action, and current product limit for each:

1. **Profile locked**
2. **Offline or queued**
3. **Capability not negotiated**
4. **Version conflict**
5. **First-link semantic collision**
6. **Post-link semantic collision**
7. **Purge pending**

Also explain why an ordinary server REST edit may not appear in Chatbook. State that post-link conflicts retain generic Sync metadata but have no dedicated Personal Context resolution screen.

- [ ] **Step 6: Add stable server links**

Link to:

- `https://github.com/rmusser01/tldw_server/blob/dev/Docs/User_Guides/Server/Personal_Context_Profile.md`
- `https://github.com/rmusser01/tldw_server/blob/dev/Docs/API-related/Personal_Context_API.md`

- [ ] **Step 7: Run claim and diff guards**

Run:

```bash
rg -n "does not currently expose|not currently published|not wired end to end|no dedicated Personal Context" \
  Docs/User_Guide/settings/personal-context-profile.md
git diff --check -- Docs/User_Guide/settings/personal-context-profile.md
git diff -- Docs/User_Guide/settings/personal-context-profile.md
```

Expected: current limitations are explicit and the existing reference is refined rather than duplicated.

- [ ] **Step 8: Commit the user guide**

Run:

```bash
git add Docs/User_Guide/settings/personal-context-profile.md
git commit -m "docs: clarify Chatbook Personal Context workflows"
```

### Task 3: Add the focused developer guide

**Files:**

- Create: `Docs/Development/personal-context-profile.md`

- [ ] **Step 1: Write contract and ownership sections**

Cover Shared Core `0.1.0`, separate Sync-v2 envelopes, post-link identity convergence, peer-local at-rest keys/ciphertext, and wrapped server-owned Sync integrity-key bootstrap. Link relatively to:

- `../superpowers/specs/2026-08-28-unified-personal-context-profile-design.md`
- `../../backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md`
- `Sync-v2-client.md`

- [ ] **Step 2: Add the component map**

Document these exact owners, using repository-root paths:

- `tldw_chatbook/Personal_Context/bootstrap.py` — `bootstrap_personal_context_service`
- `tldw_chatbook/Personal_Context/key_protector.py` — local at-rest key protection and recovery boundary
- `tldw_chatbook/Personal_Context/repository.py` — `PersonalContextRepository`
- `tldw_chatbook/Personal_Context/service.py` — `PersonalContextService`
- `tldw_chatbook/Personal_Context/context_service.py` — `ProfileContextService`
- `tldw_chatbook/Personal_Context/proposal_service.py` — `ProfileProposalService`
- `tldw_chatbook/Personal_Context/runtime_policy.py` — `AgentAuthority`
- `tldw_chatbook/Personal_Context/interview_coordinator.py` — reviewed interview execution
- `tldw_chatbook/Personal_Context/interview_draft_repository.py` — unfinished interview-draft storage
- `tldw_chatbook/Personal_Context/interview_provider.py` — interview model-provider boundary
- `tldw_chatbook/Personal_Context/link_service.py` — `PersonalContextLinkService`
- `tldw_chatbook/Personal_Context/link_key_custody.py` — wrapping/integrity-key custody
- `tldw_chatbook/Personal_Context/sync_outbox.py` — encrypted `ProfileSyncOutbox` lifecycle boundary
- `tldw_chatbook/Sync_Interop/personal_context_adapter.py` — `PersonalContextSyncAdapter`
- `tldw_chatbook/Sync_Interop/personal_context_dispatcher.py` — `PersonalContextOutboxDispatcher`
- `tldw_chatbook/Sync_Interop/personal_context_first_link_sync.py` — `PersonalContextFirstLinkSync`
- `tldw_chatbook/tldw_api/client.py` — Personal Context bootstrap and reviewed-link completion client methods
- `tldw_chatbook/Agents/profile_tool_provider.py` — `ProfileToolProvider`
- `tldw_chatbook/Chat/console_chat_controller.py` — Console snapshot/context injection
- `tldw_chatbook/Chat/console_agent_bridge.py` — Console agent-tool bridge
- `tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py` — Settings presentation and user actions only
- `tldw_chatbook/Widgets/Settings_Widgets/personal_context_link_modal.py` — reviewed linking presentation only
- `tldw_chatbook/Widgets/Settings_Widgets/personal_context_review_modal.py` — proposal/review presentation only
- `tldw_chatbook/UI/Screens/profile_interview_screen.py` — interview presentation only

State that UI, agents, and transport use the service/repository boundary; they do not write profile tables directly.

- [ ] **Step 3: Document read/write lifecycles and current gaps**

Cover manual edits, reviewed interview output, proposal/direct-write distinction, immutable encrypted versions, controls/expiry/tombstones/receipts, context selection and **Next Send**, transactional Chatbook outbox, reviewed first linking, generic post-link conflict metadata, and protocol-only purge without end-to-end production/distribution/acknowledgement.

Include these exact current-limit sentences so final verification can fail closed per document:

- `Ordinary server REST edits are not currently published to linked Chatbook clients.`
- `The Personal Context purge domain is protocol-only in the current linked flow: Chatbook has no producer, and end-to-end distribution and acknowledgement are not wired.`
- `Post-link conflicts retain generic Sync metadata but have no dedicated Personal Context resolution screen.`

Repeat the full boundary matrix in developer terms so every shared and peer-local category is explicit:

| Shared through the current linked flow when eligible | Remains peer-local or is not currently published |
| --- | --- |
| Canonical manifest after successful reviewed linking | Peer-local at-rest encryption and recovery keys |
| Required global and linked-workspace scope objects | Raw interview answers and unfinished drafts |
| Records and tombstones whose controls permit synchronization | Runtime agent authority grants and tool availability |
| Eligible proposals and their canonical review state | Device-only records or records marked non-syncable |
| Exact canonical object identities, versions, and bytes for eligible shared objects | Local undo history, caches, ciphertext, database row identities, and other operational metadata |
| — | Conflict-review objects and acknowledgement tracking |

- [ ] **Step 4: Add the complete extension checklist and test map**

Include all ten checklist items:

1. Decide whether the change affects the shared contract or only one peer.
2. Make shared canonical object changes in `tldw_profile_core` first; change Sync transport separately.
3. Preserve canonical identities and explicit syncability.
4. Route through the owning service; never access profile tables directly.
5. Enforce authority, scope, expiry, visibility, and secret-rejection rules.
6. Keep plaintext out of logs, diagnostics, outbox metadata, and unencrypted fixtures.
7. Add parity/conformance coverage in both repositories.
8. Add peer-specific migration, repository, service, API/UI, and recovery tests.
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

- [ ] **Step 5: Validate referenced paths and Markdown**

Run:

```bash
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

- [ ] **Step 6: Commit the developer guide**

Run:

```bash
git add Docs/Development/personal-context-profile.md
git commit -m "docs: add Chatbook Personal Context developer guide"
```

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

Near the top of `Docs/Development/Developer_Guide.md`, link `personal-context-profile.md` as the canonical guide for Personal Context internals and extension work. Do not add a second architecture summary.

- [ ] **Step 3: Confirm Settings already links the page**

Run:

```bash
rg -n "My Profile.*personal-context-profile\.md" Docs/User_Guide/settings.md
```

Expected: existing links are present; leave `settings.md` unchanged.

- [ ] **Step 4: Validate and commit discovery links**

Run:

```bash
rg -n "personal-context-profile\.md" Docs/User_Guide/index.md Docs/User_Guide/settings.md Docs/Development/Developer_Guide.md
git diff --check -- Docs/User_Guide/index.md Docs/Development/Developer_Guide.md
git add Docs/User_Guide/index.md Docs/Development/Developer_Guide.md
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

# Re-inventory the exact merged product claims after the final rebase.
rg -Fq 'Remove local profile' tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py
rg -Fq 'Run interview again' tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py
rg -Fq 'Get to know you' tldw_chatbook/UI/Screens/profile_interview_screen.py
rg -Fq 'Define project context after creating' tldw_chatbook/Widgets/workspace_create_modal.py
if rg -n -i -e 'action_delete_everywhere' -e 'delete_everywhere' \
  -e 'id="[^"]*delete[^"]*everywhere' tldw_chatbook --glob '*.py'; then
  echo 'Unexpected Delete Everywhere action/control'
  exit 1
fi
for profile_component in \
  tldw_chatbook/Personal_Context/repository.py \
  tldw_chatbook/Personal_Context/service.py \
  tldw_chatbook/Personal_Context/context_service.py \
  tldw_chatbook/Personal_Context/proposal_service.py \
  tldw_chatbook/Personal_Context/key_protector.py \
  tldw_chatbook/Personal_Context/interview_coordinator.py \
  tldw_chatbook/Personal_Context/link_service.py \
  tldw_chatbook/Personal_Context/link_key_custody.py \
  tldw_chatbook/Personal_Context/sync_outbox.py \
  tldw_chatbook/Sync_Interop/personal_context_adapter.py \
  tldw_chatbook/Sync_Interop/personal_context_dispatcher.py \
  tldw_chatbook/Sync_Interop/personal_context_first_link_sync.py \
  tldw_chatbook/Agents/profile_tool_provider.py \
  tldw_chatbook/Chat/console_chat_controller.py \
  tldw_chatbook/Chat/console_agent_bridge.py \
  tldw_chatbook/tldw_api/client.py; do
  test -f "$profile_component"
done
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
if rg -n -U -P '_insert_outbox\(\s*\n\s*connection,\s*\n\s*object_type="purge"' \
  tldw_chatbook/Personal_Context/repository.py; then
  echo 'Unexpected Chatbook Personal Context purge producer'
  exit 1
fi
if rg -n -i -e 'personal.context.*post.?link.*resolve' \
  -e 'post.?link.*personal.context.*resolve' \
  tldw_chatbook/Widgets/Settings_Widgets tldw_chatbook/UI/Screens/profile_interview_screen.py; then
  echo 'Unexpected dedicated Personal Context post-link resolver'
  exit 1
fi
```

Expected: the branch is based on current `origin/dev`; TASK-27019 resolves uniquely; the current controls, components, five domains, reviewed first-link, outbox/dispatcher/client boundaries, and negative purge/resolver claims still match the guides. Any scanner failure or newly shipped seam stops execution for re-inventory. There must be no later rebase after the task is marked Done.

- [ ] **Step 2: Verify server docs have landed on `dev`**

Run:

```bash
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

Run in `zsh`:

```bash
diff -u \
  <(sed -n '/<!-- shared-personal-context-contract:start -->/,/<!-- shared-personal-context-contract:end -->/p' \
    Docs/User_Guide/settings/personal-context-profile.md | tr -s '[:space:]' ' ') \
  <(gh api -X GET repos/rmusser01/tldw_server/contents/Docs/User_Guides/Server/Personal_Context_Profile.md \
    -f ref=dev --jq .content | tr -d '\n' | base64 -D | \
    sed -n '/<!-- shared-personal-context-contract:start -->/,/<!-- shared-personal-context-contract:end -->/p' | \
    tr -s '[:space:]' ' ')
```

Expected: no diff. This is the automated normalized-text parity check for the four shared-contract bullets.

- [ ] **Step 4: Run targeted contract, Settings, Console, linking, dispatcher, and client checks**

Run:

```bash
profile_python=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python
if [ ! -x "$profile_python" ]; then
  profile_python=.venv/bin/python
fi
if [ ! -x "$profile_python" ]; then
  echo 'No executable Chatbook project Python found'
  exit 1
fi
test "$("$profile_python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" = '3.12'
printf 'Using checked project Python 3.12: %s\n' "$profile_python"
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

Expected: the selected tests pass under the checked Chatbook Python 3.12 environment. Results from a different interpreter are not completion evidence.

- [ ] **Step 5: Run claim, path, and diff guards**

Run:

```bash
set -e
profile_user_guide=Docs/User_Guide/settings/personal-context-profile.md
profile_developer_guide=Docs/Development/personal-context-profile.md
profile_plan=Docs/superpowers/plans/2026-09-01-personal-context-documentation-chatbook.md
profile_task='backlog/tasks/task-27019 - Document-Personal-Context-Profile-for-Chatbook-users-and-developers.md'
for profile_shared_doc in "$profile_user_guide" "$profile_developer_guide"; do
  rg -Fq '<!-- shared-personal-context-contract:start -->' "$profile_shared_doc"
  rg -Fq '<!-- shared-personal-context-contract:end -->' "$profile_shared_doc"
done
rg -Fq 'Chatbook does not currently expose **Delete everywhere**.' "$profile_user_guide"
rg -Fq 'Ordinary server REST edits are not currently published to linked Chatbook clients.' \
  "$profile_user_guide"
rg -Fq 'Chatbook has no producer' "$profile_user_guide"
rg -Fq 'no dedicated Personal Context resolution screen' "$profile_user_guide"
rg -Fq 'Ordinary server REST edits are not currently published to linked Chatbook clients.' \
  "$profile_developer_guide"
rg -Fq 'Chatbook has no producer' "$profile_developer_guide"
rg -Fq 'no dedicated Personal Context' "$profile_developer_guide"
for profile_label in \
  "Profile locked" \
  "Offline or queued" \
  "Capability not negotiated" \
  "Version conflict" \
  "First-link semantic collision" \
  "Post-link semantic collision" \
  "Purge pending"; do
  rg -Fq "$profile_label" Docs/User_Guide/settings/personal-context-profile.md || {
    echo "Missing failure-state label: $profile_label"
    exit 1
  }
done

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

if profile_pending_steps=$(sed -n '/^### Task 1:/,/^### Task 6:/p' "$profile_plan" | rg -n '^- \[ \]'); then
  printf 'Unexecuted Task 1-5 plan steps:\n%s\n' "$profile_pending_steps"
  exit 1
else
  profile_pending_status=$?
  test "$profile_pending_status" -eq 1 || exit "$profile_pending_status"
fi

profile_changed_paths=$(
  {
    git diff --name-only origin/dev...HEAD
    git diff --name-only
    git diff --cached --name-only
  } | sed '/^$/d' | sort -u
)
profile_unexpected_paths=$(
  printf '%s\n' "$profile_changed_paths" | awk '
    $0 == "Docs/Development/Developer_Guide.md" { next }
    $0 == "Docs/Development/personal-context-profile.md" { next }
    $0 == "Docs/User_Guide/index.md" { next }
    $0 == "Docs/User_Guide/settings/personal-context-profile.md" { next }
    $0 == "Docs/superpowers/plans/2026-09-01-personal-context-documentation-chatbook.md" { next }
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

Expected: each guide independently proves its required shared-contract and current-limit claims; all seven user failure-state labels are explicit; every internal and server link target is checked; Tasks 1-5 have no unexecuted checkbox; and the allowed-path assertion accepts only the two guides, two discovery indexes, plan, and TASK-27019.

### Task 6: Close TASK-27019 and open the Chatbook PR

**Files:**

- Modify: `backlog/tasks/task-27019 - Document-Personal-Context-Profile-for-Chatbook-users-and-developers.md`

- [ ] **Step 1: Complete every acceptance criterion, record evidence, and mark Done as the final repository mutation**

Run, replacing the bracketed evidence with the exact commands and results from Task 5:

```bash
backlog task edit 27019 \
  --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 \
  --notes "Implemented the Chatbook Personal Context user and developer guides, discovery links, exact shared-contract parity block, current sync/non-sync matrix, seven failure states, and ten-item extension checklist. Verification: [exact Task 5 results]. ADR required: no new ADR required; existing ADR applies. ADR path: backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md. Reason: documentation only; the existing Personal Context authority, Sync, and encryption ADR applies. Lessons learned: [record a genuine lesson with its incident, or state none]." \
  -s Done
backlog task 27019 --plain
git add "backlog/tasks/task-27019 - Document-Personal-Context-Profile-for-Chatbook-users-and-developers.md"
git diff --check --cached
git commit -m "docs: close Chatbook Personal Context documentation task"
```

Expected: all ACs are checked, Implementation Notes/evidence are present, and TASK-27019 is Done. Do not rebase or modify repository files after this commit.

- [ ] **Step 2: Push and open the PR against `dev`**

Prepare `/tmp/personal-context-chatbook-pr.md` with summary, current limitations, and exact evidence, then run:

```bash
git push -u origin codex/personal-context-docs
gh pr create --base dev --head codex/personal-context-docs --title "docs: add Personal Context user and developer guides" --body-file /tmp/personal-context-chatbook-pr.md
```

Expected: a docs-only Chatbook PR against `dev` whose changed files match Task 5's inventory.
