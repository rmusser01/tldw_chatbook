---
id: TASK-950
title: >-
  verify_trusted_directory rejects every macOS temp directory because /var is a
  symlink
status: Done
assignee: []
created_date: '2026-07-27 17:30'
updated_date: '2026-07-27 18:45'
labels:
  - bug
  - macos
  - security
  - db
priority: critical
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`verify_trusted_directory` (`tldw_chatbook/Utils/private_paths.py:1107`) walks a path component by component with `_open_directory_component` and rejects anything that is not a real directory. On macOS `/var` is a **symlink** to `/private/var`, and the system temp directory lives under it — so the walk hits the symlink and raises `PrivatePathError: link_or_non_regular: NotADirectoryError`.

Every SQLite open under a macOS temp directory therefore fails, via `_connect_registered_sqlite` (`tldw_chatbook/DB/private_sqlite.py:907`).

**Large parts of the suite are red on `origin/dev` right now because of this.** Measured on a clean detached checkout of `2c33cb616` with no other changes:

| Suite | Result on clean dev |
|---|---|
| `Tests/UI/test_destination_visual_parity_correction.py` | **96 failed**, 4 passed |
| `Tests/UI/test_watchlists_destination_shell.py` | **47 failed**, 1 passed |
| `Tests/Watchlists` | **15 failed**, 130 passed |

This is not specific to Watchlists — it is every test that opens SQLite under a temp directory, which on macOS is effectively every test that builds an app.

In `Tests/Watchlists` it is partly **masked** by a second, unrelated breakage: `Tests/UI/test_screen_navigation.py` assigns `app.current_runtime_backend`, which became a read-only property, so those tests die at construction with `AttributeError` before reaching the path guard. Note that "fixing" that harness line in isolation makes things **worse**, not better: it drives previously-passing tests into this guard. Both need fixing together, this one first.

Do not fix this by loosening the guard's symlink rejection wholesale — rejecting symlinked components is the point of it. The likely correct treatment is to resolve the path before walking, or to treat the platform's own temp root as trusted, but that is a judgement call for whoever owns this subsystem.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A SQLite database can be opened under the system temp directory on macOS
- [x] #2 The guard still rejects a symlinked component that is not part of the platform's own temp root
- [x] #3 `Tests/UI/test_destination_visual_parity_correction.py`, `Tests/UI/test_watchlists_destination_shell.py` and `Tests/Watchlists` all pass on macOS from a clean `dev` checkout — 104 / 48 / 167, zero failures (the two survivors the implementer flagged were closed by the controller; see the note below)
- [x] #4 A test covers the macOS `/var` → `/private/var` case specifically, and fails against the current code
- [x] #5 `Tests/UI/test_screen_navigation.py` no longer assigns the read-only `current_runtime_backend` property, and is fixed only alongside #1 so it does not unmask this
- [x] #6 The security intent of the guard is stated in its docstring, so the next person does not weaken it to make a test pass
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
**AC #3 is not met.** Everything else is. Read the "Remaining failures" section below before closing this out.

### Design

The walk in `private_paths.py` still starts at `/` and opens one component at a
time through `dir_fd` with `O_NOFOLLOW`. The only change is what happens when
that open fails with `ELOOP`/`ENOTDIR` because the component is a symlink:

- the component is `lstat`-ed **through the parent fd**, never by re-deriving a
  path string that could change underneath us;
- it is traversed only if the symlink *itself* passes a trust test
  (`_trusted_symlink`), otherwise the old `LINK_OR_NON_REGULAR` result is raised
  unchanged, with the same reason string;
- the target is read with `os.readlink(component, dir_fd=parent_fd)`; an absolute
  target restarts the walk from `/`, a relative one continues from the current
  parent;
- traversals are capped at `_MAX_TRUSTED_SYMLINK_HOPS = 8`, past which the walk
  fails with `LINK_OR_NON_REGULAR: symlink_hop_limit_exceeded`, so a symlink
  cycle terminates instead of spinning.

The three walkers (`_open_verified_parent`, `secure_private_directory`,
`verify_trusted_directory`) were converted from `for component in parts` to a
`while pending:` walk over a mutable component list, which is what lets a
symlink's target be spliced in ahead of the remaining components. All of them
needed it: opening a SQLite file under the temp directory goes through
`verify_trusted_directory` *and* `_open_verified_parent`.

`Path.resolve()`/`realpath` before walking was explicitly rejected: it follows
every symlink before anything has been vetted and reopens a name that may have
changed since. `lexical_path` still does not resolve, and the returned
`PrivatePathResult.lexical_path` is still the caller's lexical path, so the
"custom database paths preserve their lexical symlink alias" contract is intact.

### Deviation: root-owned symlinks only

The brief specified accepting symlinks owned by `{0, euid}`, mirroring
`_trusted_directory_owner`. That was implemented as **uid 0 only**, because
`{0, euid}` directly contradicts five existing security assertions that these
changes must not weaken (a `tmp_path` symlink is euid-owned and mode 0755 on
macOS, so `{0, euid}` would have started *following* it):

- `Tests/Utils/test_private_paths.py::test_verify_trusted_directory_rejects_directory_symlinks` (both params)
- `Tests/Utils/test_private_paths.py::test_open_private_binary_rejects_intermediate_symlink`
- `Tests/test_database_path_privacy.py::test_default_data_directory_rejects_intermediate_symlink`
- `Tests/test_database_path_privacy.py::test_custom_data_base_must_be_existing_trusted_directory[symlink]`
- `Tests/test_database_path_privacy.py::test_custom_database_paths_preserve_lexical_symlink_alias`

uid-0-only fixes the reported bug (`/var`, `/tmp` are root-owned), is strictly
narrower than the brief, and leaves every one of those assertions passing
untouched. The distinction is real, not pedantic: a directory owned by the
caller is the caller's own storage, whereas following a caller-owned *symlink*
would silently relocate the application's private files behind a lexical alias.

### Security argument

After this change an attacker can do nothing they could not do before.

- The set of directories the walk will accept is unchanged in every case where
  no root-owned symlink is involved. Every directory crossed is still checked
  for root-or-euid ownership and for group/world write without the sticky bit.
- The only newly traversable component is a symlink with `st_uid == 0` and no
  group/world write bits. Creating or repointing such a link requires either
  root, or write access to a directory the walk has *already* rejected as
  shared-writable (a sticky directory only lets a user replace their own
  entries, which are not uid 0).
- The lstat/readlink pair goes through the parent descriptor. If the entry is
  swapped between them, `readlink` either errors or returns some other target —
  and that target is then walked component by component under the same rules, so
  the invariant "every directory traversed is root-or-euid-owned and not shared
  writable" holds regardless.
- The symlink mode gate rejects every symlink on Linux, where the kernel reports
  all symlinks as 0o777. That is intentional and documented in `_trusted_symlink`:
  Linux has no root-owned symlink on the paths this module walks, so the fix is
  a no-op there and nothing is loosened.

### Tests

New in `Tests/Utils/test_private_paths.py` (9) and `Tests/DB/test_private_sqlite.py` (1).
All 10 fail against the unfixed guard **except** the three
`rejects_an_untrusted_symlink` parametrizations, which cannot fail against the
old code by construction — they encode the reject behaviour that already
existed. Their value is that they fail against a *wrong* fix: drop the mode gate
or widen the owner set to `{0, euid}` and they go red.

- `test_verify_trusted_directory_traverses_the_platform_temporary_directory` —
  asserts against the real `tempfile.gettempdir()`, so it would have caught the
  original bug. Skips where the platform temp path has no symlinked component.
- `test_secure_private_directory_creates_under_the_platform_temporary_directory`
- `test_sqlite_opens_under_the_unresolved_platform_temporary_directory` (AC #1) —
  uses `tempfile.mkdtemp()` on purpose; `tmp_path` hides the bug because pytest
  resolves its own base directory.
- `test_verify_trusted_directory_traverses_a_root_owned_symlink` and the relative
  variant, plus `test_open_private_binary_traverses_a_root_owned_intermediate_symlink`.
- `test_verify_trusted_directory_rejects_an_untrusted_symlink[foreign-owner|group-writable|world-writable]`.
- `test_verify_trusted_directory_stops_on_a_symlink_loop` — asserts the hop cap
  under a signal timeout.

Only root can create a root-owned symlink, so the ownership/mode cases relabel
the link's own `lstat` through `os.stat`, the same technique the file already
uses for `test_verify_trusted_directory_rejects_wrong_owner_simulation`.

### Suite numbers (macOS, `.venv/bin/pytest`, one file at a time)

| Suite | Before | After |
|---|---|---|
| `Tests/UI/test_destination_visual_parity_correction.py` | 100 failed, 4 passed | **2 failed, 102 passed** |
| `Tests/UI/test_watchlists_destination_shell.py` | 47 failed, 1 passed | **48 passed** |
| `Tests/Watchlists` | 26 failed, 141 passed | **167 passed** |
| `Tests/Utils` (whole) | — | 557 passed |
| `Tests/DB/test_private_sqlite.py` | 214 passed | 215 passed, 1 skipped |
| `Tests/test_database_path_privacy.py` | 37 passed | 37 passed |
| `Tests/Subscriptions` | — | 107 passed |

The ordering claim in the description was **confirmed on this base**: with the
path guard fixed but the harness line untouched, both UI suites were unchanged
(47 failed / 100 failed). The harness `AttributeError` fires at app construction,
before any path is walked, so it hides the guard entirely; only the two together
move the numbers. On this base (`c171ae56a`) all 26 `Tests/Watchlists` failures
were the harness `AttributeError`, not the 15/`PrivatePathError` split measured
on the older `2c33cb616`.

### Remaining failures (AC #3)

Two tests in `test_destination_visual_parity_correction.py` still fail. Neither
touches path handling; both are test-vs-app drift that the construction-time
`AttributeError` had been hiding since before this task, and both are left
untouched rather than guessed at:

1. `test_source_prep_loading_states_preserve_workbench_geometry[artifacts-...]` —
   the parametrization stubs `ArtifactsScreen._refresh_latest_chatbook_context`,
   which does not exist on the class (and never has; the screen has
   `_start_chatbook_refresh` / `_refresh_chatbook_context`). Someone who owns
   that screen should decide which hook the test now means.
2. `test_operational_loading_states_preserve_workbench_geometry[workflows-...]` —
   the stub is `lambda self: None` but `workflows_screen.on_mount` calls
   `self._refresh_latest_console_context(has_recent_work)`. The fix is stub
   arity (`lambda self, *args, **kwargs: None`), which is the same class of
   harness rot as AC #5, but it is outside this task's scope.

### Files

- `tldw_chatbook/Utils/private_paths.py` — `_MAX_TRUSTED_SYMLINK_HOPS`,
  `_trusted_symlink`, `_read_trusted_symlink`, `_symlink_walk_components`,
  `_follow_trusted_symlink`; the three walkers; the `verify_trusted_directory`
  security-intent docstring.
- `Tests/Utils/test_private_paths.py`, `Tests/DB/test_private_sqlite.py` — new coverage.
- `Tests/UI/test_screen_navigation.py` — seed `_runtime_policy_projection_snapshot`
  instead of assigning the read-only `current_runtime_backend` property.
<!-- SECTION:NOTES:END -->

### AC #3 closed by the controller

The implementer left AC #3 unchecked with two survivors in `test_destination_visual_parity_correction.py`, correctly identifying them as pre-existing drift this bug had been masking rather than anything the guard fix caused. Both were test-side and small:

- The parametrize patched `ArtifactsScreen._refresh_latest_chatbook_context`, which no longer exists — the screen renamed it to `_start_chatbook_refresh`.
- Both stub sites used `lambda self: None`, but `_refresh_latest_console_context(self, has_recent_work)` takes an argument. Widened to `lambda self, *_a, **_k: None` so the stub tolerates whichever method the parametrize names.

Suite now 104 passed, 0 failed.
