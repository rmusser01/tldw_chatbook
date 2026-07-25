# Skill Script Execution

Skills authored to the [Agent Skills](https://agentskills.io) spec often ship helper
scripts (`scripts/extract.py`, `bin/render.sh`) that the skill's instructions tell the
agent to run. This feature lets those scripts actually execute, under three independent
gates and a best-effort sandbox.

## Platform support

**Script execution is POSIX-only (macOS/Linux) and is unavailable on Windows.** The
sandbox (`Skills_Interop/skill_script_runner.py`) depends on POSIX-only primitives —
`start_new_session=True` for process-group teardown, `os.killpg`/`os.getpgid`, and a
trampoline that applies resource limits via the POSIX-only `resource` module — none of
which exist on Windows.

Rather than fail mid-run on an unsupported platform, `sandbox_supported()` is checked
first and the `run_skill_script` runtime tool is simply never wired up (never advertised
to the agent) when it returns `False`. The rest of tldw_chatbook, including reading skill
files and every other runtime tool, is unaffected — only script execution is gated by
platform.

## How a run is gated

Every single execution passes all three, in this order:

1. **Runtime policy.** The `skills.run_script.launch.local` action must be allowed.
   Unknown action ids fail closed, so a missing registry row denies rather than permits.
2. **Trust re-verification.** The skill's on-disk fingerprints are re-scanned on *every*
   run, and the requested script must itself appear in the skill's trusted manifest. A
   skill that is revoked, mutated, or quarantined mid-conversation stops being runnable
   immediately — nothing is cached from an earlier check.

   The manifest check matters because trust review is not a whole-directory guarantee.
   The fingerprint scan deliberately prunes VCS/OS/build junk (`node_modules/`, `.git/`,
   `__pycache__/`, `*.tmp`, `*.pyc`, `*~`, `.DS_Store`) so a real bundle's litter cannot
   make a skill permanently untrustable. A pruned file is therefore never fingerprinted,
   never shown in the trust review, and never covered by any of the guarantees on this
   page — so it is **not runnable at all**, executable bit or not. Execution asks the
   manifest, not the path validator: explicitly trusted, or refused. A file that exists
   but is not trust material is refused with the same "script not found" error as a
   missing one, so the refusal cannot be used to probe what a bundle contains.
3. **A human decision.** An in-chat card shows the skill, the script path, how it will be
   invoked, and the arguments, with **Allow once**, **Always allow this skill**, and
   **Deny**. Policy and resolution failures happen *before* the card, so you are never
   asked to approve something that was going to fail anyway. Any error in the confirm
   path denies.

"Always allow this skill" records a standing grant pinned to the skill's current
fingerprint digest. **Any change to a fingerprinted file invalidates that grant
automatically** — content, size, or executable bit, on any file the trust review covers —
and the next run asks again. (Junk-pruned paths are outside that digest, which is exactly
why they are not runnable in the first place; nothing a grant can cover falls outside it.)
Deleting a skill also drops its grant, so reinstalling the same name never silently
reactivates a permission you gave to a previous installation. Grants are visible and
revocable per skill in **Library ▸ Skills**, in the trust panel ("Revoke script access").

## What can be run

Two mechanisms, chosen by the file itself:

- **Interpreter** — a text file whose extension is in a fixed map: `.py` → `python3`,
  `.sh` → `sh`, `.bash` → `bash`, `.js` → `node`. Interpreters resolve only against a
  scrubbed `PATH` (`/usr/bin:/bin`), never your environment, so a skill cannot shadow
  `sh` or `node`. Shebangs are ignored for these files.
- **Direct execution** — a file that is executable *on disk at run time* runs directly,
  shebang and all. The mechanism is chosen from a live `stat()`, not from the manifest;
  the manifest is what decides whether the file may run at all. The two cannot drift
  apart in practice, because the executable bit is itself part of a file's fingerprint:
  flipping it changes that file's manifest entry, which quarantines the skill
  (`quarantined_modified`) and drops any standing grant, so the change has to be
  re-reviewed before anything runs again. Direct execution includes compiled binaries;
  the confirm card labels those explicitly, because a human cannot meaningfully review a
  binary at trust time.

Anything else (a text file with no mapped extension and no exec bit) is refused. Files
the trust manifest does not fingerprint are refused before this classification even
happens. `SKILL.md` itself is never runnable.

## Sandbox

Scripts run with:

- **No shell.** The argv is always a list; nothing is shell-interpreted.
- **A scrubbed environment** — only `PATH=/usr/bin:/bin`, `HOME`, `TMPDIR`, and locale
  variables. **Your API keys are never passed to a skill script.**
- **A fresh scratch working directory** per run, deleted afterwards. It is never the
  skill's own directory, so *relative* writes land in scratch rather than in the bundle.
  This is a default, not confinement: a script that writes to an absolute path can still
  write anywhere your user account can (see Residual risks).
- **Resource limits** applied in the child before exec: CPU seconds, address space,
  open files, and maximum file size.
- **A wall-clock deadline**, after which the whole process *group* is killed — so a
  script cannot leave background helpers running.
- **Capped output.** stdout and stderr are each retained up to a byte cap while the
  remainder is drained and discarded, so a runaway writer can neither exhaust memory nor
  deadlock on a full pipe.

Only stdout, stderr, and the exit code come back to the agent. Files a script writes into
its working directory are discarded with the scratch directory; files it writes elsewhere
by absolute path are not.

### Current limits

These are fixed defaults in `Skills_Interop/skill_script_runner.py`
(`ScriptRunLimits`) and are **not** currently exposed as configuration:

| Limit | Default |
|---|---|
| CPU time | 10 s |
| Address space | 512 MiB |
| Open files | 128 |
| Max single file size | 8 MiB |
| Wall clock | 60 s |
| Retained output (per stream) | 64 KiB |

## Configuration

One knob exists today, under a `[skills]` section in `~/.config/tldw_cli/config.toml`:

```toml
[skills]
# Optional. Parent directory for the per-run scratch working directory.
# Defaults to the OS temp directory when unset.
script_scratch_root = "/path/to/scratch"
```

A configured root that resolves inside the skills store or the trust store is **rejected**
and the OS temp directory is used instead — otherwise a script could be handed a working
directory inside its own bundle.

> **Implementation note:** read this setting with the three-argument form,
> `get_cli_setting("skills", "script_scratch_root", default)`. The section-dict form
> (`get_cli_setting("skills", {})`) silently returns `{}` for any section name without a
> dot, which would make the setting permanently unreachable.

## Residual risks

The sandbox is best-effort, not a jail. Known and accepted:

- **Network access is not blocked.** A script may open sockets. Blocking this requires
  real OS-level sandboxing.
- **Reads outside the scratch directory are still possible.** The scrubbed environment
  and scratch cwd stop casual access, but a determined script can read files your user
  account can read.
- **Writes outside the scratch directory are still possible.** Only the *working
  directory* is scratch; nothing blocks an absolute path. A script can write anywhere
  your user account can — including back into its own bundle or another skill's. A write
  that lands on a *fingerprinted* file is caught: it quarantines the skill and drops any
  standing grant at the next check. A write to a junk-pruned path (`node_modules/`,
  `*.tmp`, ...) leaves the digest untouched and raises no alarm. That cannot escalate into
  execution, because only manifest-fingerprinted files can ever run — but it is not
  invisible either: a pruned file's contents are still *readable* by the agent through the
  `skill_file` tool, so a script can leave text there that a later turn reads and acts on,
  and the trust review will never have shown it to you.
- **Memory is not capped on macOS/BSD.** `RLIMIT_AS` cannot be lowered there, so peak
  memory is bounded only by the CPU and wall-clock limits. A warning is surfaced with the
  run when this applies.
- **Compiled binaries are approved sight-unseen.** Trust review shows binaries only as a
  size and hash, so "direct execution" of a binary means running bytes no human read. The
  confirm card names this case explicitly.
- **A standing grant means later runs are silent**, and any agent in the run tree —
  including a subagent processing untrusted content — can trigger them with
  runtime-chosen arguments. Arguments are always runtime input; the sandbox, not review,
  is what contains hostile arguments. If you want every run re-confirmed, do not grant
  "Always allow" for that skill.

Real containment (a macOS `sandbox-exec` profile or Linux seccomp/namespaces) is a
possible future layer; it is deliberately not part of this one.
