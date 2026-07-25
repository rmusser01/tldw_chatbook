# Skill Script Execution

Skills authored to the [Agent Skills](https://agentskills.io) spec often ship helper
scripts (`scripts/extract.py`, `bin/render.sh`) that the skill's instructions tell the
agent to run. This feature lets those scripts actually execute, under three independent
gates and a best-effort sandbox.

## How a run is gated

Every single execution passes all three, in this order:

1. **Runtime policy.** The `skills.run_script.launch.local` action must be allowed.
   Unknown action ids fail closed, so a missing registry row denies rather than permits.
2. **Trust re-verification.** The skill's on-disk fingerprints are re-scanned on *every*
   run. A skill that is revoked, mutated, or quarantined mid-conversation stops being
   runnable immediately — nothing is cached from an earlier check.
3. **A human decision.** An in-chat card shows the skill, the script path, how it will be
   invoked, and the arguments, with **Allow once**, **Always allow this skill**, and
   **Deny**. Policy and resolution failures happen *before* the card, so you are never
   asked to approve something that was going to fail anyway. Any error in the confirm
   path denies.

"Always allow this skill" records a standing grant pinned to the skill's current
fingerprint digest. **Any change to the skill's files invalidates that grant
automatically** and the next run asks again. Grants are visible and revocable per skill
in **Library ▸ Skills**, in the trust panel ("Revoke script access").

## What can be run

Two mechanisms, chosen by the file itself:

- **Interpreter** — a text file whose extension is in a fixed map: `.py` → `python3`,
  `.sh` → `sh`, `.bash` → `bash`, `.js` → `node`. Interpreters resolve only against a
  scrubbed `PATH` (`/usr/bin:/bin`), never your environment, so a skill cannot shadow
  `sh` or `node`. Shebangs are ignored for these files.
- **Direct execution** — a file whose executable bit was captured in the trust
  fingerprint runs directly, shebang and all. This includes compiled binaries; the
  confirm card labels those explicitly, because a human cannot meaningfully review a
  binary at trust time.

Anything else (a text file with no mapped extension and no exec bit) is refused.
`SKILL.md` itself is never runnable.

## Sandbox

Scripts run with:

- **No shell.** The argv is always a list; nothing is shell-interpreted.
- **A scrubbed environment** — only `PATH=/usr/bin:/bin`, `HOME`, `TMPDIR`, and locale
  variables. **Your API keys are never passed to a skill script.**
- **A fresh scratch working directory** per run, deleted afterwards. It is never the
  skill's own directory, so a script cannot tamper with its own trusted bundle.
- **Resource limits** applied in the child before exec: CPU seconds, address space,
  open files, and maximum file size.
- **A wall-clock deadline**, after which the whole process *group* is killed — so a
  script cannot leave background helpers running.
- **Capped output.** stdout and stderr are each retained up to a byte cap while the
  remainder is drained and discarded, so a runaway writer can neither exhaust memory nor
  deadlock on a full pipe.

Only stdout, stderr, and the exit code come back to the agent. Files a script writes are
discarded with the scratch directory.

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
  and scratch cwd stop casual access and self-tampering, but a determined script can read
  files your user account can read.
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
