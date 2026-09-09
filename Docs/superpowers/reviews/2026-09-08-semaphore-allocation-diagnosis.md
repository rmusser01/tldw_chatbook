# Read-only semaphore allocation diagnosis — 2026-09-08

TASK-32160 continuation, authorized by the user after the SQLite review-fix
checkpoint `50a57004220645665f528edad31f7a31f9a608f2`.

## Result

A fresh isolated **standard-library-only** process cannot allocate one spawn
context lock: `OSError`, errno 28, both inside and outside the execution sandbox.
No Chatbook, SQLite, pytest, configuration or helper module is imported by that
probe. The failure therefore exists independently of Chatbook's live owner path;
it is not repaired by the completed SQLite correction.

The host exposes `kern.posix.sem.max: 10000`. Apple's published XNU implementation
returns ENOSPC when adding a named semaphore would exceed that global cache
limit. This strongly supports named-semaphore capacity exhaustion as the cause;
the running release does not expose a current-count key in this sysctl subtree,
so 10,000 occupied entries were not directly measured. The source also separates
name-cache removal from closing open handles, explaining why a process-handle
count need not account for all occupied names. This source is explanatory, not
a byte-for-byte validation of the running kernel.
[Apple XNU semaphore implementation](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/posix_sem.c).

The data volume reported about 21 GiB available with 99% capacity used and about
223 million free inodes. Disk space is low, but it was not exhausted at this
measurement; deleting disk files is not a justified remedy for this identified
semaphore allocation failure.

## Reproduction

Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/canvas-v1`.
Interpreter: shared venv's CPython 3.12.11, using `-I` isolated mode. No environment
or package changes were made. Exact probe:

```sh
../../.venv/bin/python -I -c 'import gc, json, multiprocessing as mp, sys
try:
    probe = mp.get_context("spawn").Lock()
    acquired = probe.acquire(False)
    if acquired:
        probe.release()
    del probe
    gc.collect()
    print(json.dumps({"probe": "stdlib_spawn_lock", "allocated": True, "acquired": acquired, "owned_cleanup": "finalizer_completed"}))
except OSError as error:
    print(json.dumps({"probe": "stdlib_spawn_lock", "allocated": False, "error_type": type(error).__name__, "errno": error.errno}))
    sys.exit(1)'
```

Both executions exited 1 and returned:

```json
{"probe": "stdlib_spawn_lock", "allocated": false, "error_type": "OSError", "errno": 28}
```

No successful allocation occurred; neither run created a semaphore requiring
cleanup. The success arm would dispose only of the probe's own lock. Static
inspection of the installed `multiprocessing/synchronize.py` confirms that the
failing construction precedes resource-tracker registration.

## Read-only host observations and limits

- `sysctl kern.posix.sem.max`: 10000. `sysctl kern.posix.sem` exposes only this
  maximum here. The optional `kern.posix.sem.value_max` key is unavailable.
- `df -h /private/tmp`: 1.8 TiB volume, 1.7 TiB used, 21 GiB available, 99% used.
- Process inspection read only PID, parent PID, elapsed time and executable
  name, not arguments or file contents. Long-running Python processes exist,
  including some adopted by PID 1; age or orphan status does not establish that
  they own leaked semaphore names or are safe to terminate.
- `lsof -nP -F pt`, reduced immediately to PID/type counts, exposed 48 POSIX
  semaphore handles across three processes and 1,343 visible processes in that
  snapshot. The six inspected orphaned Python processes had zero visible handles.
  This is **not** a complete name-cache census or evidence identifying the
  creator of exhausted capacity. It does not justify blaming those processes.
- Initial sandbox reads of sysctl/process metadata were denied. Repeating only
  the authorized reads outside the sandbox succeeded. The allocation failure
  itself remained errno 28 outside the sandbox, not a sandbox-denial result.

No process was stopped, no named semaphore was unlinked manually, no kernel
limit was changed, and no disk cleanup, reboot or dependency repair was attempted.
All process/file metadata inspection was for this diagnosis; no user content was
read or included here.

## Qualification disposition

The eleven previously blocked spawned TTS tests remain unqualified. Repeating
all eleven while this one-lock control fails would not exercise their bodies.
Run the control again only after an external-state change; once it succeeds,
rerun the exact three parameter groups from Task7 before declaring those gates
passed. Do not substitute a threading lock or serial-only test for their
cross-process contract.

Safe next choices are a clean qualification host/runner, or a user-scheduled
restart after saving and coordinating active work. Removing arbitrary semaphore
names or terminating unowned processes is outside this authorization and cannot
be justified from the available attribution evidence. No recovery has yet been
performed or claimed.
