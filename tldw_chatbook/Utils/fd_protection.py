"""Shared file-descriptor protection for macOS subprocess-spawning issues.

task-640 item 5: this function used to be duplicated verbatim in three
places (``Embeddings/Embeddings_Lib.py``, ``Local_Ingestion/
transcription_service.py``, ``TTS/backends/higgs.py``; ``TTS/backends/
chatterbox.py`` imported its copy from ``Embeddings_Lib.py``) -- every fix
to it (the ``closefd=False`` fix and the ``created_out``/``created_err``
tracking fix, both task-641 round 3) had to land 3 times. It now lives here
once; all four call sites import it from this module instead.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from contextlib import contextmanager

__all__ = ["protect_file_descriptors"]

# task-640 item 4: protect_file_descriptors() mutates GLOBAL process state
# (sys.stdout/sys.stderr/sys.stdin/os.environ) with no synchronization of
# its own. Two threads calling it concurrently (e.g. two worker threads
# each loading a HuggingFace model at once -- RAG embeddings + Chatterbox/
# Higgs TTS are all real call sites) can interleave their save/reassign/
# restore sequences: thread A saves the "original" streams, thread B
# reassigns them again before A's protected region is done, and either
# thread's `finally` can restore the WRONG "original" for the other,
# leaving sys.stdout/stderr pointed at an already-closed wrapper. This lock
# is held for the ENTIRE context-manager body below -- not just the
# sys.stdout/stderr swap at entry -- because the hazard spans the whole
# protected region: the actual model-loading call happens at the `yield`,
# and that is exactly the window a second caller must not be allowed to
# start its own save/reassign into. A plain (non-reentrant) Lock is used
# deliberately: every known call site wraps a single, non-recursive
# operation (a `from_pretrained()`-style call), so there is no legitimate
# same-thread nesting to accommodate -- if one ever appeared, a plain Lock
# fails loudly (deadlock) rather than an RLock silently granting the inner
# call access to a half-restored outer state.
#
# Known cross-feature cost of a process-wide lock (task-640 review,
# accepted for this wave, not fixed): this is one lock shared by every
# call site, not scoped per-feature -- `TTS/backends/chatterbox.py` wraps
# its FULL audio-generation call (`self.model.generate(...)`, not just
# the model-load) in `protect_file_descriptors()` at two sites, so a
# single long Chatterbox generation can hold this lock for as long as
# generation takes (potentially minutes for long text). For that entire
# window, every OTHER caller -- a RAG embedding model load, a
# transcription model load, or a concurrent Higgs/second Chatterbox
# generation -- blocks waiting for the lock, even though none of them
# touch the same audio model. This is a real, currently-accepted
# regression in cross-feature concurrency versus the pre-lock behavior
# (where at worst streams could race, but callers never serialized on
# each other); narrowing `protect_file_descriptors()`'s callers to just
# the actual model-load/subprocess-spawn moment (not the full generation
# call) instead of holding the lock across unrelated, potentially slow
# work is a follow-up, not addressed here.
_fd_protection_lock = threading.Lock()


@contextmanager
def protect_file_descriptors():
    """Context manager to protect file descriptors during subprocess operations.

    This fixes the "bad value(s) in fds_to_keep" error on macOS when the
    transformers library spawns subprocesses for model downloads.

    task-641 round 3: when ``sys.stdout``/``sys.stderr`` are non-fd-backed
    (e.g. Textual redirects BOTH to non-fd capture objects for the ENTIRE
    ``App.run()`` lifetime, on every thread -- see ``textual/app.py``'s
    ``with redirect_stdout(self._capture_stdout): with
    redirect_stderr(self._capture_stderr): await run_process_messages()``),
    the except-branch below used to do
    ``sys.stdout = os.fdopen(1, "w")`` / ``sys.stderr = os.fdopen(2, "w")``
    with ``os.fdopen``'s default ``closefd=True``. Those temporary wrapper
    objects OWNED the real, process-shared fd 1/2; the very next statement
    in the ``finally`` below (``sys.stdout = original_stdout``) dropped the
    only reference to them, CPython's refcounting GC finalized them
    synchronously right there, and their ``__del__``/``close()`` closed the
    shared fd 1/2 for the WHOLE PROCESS -- including whatever else (like
    Textual's own output ``WriterThread``, which writes to
    ``sys.__stderr__``, captured once at driver init and never
    re-resolved) still depended on that fd staying open. A single RAG
    Backfill worker thread loading a HuggingFace embedding model
    (``_HuggingFaceEmbedder``) through here was enough to silently kill
    Textual's ``WriterThread`` (an unguarded ``OSError`` on its next
    ``write()``) and permanently deadlock the main thread on its next
    (bounded, 30-slot) output-queue write -- a live re-UAT 100%-reproducible
    freeze traced via ``faulthandler`` all-threads dumps, confirmed
    empirically. ``closefd=False`` is the fix: a throwaway text wrapper
    around a fd it does not own must never be allowed to close that fd.

    task-641 round-3 review: the ``finally`` block must never close
    "whatever is currently in ``sys.stdout``/``sys.stderr``" -- if code
    inside the protected ``yield`` (e.g. a nested library call) reassigns
    ``sys.stdout``/``sys.stderr`` itself and leaves it there (for example
    ``sys.stdout = sys.__stdout__``), that would hand the SAME cleanup
    logic a REAL, fd-owning stream to close -- reopening the exact
    WriterThread-killing hazard through a different door. Only the wrapper
    objects THIS function itself creates (tracked via ``created_out``/
    ``created_err``, assigned once at creation and never re-read from
    ``sys.stdout``/``sys.stderr``) are ever closed here.

    task-640 item 4: the entire body below runs under a module-level lock
    (see ``_fd_protection_lock`` above) so two threads can never interleave
    their save/reassign/restore sequences against the shared global state.
    """
    with _fd_protection_lock:
        # Save original file descriptors
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        original_stdin = sys.stdin

        # Save original environment
        env_backup = os.environ.copy()

        # Save original subprocess.Popen to restore later
        original_popen = subprocess.Popen

        # Tracks ONLY the wrapper object(s) this function itself creates below
        # -- never read back from sys.stdout/sys.stderr at cleanup time (see
        # docstring). A single shared devnull (the fallback path) is assigned
        # to both; closing it twice is harmless.
        created_out = None
        created_err = None

        try:
            # Ensure we have real file descriptors, not wrapped objects
            # This is crucial for subprocess operations
            try:
                # Test if stdout/stderr are real files with valid file descriptors
                stdout_fd = sys.stdout.fileno()
                stderr_fd = sys.stderr.fileno()
                # Verify they're valid by attempting to use them
                os.fstat(stdout_fd)
                os.fstat(stderr_fd)
            except (AttributeError, ValueError, OSError):
                # stdout/stderr are wrapped/captured or invalid, create new ones.
                # Use the original file descriptors 1 and 2 directly --
                # closefd=False: these text wrappers do NOT own fd 1/2 (the
                # process's shared stdout/stderr), so they must never close
                # them, whether explicitly or via GC finalization. See this
                # function's docstring (task-641 round 3).
                try:
                    created_out = os.fdopen(1, "w", closefd=False)
                    created_err = os.fdopen(2, "w", closefd=False)
                    sys.stdout = created_out
                    sys.stderr = created_err
                except OSError:
                    # If that fails, use devnull as a fallback. This process
                    # DOES fully own this file, so it's fine (and correct) to
                    # close it in the finally below.
                    devnull = open(os.devnull, "w")
                    created_out = devnull
                    created_err = devnull
                    sys.stdout = devnull
                    sys.stderr = devnull

            # Set environment to prevent subprocess issues
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

            # For macOS specifically
            if sys.platform == "darwin":
                os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
                # Ensure subprocess doesn't inherit bad file descriptors
                os.environ["PYTHONNOUSERSITE"] = "1"
                # Force subprocess to close all file descriptors except 0,1,2
                os.environ["PYTHON_SUBPROCESS_CLOSE_FDS"] = "1"

            yield

        finally:
            # Always restore the TRUE originals, regardless of what code inside
            # `yield` may have reassigned sys.stdout/sys.stderr to.
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            sys.stdin = original_stdin

            # Close ONLY the wrapper(s) this function created -- NEVER whatever
            # currently sits in sys.stdout/sys.stderr (task-641 round-3 review;
            # see docstring). closefd=False already makes the fdopen(1/2) case
            # harmless either way, but this also correctly closes the devnull
            # fallback, which this process does fully own.
            if created_out is not None:
                try:
                    created_out.close()
                except Exception:
                    pass
            if created_err is not None and created_err is not created_out:
                try:
                    created_err.close()
                except Exception:
                    pass

            # Restore original environment
            os.environ.clear()
            os.environ.update(env_backup)

            # Restore subprocess.Popen
            subprocess.Popen = original_popen
