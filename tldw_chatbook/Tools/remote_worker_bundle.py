"""Single-file remote workspace worker — GENERATED, COMMITTED ARTIFACT.

Produced by ``python -m tldw_chatbook.Tools.build_remote_worker_bundle``
from the real dependency modules listed in its ``BUNDLE_MODULES``. Do not
edit by hand: regenerate instead, or the drift guard
(``Tests/Tools/test_remote_worker_bundle.py``) fails on the next run.

Contract:

* Runs on a bare remote interpreter, Python 3.10 or newer, with ONLY
  the standard library available — at import time AND at runtime. Every
  import root in this file is stdlib; the builder rewrote the parent's
  lazy third-party imports into loud ``ImportError`` raises that the
  source-level guards degrade around.
* ``main(stream)`` is the entry: ``stream`` is the buffered stdin
  positioned AFTER the bootstrap consumed this bundle's own bytes.
  Loaders that ``exec`` this file must register the executing namespace
  in ``sys.modules`` first — the closure's ``dataclass(slots=True)``
  classes resolve their defining module through it. The fixed remote
  bootstrap needs no such registration: it execs this file inside the
  interpreter's own ``__main__`` namespace, and the artifact's FINAL
  line — the ``BUNDLE_SHA256`` assignment — triggers
  ``_enter_worker_exchange`` (defined above, inside the stamped region),
  which runs the one exchange and propagates the worker exit code.
* Every response frame is emitted as ``RESPONSE_MAGIC + <json frame>``;
  use ``split_magic(raw)`` to strip before parsing. The LOCAL worker
  does not add this prefix — its pipe has no noise source.
* The ``ping`` operation (Task 9) is dispatched BEFORE the root pin: it
  captures the full root-to-``/`` directory identity chain, canonical
  path, remote python version, and ``BUNDLE_SHA256`` so a first-contact
  caller can build every other operation's pinned request.
* ``BUNDLE_SHA256`` (the final line) is the SHA-256 of this file's
  bytes ABOVE its own assignment line — a full-file digest is not
  self-embeddable (the stamp would change its own input). Every
  executable byte, including the bootstrap entry logic, lives in that
  stamped region; only the assignment's own line falls outside it (and
  under the bootstrap the entry helper's ``SystemExit`` fires from that
  line, so nothing after it could ever run). Derive the same value from
  the artifact with
  ``build_remote_worker_bundle.expected_bundle_stamp``.
* Two-tier hard timeout (Task 12): ``run_workspace_worker`` arms the
  watchdog from ``Tools/worker_watchdog`` with the request's
  ``timeout_seconds`` right after decoding — a graceful ``Timer`` that
  sweeps ``TEMP_REGISTRY``, writes the fixed ``tldw-worker-watchdog``
  stderr line and ``os._exit(75)``, backed by a default-action
  ``signal.alarm`` that kills the process even when a catastrophic
  regex starves the GIL (and the Timer with it). Exit 75 is reserved
  to the watchdog; every other worker failure path exits 2.
* ``REMOTE_SENSITIVE_PATHS`` (embedded from
  ``Tools/remote_sensitive_paths.py``) are remote-home-relative paths the
  worker must never touch; enforcement wiring lands with the remote
  binding tasks.
"""
from __future__ import annotations

# ===========================================================================
# Section: tldw_chatbook.Utils.filesystem_identity (extracted from the live module by the
# builder; regenerate rather than editing)
# ===========================================================================
"""Fail-closed canonical directory identity capture shared by workspace code."""
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
_WINDOWS = os.name == 'nt'
_REPARSE_POINT = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', None)

class DirectoryIdentityError(ValueError):
    """Raised when a directory identity cannot be established safely."""

@dataclass(frozen=True, slots=True)
class DirectoryIdentity:
    """Stable metadata needed to recognize one directory."""
    device: int
    inode: int
    mode: int
    reparse: bool

@dataclass(frozen=True, slots=True)
class DirectoryChain:
    """Canonical root and its root-first directory ancestor identities."""
    canonical_root: Path = field(repr=False)
    identities: tuple[DirectoryIdentity, ...] = field(repr=False)

def directory_identity_from_stat(value: object) -> DirectoryIdentity:
    """Build one directory identity, refusing incomplete platform metadata."""
    try:
        device = int(getattr(value, 'st_dev'))
        inode = int(getattr(value, 'st_ino'))
        mode = int(getattr(value, 'st_mode'))
    except (AttributeError, TypeError, ValueError) as error:
        raise DirectoryIdentityError('directory metadata unavailable') from error
    reparse = _reparse_from_stat(value)
    return DirectoryIdentity(device=device, inode=inode, mode=mode, reparse=reparse)

def capture_directory_chain(root: Path) -> DirectoryChain:
    """Resolve ``root`` once and capture its root-first safe ancestor chain."""
    try:
        canonical_root = root.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise DirectoryIdentityError('canonical directory unavailable') from error
    identities: list[DirectoryIdentity] = []
    for ancestor in (canonical_root, *canonical_root.parents):
        try:
            metadata = os.lstat(ancestor)
        except OSError as error:
            raise DirectoryIdentityError('directory metadata unavailable') from error
        identity = directory_identity_from_stat(metadata)
        if not stat.S_ISDIR(identity.mode) or stat.S_ISLNK(identity.mode) or identity.reparse:
            raise DirectoryIdentityError('unsafe directory metadata')
        identities.append(identity)
    return DirectoryChain(canonical_root=canonical_root, identities=tuple(identities))

def _reparse_from_stat(value: object) -> bool:
    if not _WINDOWS:
        return False
    try:
        attributes = getattr(value, 'st_file_attributes')
        if attributes is None or _REPARSE_POINT is None:
            raise TypeError
        return bool(int(attributes) & int(_REPARSE_POINT))
    except (AttributeError, TypeError, ValueError) as error:
        raise DirectoryIdentityError('directory file attributes unavailable') from error


# ===========================================================================
# Section: tldw_chatbook.Utils.path_validation (extracted from the live module by the
# builder; regenerate rather than editing)
# ===========================================================================
"""
Path validation utilities to prevent directory traversal attacks.

This module provides functions to validate file paths and ensure they don't
escape allowed directories.
"""
import os
import stat
import time
from pathlib import Path
from typing import Optional, Sequence, Union

def log_counter(*args, **kwargs):
    try:
        raise ImportError("'..Metrics.metrics_logger' (importing log_counter) is not available inside the remote worker bundle")
    except ImportError:
        return None
    return emit(*args, **kwargs)

def log_histogram(*args, **kwargs):
    try:
        raise ImportError("'..Metrics.metrics_logger' (importing log_histogram) is not available inside the remote worker bundle")
    except ImportError:
        return None
    return emit(*args, **kwargs)

class _NoOpLogger:
    """Silent stand-in for loguru's logger where loguru is not installed.

    Only the methods called on ``logger`` in this module are defined. The
    remote worker bundle (Task 8) runs on a bare interpreter without
    loguru; validation diagnostics there are dropped rather than allowed
    to break ``validate_path``/``validate_path_simple``.
    """
    __slots__ = ()

    def debug(self, *_args, **_kwargs) -> None:
        return None

    def info(self, *_args, **_kwargs) -> None:
        return None

    def warning(self, *_args, **_kwargs) -> None:
        return None

    def error(self, *_args, **_kwargs) -> None:
        return None

def _logging_logger():
    """Return loguru's logger, or a silent no-op when loguru is absent.

    Same laziness rule as ``log_counter`` above: the import happens per
    call, never at module scope, and the ImportError fallback exists for
    the stdlib-only remote worker bundle (Phase 1d). In the parent, where
    loguru is an installed dependency, the return value is unchanged.
    """
    try:
        raise ImportError("'loguru' (importing logger) is not available inside the remote worker bundle")
    except ImportError:
        return _NoOpLogger()
    return logger

def validate_recovery_relative_path(value: str) -> str:
    """Validate one bounded portable tree name; empty string names its root."""
    if type(value) is not str or len(value.encode('utf-8')) > 1024:
        raise ValueError('invalid_recovery_relative_path')
    if not value:
        return value
    if any((part in {'', '.', '..'} for part in value.split('/'))) or any((ord(char) < 32 or char in '\\:' for char in value)):
        raise ValueError('invalid_recovery_relative_path')
    return value
ROOT_DENIAL_RECOVERY_POINTER = 'Need that folder? Add it to a named Workspace.'
ROOT_DENIAL_RECOVERY_HINT = 'Chats do not need a folder. To give local file tools access outside private scratch, bind that folder in Settings > Workspaces and use a chat in that Workspace.'

def validate_browsing_path(value: str | Path) -> Path:
    """Validate a local picker path without restricting interactive navigation.

    Args:
        value: Absolute path supplied by the picker navigation boundary.

    Returns:
        The lexical path, preserving parent segments and symlink aliases.
        Existence and permissions are checked by the subsequent filesystem call.

    Raises:
        ValueError: The value has an invalid type, contains NUL, or is relative.
    """
    if not isinstance(value, (str, Path)) or '\x00' in str(value):
        raise ValueError('Invalid browsing path')
    path = value if isinstance(value, Path) else Path(value)
    if not path.is_absolute():
        raise ValueError('Browsing path must be absolute')
    return path

def validate_canonical_directory(value: os.PathLike[str] | str) -> Path:
    """Validate an existing absolute directory without normalizing aliases.

    Args:
        value: Directory spelling that must already be canonical, including
            every ancestor. Hidden profile directories are permitted.

    Returns:
        The validated canonical directory. Callers must still pin descriptors
        and enforce their own authority and containment during filesystem use.

    Raises:
        ValueError: If the value is not an existing canonical directory. The
            error contains no path content.
    """
    try:
        raw = os.fspath(value)
        if type(raw) is not str or not raw or '\x00' in raw:
            raise ValueError
        path = Path(raw)
        if not path.is_absolute() or str(path) != raw:
            raise ValueError
        if path.resolve(strict=True) != path:
            raise ValueError
        if not stat.S_ISDIR(os.lstat(path).st_mode):
            raise ValueError
        return path
    except (OSError, TypeError, ValueError, RuntimeError):
        raise ValueError('Canonical directory required') from None

def validate_path(user_path: Union[str, Path], base_directory: Union[str, Path], *, redact_paths: bool=False, allow_hidden: bool=False) -> Path:
    """
    Validates that a user-provided path is within the allowed base directory.

    Args:
        user_path: The path provided by the user
        base_directory: The allowed base directory
        redact_paths: Log only bounded failure categories when the path is
            privacy-sensitive.
        allow_hidden: If True, permit hidden path components (e.g. ``.github/``)
            as long as the path stays within the base directory. Defaults to
            False, preserving the original behavior of rejecting hidden files
            and directories.

    Returns:
        Path: The validated absolute path

    Raises:
        ValueError: If the path is invalid or attempts directory traversal
    """
    logger = _logging_logger()
    start_time = time.time()
    log_counter('path_validation_validate_path_attempt')
    redacted_failure: str | None = None
    try:
        user_path = Path(user_path)
        base_directory = Path(base_directory).resolve()
        if user_path.is_absolute():
            full_path = user_path.resolve()
        else:
            full_path = (base_directory / user_path).resolve()
        try:
            full_path.relative_to(base_directory)
        except ValueError:
            if redact_paths:
                logger.warning('Path traversal attempt detected.')
            else:
                logger.warning(f'Path traversal attempt detected: {user_path} -> {full_path}')
            log_counter('path_validation_security_violation', labels={'type': 'directory_traversal'})
            if redact_paths:
                redacted_failure = 'Path is outside the allowed directory'
                raise ValueError(redacted_failure)
            raise ValueError(f"Path '{user_path}' is outside the allowed directory")
        relative_parts = full_path.relative_to(base_directory).parts
        if not allow_hidden and any((part.startswith('.') for part in relative_parts if part != '.')):
            if redact_paths:
                logger.warning('Hidden file/directory access attempt detected.')
            else:
                logger.warning(f'Hidden file/directory access attempt: {full_path}')
            log_counter('path_validation_security_violation', labels={'type': 'hidden_file_access'})
            if redact_paths:
                redacted_failure = 'Access to hidden files/directories is not allowed'
            raise ValueError('Access to hidden files/directories is not allowed')
        if not allow_hidden and base_directory.name.startswith('.'):
            if redact_paths:
                logger.warning('Hidden base directory rejected.')
            else:
                logger.warning(f'Hidden base directory rejected: {base_directory}')
            log_counter('path_validation_security_violation', labels={'type': 'hidden_file_access'})
            if redact_paths:
                redacted_failure = 'Access to hidden files/directories is not allowed'
            raise ValueError('Access to hidden files/directories is not allowed')
        duration = time.time() - start_time
        log_histogram('path_validation_validate_path_duration', duration, labels={'status': 'success'})
        log_counter('path_validation_validate_path_success')
        return full_path
    except Exception as e:
        duration = time.time() - start_time
        log_histogram('path_validation_validate_path_duration', duration, labels={'status': 'error'})
        log_counter('path_validation_validate_path_error', labels={'error_type': type(e).__name__})
        if redact_paths:
            logger.error('Path validation failed (category={}).', type(e).__name__)
        else:
            logger.error(f"Path validation error for '{user_path}': {e}")
        if redact_paths:
            raise ValueError(redacted_failure or 'Invalid path') from None
        if isinstance(e, ValueError):
            raise
        raise ValueError(f'Invalid path: {user_path}')

def validate_filename(filename: str) -> str:
    """
    Validates a filename to ensure it doesn't contain path separators or other dangerous characters.

    Args:
        filename: The filename to validate

    Returns:
        str: The validated filename

    Raises:
        ValueError: If the filename is invalid
    """
    start_time = time.time()
    log_counter('path_validation_validate_filename_attempt')
    if not filename:
        log_counter('path_validation_validate_filename_error', labels={'error_type': 'empty_filename'})
        raise ValueError('Filename cannot be empty')
    if os.path.sep in filename or '/' in filename or '\\' in filename:
        log_counter('path_validation_security_violation', labels={'type': 'path_separator_in_filename'})
        raise ValueError('Filename cannot contain path separators')
    if '..' in filename:
        log_counter('path_validation_security_violation', labels={'type': 'parent_directory_reference'})
        raise ValueError('Filename cannot contain parent directory references')
    if '\x00' in filename:
        log_counter('path_validation_security_violation', labels={'type': 'null_byte_in_filename'})
        raise ValueError('Filename cannot contain null bytes')
    reserved_names = {'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'}
    name_without_ext = filename.split('.')[0].upper()
    if name_without_ext in reserved_names:
        log_counter('path_validation_security_violation', labels={'type': 'reserved_filename'})
        raise ValueError(f"'{filename}' is a reserved filename")
    duration = time.time() - start_time
    log_histogram('path_validation_validate_filename_duration', duration)
    log_counter('path_validation_validate_filename_success')
    return filename

def safe_join_path(base_directory: Union[str, Path], *paths: Union[str, Path]) -> Path:
    """
    Safely joins paths ensuring the result stays within the base directory.

    Args:
        base_directory: The base directory
        *paths: Path components to join

    Returns:
        Path: The safely joined path

    Raises:
        ValueError: If the resulting path would escape the base directory
    """
    base = Path(base_directory).resolve()
    result = base
    for path_component in paths:
        if isinstance(path_component, str):
            validate_filename(path_component)
        result = result / path_component
    return validate_path(result, base)

def is_safe_path(user_path: Union[str, Path], base_directory: Union[str, Path]) -> bool:
    """
    Checks if a path is safe without raising exceptions.

    Args:
        user_path: The path to check
        base_directory: The allowed base directory

    Returns:
        bool: True if the path is safe, False otherwise
    """
    try:
        validate_path(user_path, base_directory)
        return True
    except ValueError:
        return False

def get_safe_relative_path(full_path: Union[str, Path], base_directory: Union[str, Path]) -> Optional[Path]:
    """
    Gets the relative path from base_directory, or None if the path is unsafe.

    Args:
        full_path: The full path
        base_directory: The base directory

    Returns:
        Optional[Path]: The relative path, or None if unsafe
    """
    try:
        full_path = Path(full_path).resolve()
        base_directory = Path(base_directory).resolve()
        return full_path.relative_to(base_directory)
    except ValueError:
        return None

def validate_path_simple(user_path: Union[str, Path], require_exists: bool=False, *, probe_existing: bool=True) -> Path:
    """
    Simple path validation that checks for common security issues without requiring a base directory.

    Args:
        user_path: The path to validate
        require_exists: Whether to require the path exists
        probe_existing: Whether to inspect and resolve an existing selected path.
            Disable this when a later no-follow boundary owns link validation.

    Returns:
        Path: The validated path

    Raises:
        ValueError: If the path contains security risks
    """
    logger = _logging_logger()
    start_time = time.time()
    log_counter('path_validation_validate_path_simple_attempt')
    try:
        path_str = str(user_path)
        if '\x00' in path_str:
            log_counter('path_validation_security_violation', labels={'type': 'null_byte'})
            raise ValueError('Path cannot contain null bytes')
        dangerous_patterns = ['../..', '..\\..', '~/', '~\\', '\x00', '|', ';', '&&', '||', '`', '$(', '${']
        for pattern in dangerous_patterns:
            if pattern in path_str:
                log_counter('path_validation_security_violation', labels={'type': 'dangerous_pattern', 'pattern': pattern})
                raise ValueError(f'Path contains dangerous pattern: {pattern}')
        path = Path(user_path)
        if probe_existing:
            if path.exists():
                resolved = path.resolve()
                if path.is_absolute() and resolved != path:
                    logger.warning('Path resolution changed during validation')
            elif require_exists:
                raise ValueError('Path does not exist')
        elif require_exists:
            raise ValueError('require_exists requires probe_existing=True')
        duration = time.time() - start_time
        log_histogram('path_validation_validate_path_simple_duration', duration)
        log_counter('path_validation_validate_path_simple_success')
        return path
    except Exception as e:
        duration = time.time() - start_time
        log_histogram('path_validation_validate_path_simple_duration', duration, labels={'status': 'error'})
        log_counter('path_validation_validate_path_simple_error', labels={'error_type': type(e).__name__})
        if isinstance(e, ValueError):
            raise
        raise ValueError(f'Invalid path: {user_path}')

def validate_existing_absolute_directory(user_path: Union[str, Path]) -> Path:
    """Return one normalized existing directory without imposing a root.

    This is the central mode for explicitly host-authorized process working
    directories. It validates and normalizes the path, but intentionally does
    not claim workspace confinement.
    """
    try:
        path = Path(user_path)
        path_text = str(path)
        path_text.encode('utf-8')
        if '\x00' in path_text or not path.is_absolute():
            raise ValueError
        normalized = path.resolve(strict=True)
        if not normalized.is_dir():
            raise ValueError
        return normalized
    except (OSError, TypeError, ValueError, UnicodeError):
        raise ValueError('Path must be an absolute existing directory') from None

def validate_path_multi(user_path: Union[str, Path], roots: Sequence[Union[str, Path]]) -> Path:
    """Validate ``user_path`` against several allowed roots (first match wins).

    Relative paths resolve against ``roots[0]`` (the primary root — callers
    pass the tool sandbox first so legacy relative-path behavior is
    unchanged). The rejection message names every consulted root so a
    denial is actionable.

    Every attempt is made with ``redact_paths=True`` (TASK-19558). This is
    the choke point for the agent file-tool family (``ReadFileTool`` /
    ``WriteFileTool`` / ``ListDirectoryTool`` / ``EditFileTool``, via
    ``Tools/file_operation_tools.py``), so ``user_path`` here is
    MODEL-supplied and prompt-injection-reachable; without redaction a
    single traversal probe wrote attacker-chosen text AND the user's real
    directory layout into the log once PER ROOT. The refusal below is
    unaffected -- it is built here, from ``user_path``, and still names the
    path and every consulted root, because that message goes to the model
    as a recovery route rather than into diagnostics.

    Args:
        user_path: The path provided by the user or model.
        roots: Allowed base directories, in priority order.

    Returns:
        The validated absolute path.

    Raises:
        ValueError: No roots given, or the path escapes all of them.
    """
    root_list = [Path(root) for root in roots]
    if not root_list:
        raise ValueError('No allowed roots configured for path validation.')
    candidate = Path(user_path)
    for index, root in enumerate(root_list):
        if index > 0 and (not candidate.is_absolute()):
            continue
        try:
            return validate_path(user_path, root, redact_paths=True)
        except ValueError:
            continue
    consulted = ', '.join((str(root.resolve()) for root in root_list))
    raise ValueError(f"Outside every allowed root. {ROOT_DENIAL_RECOVERY_POINTER} Path: '{user_path}'. {ROOT_DENIAL_RECOVERY_HINT} (Checked: {consulted})")


# ===========================================================================
# Section: tldw_chatbook.Utils.sensitive_paths (extracted from the live module by the
# builder; regenerate rather than editing)
# ===========================================================================
"""Paths refused by the agent-facing file tools, regardless of configured root.

TWO independent agent file-tool families enforce this, and both must:

1. ``Tools/file_operation_tools.py``'s ``ReadFileTool``, ``WriteFileTool``,
   ``ListDirectoryTool``, ``GlobFiles`` and ``GrepFiles`` (sandbox-root
   confined) -- each calls :func:`is_sensitive_path` (directly, or through
   that module's ``is_within`` helper) on every candidate path immediately
   after path validation and before touching the filesystem.
2. The workspace-local ``fs_*``/``git_*`` family --
   ``Tools/local_tool_impls.py``'s ``resolve_workspace_path``, the single
   choke point through which every one of them resolves a path ARGUMENT
   (``Tools/patch_tool_impls.py``'s ``patch_files`` directly; the four
   repo-scoped tools in ``Tools/git_tool_impls.py`` one hop away, via
   ``prepare_repository``/``_prepare_for_path``/``_repo_relative_path``).

   Read what that sentence does and does not say. It covers a path the
   model NAMES. It does not make a tool's OUTPUT safe:

   * The three enumerating ``fs_*`` tools (``list_directory``/
     ``glob_files``/``grep_files``) present entries the model never named,
     so each additionally filters its own walked candidates against
     :func:`is_sensitive_path` -- the choke point only ever sees their
     root.
   * The ``git_*`` tools present entries the model never named too --
     ``path`` is OPTIONAL on ``git_status``/``git_log``/``git_diff``, and
     with it omitted nothing but the repository root reaches this module
     (``Agents/local_tool_provider.py``'s ``path_targets`` returns the
     repo root and stops). They do not filter git's OUTPUT; they
     constrain its INPUT, translating this module's denials into
     ``:(exclude)`` PATHSPECS computed per call from
     :func:`sensitive_exclusions_under` (see ``Tools/git_tool_impls.py``'s
     ``_denylist_pathspecs``). git stays the authority on what matches,
     and no diff or porcelain text is ever parsed to decide what to
     withhold -- a half-parsed diff is worse than none. TASK-19632, which
     that closed: on a ``$HOME``-rooted workspace containing
     ``~/.ssh/id_rsa``, ``git_diff`` with no ``path`` returned the file's
     CONTENT from a CLEAN worktree (``commit_range="HEAD~1..HEAD"`` reads
     it out of history -- no write primitive and no dirty tree needed),
     and ``git_diff(stat=True)`` / ``git_status`` returned its NAME.
     ``git_log`` leaked nothing (commit metadata only -- no paths, no
     content) and is deliberately left unfiltered, so its output is
     unchanged by any of this.

   One consequence of using pathspecs belongs here, where the denials are
   defined: **a pathspec is not a path**. A repository file whose NAME is
   itself pathspec magic (``:(exclude)notes.txt`` is a legal POSIX
   filename) inverts the scope of whatever argv it is spliced into --
   ``git_diff(path=":(exclude)notes.txt")`` was measured returning the
   whole-repo diff, ``~/.ssh/id_rsa``'s content included, while nominally
   scoping to that one file, and it did so THROUGH the choke point, since
   the string is a real filename that resolves inside the workspace. The
   ``--`` separator does not stop it: ``--`` ends OPTIONS, and everything
   after it is parsed as a pathspec, magic and all. Every pathspec those
   tools build is therefore rendered with explicit ``:(literal)`` /
   ``:(exclude,literal)`` magic, the model-supplied ``path`` included.

   This family was added AFTER this contract was written and, until
   TASK-19551, called none of this: it confined paths to ``[console]
   workspace_root`` and stopped there. With the shipped default root (the
   app's cwd at startup) an app launched from ``$HOME`` made ``$HOME`` the
   confinement root, so ``fs_read`` returned ``~/.ssh/id_rsa`` and
   ``fs_write``/``fs_patch`` could rewrite ``mcp_permissions.json`` -- the
   one-step gate bypass described below -- reachable by prompt injection
   from fetched web content. The failure was not a wrong check; it was an
   enforcer list that could only ever name the implementations existing
   when it was written, and a second family that never joined it. Hence
   the exception above is stated rather than smoothed over: **a new
   agent-facing file tool joins one of these two families; it does not get
   a third path-resolution seam** --
   ``Tests/Tools/test_local_tool_sensitive_paths.py`` pins that
   structurally (an AST tripwire over all three ``fs_*``/``git_*`` core
   modules) and pins the two families' agreement on the denylist, so they
   cannot drift apart again.

Two instruments express what is denied, and the choice between them is
made per case rather than by taste (TASK-19633):

* **Location rules** -- ``_SENSITIVE_DIRS`` plus the accessor-resolved
  paths further down -- for anything whose LOCATION is the unambiguous
  part. ``~/.ssh``, ``~/.aws``, ``~/.config/gh``: everything under them is
  credential material, and their filenames are NOT self-identifying
  (``hosts.yml`` is just as often an Ansible inventory).
* **A name rule** (``_SENSITIVE_FILE_NAMES``) for the handful of
  filenames that identify a credential store wherever they appear.
  Adopted after ``~/.netrc``, ``~/.git-credentials``, ``~/.npmrc``,
  ``~/.pypirc``, ``~/.cargo/credentials.toml`` and
  ``~/.config/gh/hosts.yml`` were each measured returning their body
  through ``fs_read``. A location rule cannot cover most of that set: the
  credential is not confined to one directory (a project-local
  ``.npmrc``/``.pypirc`` carries an auth token exactly like the home one,
  and a copy of ``credentials.toml`` is a credential wherever it lands),
  and refusing the whole of ``~/.cargo`` or ``~/.config/git`` would take
  down things an agent legitimately reads.

BOTH are enumerations and both trail reality -- names no less than
locations. The name rule is preferred where it applies because one entry
covers unbounded locations while one location entry covers exactly one,
and because a tool's config DIRECTORY migrates between XDG/legacy/OS
conventions far more often than its credential FILENAME ever changes. It
is kept deliberately small and biased toward names that are credential
stores by definition; a name is added only when a false refusal would be
a curiosity rather than routine obstruction -- which is why ``.env`` is
deliberately NOT here (as often build configuration as secrets, and
refusing it would break the ADR-032 coding-agent use case this module
must not break). The cost is real and accepted: an agent cannot read a
test fixture named ``credentials``, and the refusal names it as protected
rather than failing silently.

The two families' HIDDEN-COMPONENT policies still differ, and that
difference is design, not residue. Family 1 confines through
``validate_path_multi``, which defaults ``allow_hidden=False`` and so
refuses any dotted component before this module is consulted; family 2
passes ``allow_hidden=True`` (ADR-032 -- a coding agent that cannot read
``.github/`` or ``.gitignore`` is useless). Family 1 is therefore
strictly stricter for dotted NAMES, which is acceptable because the two
roots are different kinds of place: family 1's sandbox root is app-owned
storage where a dotfile has no legitimate purpose, while family 2's root
is a user source tree where dotfiles are the point. What must not differ
is THIS module's answer -- and since TASK-19633 it does not: every
credential path measured above is refused by the denylist itself, under
either family, so the part of the gap that was weaker-by-accident is
gone and only the deliberate difference remains.

This is *not* wired into
``Utils/path_validation.validate_path``/``validate_path_multi`` themselves:
those helpers are the app's general-purpose validators, used by ~40
first-party call sites (config screens, DB path resolution, exports, ...) to
validate paths to this application's own config and database files -- which
are exactly the paths this module refuses. Baking the check in there would
block legitimate first-party access; it belongs at the agent-tool boundary
instead, shared by every file tool, so they cannot drift from each other.

Two distinct reasons a path lands here:

1. **Credentials.** ``read_file`` carries no elevated risk beyond ``reads``,
   so an unconfined read is a path from a private key into a persisted
   transcript that may be sent to any provider.
2. **This application's own gate state and data.** A tool able to rewrite
   ``mcp_permissions.json`` or ``config.toml`` can turn every ``ask`` into
   ``allow`` -- a one-step bypass of the permission system. A tool able to
   read or rewrite this app's own SQLite databases can exfiltrate or
   corrupt every conversation, note and credential-adjacent record they
   hold, bypassing the application layer entirely.

Every one of those is resolved through the app's OWN accessors at call
time, never a hardcoded literal: ``config.toml``'s location honors the
``TLDW_CONFIG_PATH`` override (``config._get_effective_config_path()``),
the MCP permission store and its companions live under
``config.get_user_data_dir()`` (never under the ``~/.config/tldw_cli/``
literal a first look at ``app.py`` might suggest -- see
``_sensitive_single_file_paths()``), and the SQLite DB paths honor
``[database]`` overrides and the active user folder (see
``_sensitive_db_paths()``). A literal here would drift the moment any of
those is overridden -- which is exactly how the permission-store literal
went stale (Finding 1) and how a ``TLDW_CONFIG_PATH`` override defeated the
``config.toml`` entry (Finding 3).

Every file this app creates directly under ``get_user_data_dir()`` is also
refused, as a RULE rather than an enumeration (see the direct-child-file
loop in ``is_sensitive_path``): new state files land there constantly
(agent-run logs, eval/RAG-indexing/search-history/event/kanban/sync-state
DBs, ...) without ever touching ``config.py``, so an accessor-name
enumeration permanently trails reality. The SAME rule is applied to three
more directories, for the same reason: the effective config directory
(``config._get_effective_config_path().parent``, which honors
``TLDW_CONFIG_PATH`` the same way the config file itself does -- it holds
``config.toml``'s own ``.bak``/``.tmp`` backup sidecars plus
``runtime_policy.json``/``ui_state.toml``, none of which is enumerated by
name here either); the ChromaDB vector-store persist directory
(``RAG_Search.simplified.config.default_chroma_persist_directory()``,
which holds ``chroma.sqlite3`` -- plaintext chunks of the same
conversations and notes ``ChaChaNotes.db`` protects); and the RAG-profile
store (``RAG_Search.config_profiles.default_rag_profiles_dir()``, plaintext
per-profile RAG/embedding-provider config). Existing DIRECTORIES nested
directly under any of these four are excluded from the rule and stay fully
reachable -- most importantly the default file-tool sandbox root,
``get_user_data_dir() / "tool_sandbox"``; see that check's own comment for
why a directory/file distinction, not a name, is what exempts them.

The skill trust/grant store gets a DIFFERENT treatment: the WHOLE
``get_user_data_dir() / "skills" / "trust"`` subtree is refused, not just
its direct children, because ``skills`` itself is one of the exempted
container directories above and everything nested under it would otherwise
inherit that exemption -- see ``_sensitive_skill_trust_dir`` for why that
one subtree needs an explicit carve-out.

A directory can also be CREATED to collide with a not-yet-existing state
file at one of these locations (e.g. an agent asking ``write_file`` to
create parent directories for ``search_history.db/note.txt`` before this
app has ever created ``search_history.db`` as a file) -- the app's later
attempt to open its own state file then fails outright, a denial of
service. ``refuses_new_directory_chain`` is the guard against that: callers
that create directories on the agent's behalf (``WriteFileTool``'s
``create_directories=True`` path) must consult it before calling
``Path.mkdir(parents=True, ...)``.

This is a guardrail, not a security boundary: it stops accidents and naive
injected payloads, not a determined ``python -c``. The sandbox/workspace-root
track is the real answer for shell execution.
"""
from pathlib import Path
from typing import Iterable, Literal, NamedTuple

def _debug(message: str) -> None:
    """Emit one resolution-failure diagnostic.

    Loguru is imported here rather than at module scope: this module sits
    inside the pinned workspace worker's stdlib-only import closure
    (Phase 0c), so merely importing it must not pull loguru. Every call
    site is a rare resolution-failure path, so the import cost lands only
    when a diagnostic is actually emitted — the same laziness rule the
    ``config``/Skills/RAG imports throughout this module already follow.
    """
    raise ImportError("'loguru' (importing logger) is not available inside the remote worker bundle")
    logger.debug(message)
_SENSITIVE_DIRS = ('~/.ssh', '~/.aws', '~/.gnupg', '~/.config/gcloud', '~/.docker', '~/.kube', '~/.local/share/keyrings', '~/.config/gh')
_SENSITIVE_FILE_NAMES = frozenset((name.casefold() for name in ('.netrc', '_netrc', '.git-credentials', '.npmrc', '.pypirc', 'credentials', 'credentials.toml')))
_DB_PATH_ACCESSOR_NAMES = ('get_chachanotes_db_path', 'get_prompts_db_path', 'get_media_db_path', 'get_library_collections_db_path', 'get_library_ingest_jobs_db_path', 'get_workspaces_db_path', 'get_subscriptions_db_path', 'get_notifications_db_path', 'get_research_db_path', 'get_writing_db_path', 'get_scheduled_tasks_db_path', 'get_evals_db_path', 'get_rag_indexing_db_path')
_DB_SIDECAR_SUFFIXES = ('-wal', '-shm', '-journal')

def _resolved(path_str: str) -> Path | None:
    """Resolve a path string, returning ``None`` on ANY resolution failure.

    ``is_sensitive_path``'s fail-closed guarantee ("a path that cannot be
    resolved is treated as sensitive") depends on this returning ``None``
    for every way resolution can fail, not just the two most common ones.
    ``Path.resolve()``/``expanduser()`` normally raise ``OSError`` (e.g. a
    symlink loop) or ``RuntimeError`` (older Pythons' own loop-detection),
    but a path containing an embedded NUL byte raises ``ValueError``
    instead -- narrowing this catch to ``(OSError, RuntimeError)`` let that
    case escape ``is_sensitive_path`` entirely as an uncaught exception
    rather than the promised ``True`` (TASK-847). Broad by design: whatever
    exception ``pathlib`` raises for a candidate this function cannot make
    sense of, the caller must still get ``None`` back, never a propagated
    error.
    """
    try:
        return Path(path_str).expanduser().resolve()
    except Exception as exc:
        _debug(f'sensitive_paths: could not resolve {path_str!r}: {exc}')
        return None

def _sensitive_db_paths() -> tuple[Path, ...]:
    """Resolve this app's own SQLite database paths, lazily.

    These databases live under ``config.get_user_data_dir()`` -- by default
    a sibling of ``~/.config/tldw_cli`` (e.g. ``~/.local/share/tldw_cli/...``),
    not beneath it, so the static ``_SENSITIVE_DIRS`` tuple above cannot
    express their location. Each path is resolved via the app's own
    accessor (which also honors ``[database]`` path overrides and the
    active user folder) rather than hardcoded, since neither the user
    folder nor an override is known statically.

    Returns:
        Resolved paths to every database whose accessor could be called.
        An accessor that raises is skipped rather than failing the whole
        check -- it is additional coverage, not the primary guarantee.
    """
    raise ImportError("'..' (importing config) is not available inside the remote worker bundle")
    resolved: list[Path] = []
    for accessor_name in _DB_PATH_ACCESSOR_NAMES:
        accessor = getattr(_config, accessor_name, None)
        if accessor is None:
            continue
        try:
            resolved.append(accessor())
        except Exception as exc:
            _debug(f'sensitive_paths: could not resolve {accessor_name}: {exc}')
    return tuple(resolved)

def _sensitive_single_file_paths() -> tuple[Path, ...]:
    """Resolve this app's own non-DB sensitive single files, lazily.

    Two families, each resolved through the same accessor the app itself
    uses to build the real path -- never a literal -- because both can move
    at runtime:

    1. **config.toml.** ``config._get_effective_config_path()`` honors the
       ``TLDW_CONFIG_PATH`` override (set throughout this project's own
       test suite, and by any deployment that relocates the config file).
       A literal default-path check misses the file actually holding the
       user's API keys whenever that override is set (Finding 3).
    2. **The MCP permission store and its companions.** The store's real
       path is ``get_user_data_dir() / "mcp_permissions.json"`` -- built by
       ``MCP.unified_control_plane_service``'s ``permission_store`` property
       as ``Path(store.path).with_name("mcp_permissions.json")``, where
       ``store.path`` is the ``LocalMCPStore`` path ``app.py`` constructs as
       ``get_user_data_dir() / "local_mcp_store.json"``. A tool able to
       rewrite this file can turn every ``ask`` into ``allow`` -- the
       CRITICAL one-step permission-gate bypass this module exists to
       prevent (Finding 1; see the module docstring). Two companions built
       the exact same ``Path(...).with_name(...)`` way from that same base
       path carry the same class of gate-relevant state:
       ``local_mcp_store.json`` itself (server definitions and their env)
       and ``mcp_execution_log.jsonl`` (the execution audit trail).

    Returns:
        Resolved paths for every file above whose accessor could be
        called. An accessor that raises is skipped rather than failing the
        whole check -- additional coverage, not the primary guarantee (see
        ``_sensitive_db_paths``, which does the same for the DB paths).
    """
    raise ImportError("'..' (importing config) is not available inside the remote worker bundle")
    resolved: list[Path] = []
    try:
        resolved.append(_config._get_effective_config_path())
    except Exception as exc:
        _debug(f'sensitive_paths: could not resolve config.toml path: {exc}')
    try:
        user_data_dir = _config.get_user_data_dir()
    except Exception as exc:
        _debug(f'sensitive_paths: could not resolve user data dir: {exc}')
    else:
        resolved.append(user_data_dir / 'mcp_permissions.json')
        resolved.append(user_data_dir / 'local_mcp_store.json')
        resolved.append(user_data_dir / 'mcp_execution_log.jsonl')
    return tuple(resolved)

def _sensitive_skill_trust_dir() -> Path | None:
    """Resolve this app's skill trust/grant store directory, lazily.

    ``get_user_data_dir() / "skills"`` is one of the existing-directory
    exemptions the direct-child-file rule (applied in ``is_sensitive_path``)
    carves out -- every trusted skill bundle lives as a named subdirectory
    under it, so a file inside it is deliberately NOT covered by that rule,
    letting agent tools browse/read a user's own skill bundles.

    The ``trust`` subdirectory nested one level inside it is the ONE
    exception carved back OUT of that exemption: it holds
    ``skill_trust_manifest.json`` (the authenticated trust manifest),
    ``skill_script_grants.json`` (the plain, UNAUTHENTICATED JSON file
    ``SkillTrustService.has_script_grant`` consults to authorize script
    EXECUTION -- deliberately kept outside the manifest's own HMAC+keyring
    integrity check; see ``Skills_Interop/skill_trust_service.py``),
    ``generation_marker.json`` (the local rollback-protection marker), and
    ``snapshots/`` (encrypted trusted-skill snapshots). A tool able to
    rewrite the grants file can authorize its own future script execution
    -- the same class of one-step gate bypass the MCP permission store's
    entry exists to prevent (see this module's docstring) -- so this
    caller refuses the WHOLE subtree by ancestry (the same way
    ``_SENSITIVE_DIRS`` is matched), not just its direct children: a file
    several levels inside ``snapshots/`` must be refused exactly like the
    manifest itself.

    Resolved via ``Skills_Interop.local_skills_service.default_local_skills_store_dir``
    and ``Skills_Interop.skill_trust_store.default_trust_store_dir`` -- the
    SAME functions ``app.py`` calls to build the live ``SkillTrustStore`` --
    never a re-spelled ``"skills"``/``"trust"`` literal, which would drift
    the moment either name changed (see this module's docstring for why
    that class of drift is exactly how a past finding went stale).

    Returns:
        The trust store directory, or ``None`` if ``get_user_data_dir()``
        could not be resolved.
    """
    raise ImportError("'..' (importing config) is not available inside the remote worker bundle")
    raise ImportError("'..Skills_Interop.local_skills_service' (importing default_local_skills_store_dir) is not available inside the remote worker bundle")
    raise ImportError("'..Skills_Interop.skill_trust_store' (importing default_trust_store_dir) is not available inside the remote worker bundle")
    try:
        user_data_dir = _config.get_user_data_dir()
    except Exception as exc:
        _debug(f'sensitive_paths: could not resolve user data dir: {exc}')
        return None
    local_skills_store_dir = default_local_skills_store_dir(user_data_dir)
    return default_trust_store_dir(local_skills_store_dir)

def _direct_child_rule_container_dirs() -> tuple[Path, ...]:
    """Resolve every directory whose direct (non-recursive) child FILES are refused, lazily.

    Each of these is a directory this app treats as a bounded container for
    its own state, where new files land constantly without ever being
    named here individually -- this is the set the "Finding 2" rule in
    ``is_sensitive_path`` applies to. Existing DIRECTORIES nested directly
    inside any one of them (``tool_sandbox``, ``chat_dicts``, ``chromadb``,
    ``exports``, ``rag_profiles``, ``skills``, and any future sibling) are
    exempt from the rule and stay fully reachable; only a same-level FILE
    is refused. See that rule's own comment for why "is an existing
    directory", not a name, is what exempts them.

    Returns:
        Every container directory whose accessor could be resolved:

        * ``config.get_user_data_dir()``.
        * The effective config directory
          (``config._get_effective_config_path().parent``) -- honors
          ``TLDW_CONFIG_PATH`` the same way the config file itself does.
          This is what covers ``config.toml``'s own ``.bak``/``.tmp``
          backup sidecars (``UI/Screens/settings_screen.py``'s Advanced
          config save writes both, byte-identical to the live config,
          API keys included) and any other loose file dropped beside it
          (``runtime_policy.json``, ``ui_state.toml``, a hand-made backup
          copy under any other name) -- none of which is enumerated here
          by name either, for the same reason the user-data-dir rule
          isn't: an enumeration permanently trails whatever gets written
          there next.
        * The ChromaDB vector-store persist directory
          (``RAG_Search.simplified.config.default_chroma_persist_directory()``),
          which holds ``chroma.sqlite3`` -- plaintext chunks of the same
          conversations and notes ``ChaChaNotes.db`` protects.
        * The RAG-profile store directory
          (``RAG_Search.config_profiles.default_rag_profiles_dir()``),
          plaintext per-profile RAG/embedding-provider config.

        An accessor that raises is skipped rather than failing the whole
        check, as elsewhere in this module.
    """
    raise ImportError("'..' (importing config) is not available inside the remote worker bundle")
    raise ImportError("'..RAG_Search.config_profiles' (importing default_rag_profiles_dir) is not available inside the remote worker bundle")
    raise ImportError("'..RAG_Search.simplified.config' (importing default_chroma_persist_directory) is not available inside the remote worker bundle")
    resolved: list[Path] = []
    try:
        resolved.append(_config.get_user_data_dir())
    except Exception as exc:
        _debug(f'sensitive_paths: could not resolve user data dir: {exc}')
    try:
        resolved.append(_config._get_effective_config_path().parent)
    except Exception as exc:
        _debug(f'sensitive_paths: could not resolve effective config dir: {exc}')
    try:
        resolved.append(default_chroma_persist_directory())
    except Exception as exc:
        _debug(f'sensitive_paths: could not resolve chroma persist dir: {exc}')
    try:
        resolved.append(default_rag_profiles_dir())
    except Exception as exc:
        _debug(f'sensitive_paths: could not resolve rag profiles dir: {exc}')
    return tuple(resolved)

def _db_sidecar_paths(db_path: Path) -> tuple[Path, ...]:
    """Build the WAL/SHM/rollback-journal sidecar paths for one DB path.

    Args:
        db_path: A resolved path to one of this app's SQLite databases, as
            returned by ``_sensitive_db_paths()``.

    Returns:
        One path per entry in ``_DB_SIDECAR_SUFFIXES``, each formed by
        appending the suffix to ``db_path``'s own filename -- e.g.
        ``chachanotes.db`` -> ``chachanotes.db-wal``. Built from an explicit
        name construction, not a prefix, so callers must compare by exact
        equality: appending is not the same as matching anything that
        merely starts with the DB's name.
    """
    return tuple((db_path.with_name(db_path.name + suffix) for suffix in _DB_SIDECAR_SUFFIXES))

class SensitivePathContext(NamedTuple):
    """A snapshot of the resolved sensitive-path set, valid for one tool call.

    Building one of these costs the same 11 config-accessor resolutions
    ``is_sensitive_path`` would otherwise repeat on every invocation. A
    caller that tests many candidate paths within a single tool invocation
    (``GlobFiles``/``GrepFiles`` and ``ListDirectoryTool``'s recursive walk,
    all in ``Tools/file_operation_tools.py``) should build exactly ONE of
    these at the start of that invocation and pass it into every
    ``is_sensitive_path``/``is_within`` call it makes, rather than let each
    call re-resolve the set from scratch.

    Deliberately not cached at module or process scope -- see
    ``resolve_sensitive_context``.
    """
    files: tuple[Path, ...]
    dirs: tuple[Path, ...]
    db_paths: tuple[Path, ...]
    user_data_dir: Path | None
    direct_child_denied_dirs: tuple[Path, ...]

def resolve_sensitive_context() -> SensitivePathContext:
    """Resolve the full sensitive-path set once, for reuse across many checks.

    Call this ONCE per tool invocation and thread the result through to
    every ``is_sensitive_path``/``is_within`` call that invocation makes.
    Do NOT cache the return value at module or process scope: the whole
    point of the per-call ``_sensitive_db_paths()`` resolution it wraps is
    to observe a config change (e.g. the test suite swapping
    ``TLDW_CONFIG_PATH`` between cases) on the very next call rather than
    serving a stale answer. A single invocation resolving this once is
    "per call"; a global cache would not be.

    Returns:
        A ``SensitivePathContext`` snapshotting the currently configured
        sensitive files, directories, database paths, and user data
        directory (entries that failed to resolve are dropped; the user
        data directory is ``None`` if it could not be resolved).
    """
    raise ImportError("'..' (importing config) is not available inside the remote worker bundle")
    try:
        user_data_dir = _resolved(str(_config.get_user_data_dir()))
    except Exception as exc:
        _debug(f'sensitive_paths: could not resolve user data dir: {exc}')
        user_data_dir = None
    skill_trust_dir = _sensitive_skill_trust_dir()
    dynamic_dirs = (skill_trust_dir,) if skill_trust_dir is not None else ()
    return SensitivePathContext(files=tuple((p for p in (_resolved(str(raw)) for raw in _sensitive_single_file_paths()) if p is not None)), dirs=tuple((p for p in (_resolved(str(entry)) for entry in _SENSITIVE_DIRS + dynamic_dirs) if p is not None)), db_paths=tuple((p for p in (_resolved(str(raw)) for raw in _sensitive_db_paths()) if p is not None)), user_data_dir=user_data_dir, direct_child_denied_dirs=tuple((p for p in (_resolved(str(raw)) for raw in _direct_child_rule_container_dirs()) if p is not None)))

def merge_sensitive_context(base: SensitivePathContext, *, extra_files: Iterable[Path]=(), extra_dirs: Iterable[Path]=()) -> SensitivePathContext:
    """Fold per-workspace user exclusions into a per-call context snapshot.

    The workspace-exclusions injection point (spec 2026-09-20): extras join
    ``files``/``dirs`` resolved and deduped by the denylist's own
    ``_compare_key`` discipline, so ``is_sensitive_path``,
    ``sensitive_exclusions_under``, and ``refuses_new_directory_chain``
    enforce them with no further call-site changes. Extras that fail
    resolution are dropped, mirroring how the base set's own unresolved
    entries are dropped.

    Args:
        base: The call's base sensitive-path context (denylist plus any
            earlier folds).
        extra_files: Additional absolute file paths to protect (e.g.
            per-binding user exclusions of file kind).
        extra_dirs: Additional absolute directory paths to protect; a
            directory entry also shields its descendants.

    Returns:
        A new ``SensitivePathContext`` with the extras merged in; ``base``
        is never mutated. Unresolvable extras are silently omitted.
    """
    files = list(base.files)
    dirs = list(base.dirs)
    seen_files = {_compare_key(p) for p in files}
    seen_dirs = {_compare_key(p) for p in dirs}
    for raw in extra_files:
        resolved = _resolved(str(raw))
        if resolved is None:
            continue
        key = _compare_key(resolved)
        if key not in seen_files:
            seen_files.add(key)
            files.append(resolved)
    for raw in extra_dirs:
        resolved = _resolved(str(raw))
        if resolved is None:
            continue
        key = _compare_key(resolved)
        if key not in seen_dirs and key not in seen_files:
            seen_dirs.add(key)
            dirs.append(resolved)
    return base._replace(files=tuple(files), dirs=tuple(dirs))

def _compare_key(path: Path) -> tuple[str, ...]:
    """The form two paths are compared in by the denylist (TASK-19800).

    macOS and Windows filesystems are case-insensitive by DEFAULT, and
    ``Path.resolve()`` does NOT canonicalise case on them -- it resolves
    symlinks and ``..`` but preserves whatever spelling the caller typed.
    So ``~/.SSH/id_rsa`` opens the very same file as ``~/.ssh/id_rsa``
    while comparing unequal to every entry in the denylist. Verified
    end-to-end before this fix: reading ``TLDW_CLI/config.toml`` through
    ``fs_read`` returned the user's real config -- the file holding their
    provider API keys -- while the lowercase spelling was refused.

    Casefolding is applied UNCONDITIONALLY rather than gated on the
    platform or probed per volume. Platform is only a proxy for the real
    question (macOS can be configured case-sensitive; Linux can mount a
    case-insensitive volume), a per-path probe would add I/O to a check
    that runs on every candidate, and the two error directions are not
    symmetric: over-refusing a genuinely distinct ``~/.SSH`` on a
    case-sensitive filesystem costs one explained refusal of a very
    unusual path, while under-refusing leaks a credential. A denylist
    should fail in the cheap direction.

    Comparison stays COMPONENT-wise, so ancestry and lookalike behaviour
    are unchanged: ``~/.sshfoo`` is still not ``~/.ssh``.

    **Do not reuse this for CONFINEMENT checks.** The two fail in opposite
    directions. For a denylist ("is this path forbidden?") folding produces
    extra refusals -- it fails safe. For confinement ("is this path inside
    the allowed root?") folding produces extra ADMISSIONS: on a
    case-sensitive filesystem ``/Root/evil`` would start counting as inside
    ``/root``, which is a loosening, not a hardening. ``is_within`` in
    ``Tools/file_operation_tools.py`` is deliberately left case-sensitive
    for exactly that reason.

    Args:
        path: An already-resolved absolute path.

    Returns:
        The path's components, each casefolded.
    """
    return tuple((part.casefold() for part in path.parts))

def _same_path(a: Path, b: Path) -> bool:
    """Whether two resolved paths denote the same file (see :func:`_compare_key`)."""
    return _compare_key(a) == _compare_key(b)

def _is_within(child: Path, ancestor: Path) -> bool:
    """Whether ``child`` is ``ancestor`` or below it (see :func:`_compare_key`)."""
    child_key = _compare_key(child)
    ancestor_key = _compare_key(ancestor)
    return child_key[:len(ancestor_key)] == ancestor_key

def _name_key(path: Path) -> str:
    """The final component of ``path``, in the same folded form (TASK-19633).

    The name rule's half of :func:`_compare_key`. Kept here rather than
    spelled ``path.name.casefold()`` at the one call site so this module
    has exactly ONE definition of how two path spellings are compared --
    the property TASK-19800 established and the reason a name rule could
    be added without opening a second normalization path.
    """
    key = _compare_key(path)
    return key[-1] if key else ''

def is_sensitive_path(candidate: Path, context: SensitivePathContext | None=None) -> bool:
    """Whether ``candidate`` is a credential, gate-state, or app-state path.

    Comparison is by RESOLVED ancestry, never by string prefix, so
    ``~/.sshfoo`` is not mistaken for ``~/.ssh`` and a symlink cannot
    smuggle a path past the check. Each enumerated database's WAL/SHM/
    rollback-journal sidecar files are refused by the same exact-equality
    rule (see ``_db_sidecar_paths``), since they carry the same class of
    recent data as the database itself.

    This function only decides the question; it enforces nothing by
    itself. Callers must call it explicitly on their target before touching
    the filesystem:

    * ``ReadFileTool.execute``, ``WriteFileTool.execute``,
      ``ListDirectoryTool.execute``, ``GlobFiles.execute`` and
      ``GrepFiles.execute`` in ``Tools/file_operation_tools.py`` (directly,
      or via that module's ``is_within``).
    * ``Tools/local_tool_impls.py``'s ``resolve_workspace_path`` -- the
      choke point the workspace-local ``fs_*``/``git_*`` family resolves
      its path ARGUMENTS through -- plus the per-candidate filters in
      ``list_directory``/``glob_files``/``grep_files`` there (TASK-19551).
      The ``git_*`` tools additionally translate this function's denials
      into git ``:(exclude)`` pathspecs, via
      :func:`sensitive_exclusions_under`, so git never reports a denied
      path in the first place (TASK-19632); see this module's docstring.

    Args:
        candidate: The path a tool intends to touch.
        context: An optional pre-resolved ``SensitivePathContext`` from
            ``resolve_sensitive_context()``. Pass one in when checking many
            candidates within a single tool invocation, so the sensitive-path
            set is resolved once instead of once per candidate. Leave this
            ``None`` (the default) for a one-off, single-path check -- that
            keeps this function's resolution genuinely per-call, which is
            what lets it observe a config-path switch (e.g. the test suite's
            ``TLDW_CONFIG_PATH`` swaps) without going stale.

    Returns:
        True when the path is refused. Fails CLOSED: a path that cannot be
        resolved is treated as sensitive.
    """
    resolved = _resolved(str(candidate))
    if resolved is None:
        return True
    ctx = context if context is not None else resolve_sensitive_context()
    for target in ctx.files:
        if _same_path(resolved, target):
            return True
    for db_path in ctx.db_paths:
        if _same_path(resolved, db_path):
            return True
        if any((_same_path(resolved, s) for s in _db_sidecar_paths(db_path))):
            return True
    for root in ctx.dirs:
        if _is_within(resolved, root):
            return True
    if _name_key(resolved) in _SENSITIVE_FILE_NAMES and (not resolved.is_dir()):
        return True
    for denied_parent in ctx.direct_child_denied_dirs:
        if _same_path(resolved.parent, denied_parent) and (not resolved.is_dir()):
            return True
    return False

class SensitiveExclusion(NamedTuple):
    """One denial of :func:`is_sensitive_path`, expressed relative to a root.

    Produced by :func:`sensitive_exclusions_under` for callers that cannot
    ask this module about each candidate one at a time because they never
    see the candidates -- the ``git_*`` tools, which hand a whole
    repository to ``git`` and get finished output back. Such a caller
    translates these into whatever exclusion syntax its own subprocess
    speaks (git pathspecs, in the only current case) instead of parsing
    that output to decide what to withhold.

    Attributes:
        kind: Which rule produced this denial.

            * ``"subtree"`` -- ``value`` and everything beneath it.
            * ``"file"`` -- exactly ``value``.
            * ``"direct_children"`` -- the direct, non-recursive child
              FILES of the directory ``value`` (never anything deeper,
              and never the subdirectories themselves): the
              container-directory rule in :func:`is_sensitive_path`.
            * ``"name"`` -- any file named ``value`` at ANY depth under
              the root: the TASK-19633 name rule.
        value: For every kind but ``"name"``, a POSIX path RELATIVE to
            the root passed to :func:`sensitive_exclusions_under`, and
            the empty string when it IS that root (possible only for
            ``"subtree"`` -- the whole root is denied, which a caller
            must treat as "refuse outright", there being nothing left to
            show -- and for ``"direct_children"``). For ``"name"``, a
            bare filename, not a path.
    """
    kind: Literal['subtree', 'file', 'direct_children', 'name']
    value: str

def _relative_within(root: Path, candidate: Path) -> str | None:
    """POSIX path of ``candidate`` relative to ``root``, or ``None`` if outside.

    Returns ``""`` when ``candidate`` IS ``root`` (both already resolved).

    Containment is decided by :func:`_is_within`, i.e. through the SAME
    folded key every other denylist comparison uses (TASK-19800), not by
    ``Path.relative_to``'s exact-case parts. ``root`` here is a repository
    root the caller resolved from git's own output while the candidates
    come from config accessors and ``$HOME``; on a case-insensitive
    filesystem those two chains can legitimately disagree about the
    spelling of a shared ancestor, and an exclusion that silently decides
    "outside the repository" for that reason would be a hole, not a
    no-op.

    The RETURNED value is built from the candidate's own components, not
    the folded ones -- the folding decides the relationship, never what
    gets rendered.
    """
    if _same_path(candidate, root):
        return ''
    if not _is_within(candidate, root):
        return None
    return Path(*candidate.parts[len(root.parts):]).as_posix()

def sensitive_exclusions_under(root: Path, context: SensitivePathContext | None=None) -> tuple[SensitiveExclusion, ...]:
    """Every denial that could match something inside ``root``.

    The bridge for callers that delegate enumeration to a subprocess and
    therefore cannot consult :func:`is_sensitive_path` per candidate --
    today only ``Tools/git_tool_impls.py``, which renders these as git
    ``:(exclude)`` pathspecs so ``git diff``/``git status`` never emit a
    denied path's name or content (TASK-19632).

    This function is deliberately the ONE place that enumerates the
    module's rules for that purpose, so a denial added to
    :func:`is_sensitive_path` flows into those tools by being added here
    rather than by someone remembering to update a second list in a
    different package. ``Tests/Tools/test_git_tool_sensitive_paths.py``
    pins the two against each other.

    Args:
        root: The directory the exclusions will be expressed relative to
            (a repository root, in the current caller). Resolved here.
        context: Optional pre-resolved ``SensitivePathContext``; see
            ``resolve_sensitive_context``.

    Returns:
        Deduplicated ``SensitiveExclusion`` entries, in a deterministic
        order. Location-based denials that fall entirely OUTSIDE ``root``
        are omitted (nothing under the root can match them); the name
        rule is always present, since it can match at any depth.
    """
    ctx = context if context is not None else resolve_sensitive_context()
    resolved_root = _resolved(str(root))
    if resolved_root is None:
        return (SensitiveExclusion('subtree', ''),)
    found: list[SensitiveExclusion] = []
    seen: set[SensitiveExclusion] = set()

    def _record(kind: str, value: str) -> None:
        entry = SensitiveExclusion(kind, value)
        if entry not in seen:
            seen.add(entry)
            found.append(entry)
    for denied_dir in ctx.dirs:
        relative = _relative_within(resolved_root, denied_dir)
        if relative is not None:
            _record('subtree', relative)
    denied_files: list[Path] = list(ctx.files)
    for db_path in ctx.db_paths:
        denied_files.append(db_path)
        denied_files.extend(_db_sidecar_paths(db_path))
    for denied_file in denied_files:
        relative = _relative_within(resolved_root, denied_file)
        if relative:
            _record('file', relative)
    for container in ctx.direct_child_denied_dirs:
        relative = _relative_within(resolved_root, container)
        if relative is not None:
            _record('direct_children', relative)
    for name in sorted(_SENSITIVE_FILE_NAMES):
        _record('name', name)
    return tuple(found)

def find_root_binding_conflict(root: Path, context: SensitivePathContext | None=None) -> Path | None:
    """Whether granting recursive access under ``root`` would reach a protected path.

    TASK-857: consulted by ``Workspaces.registry_service.add_folder_binding``,
    the gate that decides whether a folder root may be bound as an
    additional file-tool access root (``Tools/workspace_file_roots.py``
    layers every bound folder on top of the sandbox root, and the file
    tools then trust any path under any of them, subject only to the
    the per-path checks below). That is a fundamentally different question
    from ``is_sensitive_path``'s: that function asks whether one candidate
    READ/WRITE target falls inside a denied area; this asks whether an
    entire subtree about to be granted blanket, recursive reachability
    conflicts with one, which matters in BOTH directions:

    1. ``root`` itself resolves to, or under, one of the fixed sensitive
       directories (``~/.ssh``, ``~/.aws``, ..., the skill-trust subtree)
       or one of this app's own state-container directories
       (``get_user_data_dir()``, the effective config directory, the
       ChromaDB persist directory, the RAG-profile directory). Binding a
       root already inside one of these would make everything else in
       there reachable too -- including subdirectories
       ``is_sensitive_path``'s direct-child-file rule deliberately leaves
       alone for per-path reads (e.g. ``tool_sandbox`` nested under
       ``get_user_data_dir()``), because that rule was designed to catch
       stray loose files in an otherwise-normal directory, not to make an
       entire application-state directory safe to use as a binding root.
       Nothing legitimate needs to bind these directly: the sandbox root
       is already included automatically by
       ``Tools.workspace_file_roots.allowed_file_roots`` without ever
       going through this gate.
    2. ``root`` is coarse enough to CONTAIN one of those same directories
       or one of this app's sensitive single files/database paths -- e.g.
       binding ``~/.local/share`` (which contains ``get_user_data_dir()``
       on a typical Linux install) or ``~/.config`` (which contains
       ``~/.config/tldw_cli``). ``root`` need not itself look sensitive
       for this direction to matter.

    Args:
        root: The already-resolved candidate folder-binding root.
        context: Optional pre-resolved ``SensitivePathContext``; see
            ``resolve_sensitive_context``.

    Returns:
        The protected path ``root`` conflicts with, or ``None`` if it
        conflicts with nothing this module tracks. The returned path is
        meant to be named directly in the caller's rejection message, so
        the user can see exactly what stood in the way. When more than one
        protected path would match, the most specific/direct relationship
        wins, in this priority order: (1) ``root`` IS the protected path
        itself, (2) ``root`` is nested INSIDE a protected directory --
        ties broken toward the DEEPEST (closest, most specific) enclosing
        directory, (3) ``root`` CONTAINS a protected directory or file --
        ties broken toward the SHALLOWEST (closest, most immediate)
        contained path. E.g. binding ``get_user_data_dir()`` itself is
        reported as case (1) even though it also technically contains the
        skill-trust subtree several levels down (case (3)); binding
        ``get_user_data_dir()``'s own PARENT is reported as containing
        ``get_user_data_dir()`` itself (the nearest contained protected
        directory), not the more deeply nested skill-trust subtree beneath
        it -- naming the closest conflict is more actionable than naming
        an obscure one several levels further away.
    """
    ctx = context if context is not None else resolve_sensitive_context()
    protected_dirs = ctx.dirs + ctx.direct_child_denied_dirs
    protected_files: list[Path] = list(ctx.files)
    for db_path in ctx.db_paths:
        protected_files.append(db_path)
        protected_files.extend(_db_sidecar_paths(db_path))
    for protected in tuple(protected_dirs) + tuple(protected_files):
        if _same_path(root, protected):
            return protected
    nested_inside = [protected for protected in protected_dirs if _is_within(root, protected) and (not _same_path(root, protected))]
    if nested_inside:
        return max(nested_inside, key=lambda candidate: len(candidate.parts))
    contains = [protected for protected in protected_dirs if _is_within(protected, root) and (not _same_path(protected, root))]
    contains.extend((protected for protected in protected_files if _is_within(protected, root) and (not _same_path(protected, root))))
    if contains:
        return min(contains, key=lambda candidate: len(candidate.parts))
    return None

def refuses_new_directory_chain(target_dir: Path, context: SensitivePathContext | None=None) -> bool:
    """Whether creating ``target_dir`` (or any not-yet-existing parent of it)
    would plant a directory where this app expects a plain state file.

    ``is_sensitive_path``'s direct-child-file rule is deliberately gated on
    "does this candidate already exist as a directory", so a pre-existing
    container (``tool_sandbox``, ``chromadb``, ``skills``, ...) stays fully
    reachable. That same gate means a candidate that does NOT yet exist is
    judged as if it were a plain file -- correctly refused. But
    ``WriteFileTool``'s ``create_directories=True`` path only ever validates
    the FINAL file being written, never the new directory levels
    ``Path.mkdir(parents=True)`` creates on the way there: a target like
    ``search_history.db/note.txt`` has a parent (``.../search_history.db``)
    that is never itself checked, so nothing stopped an agent from planting
    a directory at that exact name before this app ever created
    ``search_history.db`` as a SQLite file (TASK-849, verified reachable
    end to end through ``WriteFileTool`` under a widened sandbox root). The
    app's own later ``sqlite3.connect(...)`` (or equivalent open) then fails
    outright -- a denial of service, not a disclosure: the collision itself
    carries no credential and grants no elevated access.

    Walking upward from ``target_dir`` while each level still does not
    exist mirrors exactly what ``Path.mkdir(parents=True)`` is about to
    create, and checks each such level with ``is_sensitive_path`` --
    reusing the exact same direct-child-file rule, never a separate check.
    Any level found to already exist ends the walk immediately: an existing
    ancestor is never touched by ``mkdir(parents=True)``, so nothing new
    needs checking above it -- which is what keeps every legitimate
    container directory (created by the app itself before an agent tool
    ever runs) fully reachable.

    Consequence worth naming (Finding 2, follow-up hardening review): a
    not-yet-existing name always fails ``is_sensitive_path``'s ``is_dir()``
    gate, so this refuses creating **any** brand-new subdirectory directly
    inside one of the container directories the direct-child-file rule
    protects (``get_user_data_dir()``, the ChromaDB persist directory,
    ...) -- not only a name that happens to collide with a state file this
    app actually uses. Reproduced: with the sandbox root widened to
    contain the ChromaDB persist directory,
    ``write_file("chromadb/newcoll/x.txt", create_directories=True)`` is
    refused (``newcoll`` does not exist yet, so it fails the same gate a
    genuine collision would), while ``write_file("chromadb/coll1/new.txt",
    create_directories=True)`` succeeds once ``coll1`` already exists as a
    directory -- the walk stops at the first already-existing ancestor, as
    documented above. This is deliberate, not a bug to fix here: telling
    "a legitimate brand-new container" apart from "a shadow directory
    aimed at a not-yet-created state file" by name alone would require
    exactly the enumeration this design avoids (see the module
    docstring), so failing closed is the right default. It is only
    reachable when the sandbox root (or a bound workspace folder) is
    widened to actually contain one of these container directories -- the
    default sandbox root never does. Noted here so the next reader is not
    surprised by an agent's brand-new-subdirectory `write_file` call being
    refused under such a configuration.

    Args:
        target_dir: The directory ``mkdir(parents=True)`` is about to
            create -- typically a write target's parent directory.
        context: Optional pre-resolved ``SensitivePathContext``; see
            ``resolve_sensitive_context``.

    Returns:
        True if ``target_dir`` or any of its not-yet-existing ancestors
        would be a sensitive path once created.
    """
    ctx = context if context is not None else resolve_sensitive_context()
    node = target_dir
    while True:
        resolved = _resolved(str(node))
        if resolved is None:
            return True
        if resolved.exists():
            return False
        if is_sensitive_path(resolved, context=ctx):
            return True
        parent = node.parent
        if parent == node:
            return False
        node = parent
GIT_METADATA_COMPONENT = '.git'

def is_git_metadata_write(path: Path) -> bool:
    """Whether writing ``path`` would modify a repository's own git metadata.

    TASK-19700. Surfaced by TASK-16801's git-modes arc: a repository-supplied
    ``.git/config`` or ``.git/HEAD`` was the precondition for four proven
    data-destruction vectors -- an option-shaped remote or branch name
    reaching git's argv, and ``remote.push``/``remote.mirror``/
    ``push.default=matching`` turning an ordinary push into a forced update
    or a ref deletion. Each is fixed defensively inside the git engine, but
    an agent that can write ``.git/`` reconfigures git for EVERY feature
    that shells out to it, so the upstream cause is denied here.

    Deliberately WRITE-only and deliberately NOT part of
    :func:`is_sensitive_path`: that denylist governs reads as well, and an
    agent reading repository state is legitimate (ADR-032 adopted
    ``allow_hidden`` precisely so a coding agent can see dotfiles). The
    read-side question -- ``.git/config`` can embed a credential in a remote
    URL -- is tracked separately rather than smuggled in here.

    Matching is on an exact path component, so the near-miss names that
    share the prefix stay writable:

    Matching is case-insensitive (a case-insensitive filesystem makes
    ``.GIT`` the same directory), but still component-exact:

    * refused: ``.git``, ``.GIT``, ``.git/config``, ``.git/hooks/pre-commit``, and the
      ``.git`` FILE a linked worktree carries (rewriting it redirects the
      whole repository, so a directory-only check would miss it);
    * allowed: ``.gitignore``, ``.gitattributes``, ``.github/workflows/``.

    Args:
        path: An already-resolved absolute path (the callers resolve before
            calling, so a symlink cannot smuggle a ``.git`` component past
            this check).

    Returns:
        True when any component of ``path`` is exactly ``.git``.
    """
    folded = GIT_METADATA_COMPONENT.casefold()
    return any((part.casefold() == folded for part in path.parts))


# ===========================================================================
# Section: tldw_chatbook.Tools.workspace_wire_decode (extracted from the live module by the
# builder; regenerate rather than editing)
# ===========================================================================
"""Stdlib-only wire codec for the pinned workspace worker.

This module is the worker-bundle counterpart of the parent's pydantic
serde in ``Tools/workspace_tool_protocol.py``. The parent keeps pydantic;
the bundle ships this module alone, so it must import nothing outside the
standard library (the bundle also targets Python 3.10).

Accept/reject behaviour must stay identical to
``WorkspaceToolRequest.from_bytes`` / ``WorkspaceToolResponse.from_bytes``
(the Task 3 conformance corpus is the gate), and
:func:`encode_response` must emit bytes identical to
``WorkspaceToolResponse.to_bytes`` (the byte-identity conformance test in
``Tests/Tools/test_wire_conformance.py`` is that gate — the parent's
parser is order-insensitive, so only an explicit byte-equality pin catches
a field-order drift). The shared wire constants defined here are imported
by the protocol module so the two sides cannot drift structurally.
"""
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
WIRE_VERSION = 1
_GIT_MAX_OUTPUT_BYTES = 1000000
_PATCH_MAX_BYTES = 256 * 1024
_PATCH_MAX_FILES = 20
MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_RESPONSE_BYTES = _GIT_MAX_OUTPUT_BYTES * 6 + 64 * 1024
MAX_STRING_BYTES = 15 * 1024 * 1024
MAX_PATH_BYTES = 16 * 1024
MAX_COLLECTION_ITEMS = 1024
REQUEST_FIELD_NAMES = ('version', 'operation_id', 'operation', 'intent', 'root_locator', 'root_identity', 'ancestor_identities', 'arguments', 'timeout_seconds', 'output_max_bytes')
RESPONSE_FIELD_NAMES = ('version', 'operation_id', 'outcome', 'code', 'result', 'error', 'elapsed_ms', 'truncated', 'cleanup_proven')
DIRECTORY_IDENTITY_FIELD_NAMES = ('device', 'inode', 'mode', 'reparse')
WORKSPACE_OPERATIONS = frozenset({'fs_list', 'fs_read', 'fs_write', 'fs_edit', 'fs_patch', 'fs_glob', 'fs_grep', 'stat_path', 'git_status', 'git_diff', 'git_log', 'git_blame', 'git_branches', 'ping'})
WORKSPACE_WRITE_OPERATIONS = frozenset({'fs_write', 'fs_edit', 'fs_patch'})
WORKSPACE_INTENTS = frozenset({'read', 'write'})
WORKSPACE_OUTCOMES = frozenset({'admitted', 'success', 'failure'})
ARGUMENT_SCHEMAS: dict[str, tuple[frozenset[str], dict[str, str]]] = {'fs_list': (frozenset({'path', 'sensitive_exclusions'}), {'path': 'path', 'sensitive_exclusions': 'sensitive_exclusions'}), 'fs_read': (frozenset({'path', 'sensitive_exclusions'}), {'path': 'path', 'offset': 'positive_int', 'limit': 'nonnegative_int', 'sensitive_exclusions': 'sensitive_exclusions'}), 'fs_write': (frozenset({'path', 'content', 'sensitive_exclusions'}), {'path': 'path', 'content': 'text', 'dry_run': 'bool', 'expected_sha256': 'sha256', 'expected_absent': 'bool', 'sensitive_exclusions': 'sensitive_exclusions'}), 'fs_edit': (frozenset({'path', 'old_string', 'new_string', 'sensitive_exclusions'}), {'path': 'path', 'old_string': 'text', 'new_string': 'text', 'replace_all': 'bool', 'sensitive_exclusions': 'sensitive_exclusions'}), 'fs_patch': (frozenset({'diff', 'sensitive_exclusions'}), {'diff': 'patch', 'dry_run': 'bool', 'targets': 'patch_targets', 'sensitive_exclusions': 'sensitive_exclusions'}), 'fs_glob': (frozenset({'pattern', 'sensitive_exclusions'}), {'pattern': 'glob_pattern', 'max_results': 'positive_int', 'sensitive_exclusions': 'sensitive_exclusions'}), 'fs_grep': (frozenset({'pattern', 'sensitive_exclusions', 'content_exclusions'}), {'pattern': 'text', 'mode': 'grep_mode', 'max_results': 'positive_int', 'sensitive_exclusions': 'sensitive_exclusions', 'content_exclusions': 'sensitive_exclusions'}), 'stat_path': (frozenset({'path'}), {'path': 'path'}), 'git_status': (frozenset({'sensitive_exclusions'}), {'path': 'path', 'sensitive_exclusions': 'sensitive_exclusions'}), 'git_diff': (frozenset({'sensitive_exclusions'}), {'staged': 'bool', 'commit_range': 'text', 'path': 'path', 'stat': 'bool', 'sensitive_exclusions': 'sensitive_exclusions'}), 'git_log': (frozenset({'sensitive_exclusions'}), {'count': 'positive_int', 'path': 'path', 'sensitive_exclusions': 'sensitive_exclusions'}), 'git_blame': (frozenset({'path', 'sensitive_exclusions'}), {'path': 'path', 'start_line': 'positive_int', 'end_line': 'positive_int', 'sensitive_exclusions': 'sensitive_exclusions'}), 'git_branches': (frozenset({'sensitive_exclusions'}), {'sensitive_exclusions': 'sensitive_exclusions'}), 'ping': (frozenset(), {})}
_REQUEST_KEYS = frozenset(REQUEST_FIELD_NAMES)
_RESPONSE_KEYS = frozenset(RESPONSE_FIELD_NAMES)
_IDENTITY_KEYS = frozenset(DIRECTORY_IDENTITY_FIELD_NAMES)
_EXPECTED_INTENTS = {operation: 'write' if operation in WORKSPACE_WRITE_OPERATIONS else 'read' for operation in WORKSPACE_OPERATIONS}
_SENSITIVE_EXCLUSION_KINDS = frozenset({'subtree', 'file', 'direct_children', 'name'})
_GREP_MODES = frozenset({'content', 'files', 'count'})
_SHA256_ALPHABET = frozenset('0123456789abcdef')

class WireDecodeError(ValueError):
    """Raised for a frame this decoder refuses.

    Messages never reflect frame content, mirroring the hygiene of the
    parent's ``WorkspaceProtocolError``.
    """

def decode_request(raw: bytes) -> dict[str, Any]:
    """Decode one strict bounded request frame.

    Args:
        raw: The frame bytes exactly as sent by the parent.

    Returns:
        The validated payload dict; field names are identical to the
        parent's ``_RequestFrame`` model.

    Raises:
        WireDecodeError: For any frame ``WorkspaceToolRequest.from_bytes``
            would refuse.
    """
    doc = _load_object(raw, cap=MAX_REQUEST_BYTES, frame_name='request')
    if set(doc) != _REQUEST_KEYS:
        raise WireDecodeError('protocol frame has invalid keys')
    version = doc['version']
    if type(version) is not int or version != WIRE_VERSION:
        raise WireDecodeError('unsupported protocol version')
    _require_string(doc['operation_id'], 'operation_id')
    operation = _require_closed_string(doc['operation'], WORKSPACE_OPERATIONS, 'operation')
    intent = _require_closed_string(doc['intent'], WORKSPACE_INTENTS, 'intent')
    _require_path(doc['root_locator'], 'root_locator')
    _validate_directory_identity(doc['root_identity'], 'root_identity')
    ancestors = doc['ancestor_identities']
    if type(ancestors) is not list:
        raise WireDecodeError('ancestor_identities must be an array')
    if not ancestors:
        raise WireDecodeError('ancestor_identities must be a non-empty array')
    if len(ancestors) > MAX_COLLECTION_ITEMS:
        raise WireDecodeError('ancestor_identities exceeds collection ceiling')
    for ancestor in ancestors:
        _validate_directory_identity(ancestor, 'ancestor_identities')
    if intent != _EXPECTED_INTENTS[operation]:
        raise WireDecodeError('operation intent mismatch')
    _validate_arguments(doc['arguments'], operation=operation)
    _require_positive_int(doc['timeout_seconds'], 'timeout_seconds')
    _require_positive_int(doc['output_max_bytes'], 'output_max_bytes')
    return doc

def decode_response(raw: bytes) -> dict[str, Any]:
    """Decode one strict bounded response frame.

    Args:
        raw: The frame bytes exactly as emitted by a worker.

    Returns:
        The validated payload dict; field names are identical to the
        parent's ``_ResponseFrame`` model.

    Raises:
        WireDecodeError: For any frame ``WorkspaceToolResponse.from_bytes``
            would refuse (absent the optional ``expected_operation_id``
            caller-side assertion, which is not frame validity).
    """
    doc = _load_object(raw, cap=MAX_RESPONSE_BYTES, frame_name='response')
    if set(doc) != _RESPONSE_KEYS:
        raise WireDecodeError('protocol frame has invalid keys')
    version = doc['version']
    if type(version) is not int or version != WIRE_VERSION:
        raise WireDecodeError('unsupported protocol version')
    _require_string(doc['operation_id'], 'operation_id')
    outcome = _require_closed_string(doc['outcome'], WORKSPACE_OUTCOMES, 'outcome')
    _require_string(doc['code'], 'code')
    result = _require_optional_string(doc['result'], 'result')
    error = _require_optional_string(doc['error'], 'error')
    _require_nonnegative_int(doc['elapsed_ms'], 'elapsed_ms')
    if type(doc['truncated']) is not bool:
        raise WireDecodeError('truncated must be a bool')
    if type(doc['cleanup_proven']) is not bool:
        raise WireDecodeError('cleanup_proven must be a bool')
    if outcome == 'success' and error is not None:
        raise WireDecodeError('successful response cannot contain error')
    if outcome == 'failure' and result is not None:
        raise WireDecodeError('failed response cannot contain result')
    return doc

def encode_response(frame: Mapping[str, Any]) -> bytes:
    """Serialize one response frame in the parent's exact byte layout.

    The single encoder for worker-emitted frames: it rebuilds the payload
    in ``RESPONSE_FIELD_NAMES`` order — the order the parent's
    ``WorkspaceToolResponse.to_bytes`` emits — with the same JSON flags
    (``allow_nan=False``, ``ensure_ascii=False``, ``separators=(",", ":")``),
    so input-mapping order cannot leak into the wire bytes. A byte-identity
    conformance test pins this against the parent encoder; the parent's own
    parser is order-insensitive, which is exactly why the pin exists.

    Args:
        frame: The response fields; must have exactly the response keys.

    Returns:
        The serialized frame bytes.

    Raises:
        WireDecodeError: If the keys are wrong or the values cannot be
            serialized under the frame contract.
    """
    if set(frame) != _RESPONSE_KEYS:
        raise WireDecodeError('protocol frame has invalid keys')
    payload = {name: frame[name] for name in RESPONSE_FIELD_NAMES}
    try:
        return json.dumps(payload, allow_nan=False, ensure_ascii=False, separators=(',', ':')).encode('utf-8', errors='strict')
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise WireDecodeError('protocol frame cannot be serialized') from error

def _load_object(raw: bytes, *, cap: int, frame_name: str) -> dict[str, Any]:
    if type(raw) is not bytes:
        raise WireDecodeError(f'{frame_name} frame must be bytes')
    if len(raw) > cap:
        raise WireDecodeError(f'{frame_name} frame exceeds byte ceiling')
    try:
        decoded = raw.decode('utf-8', errors='strict')
        value = json.loads(decoded, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_non_finite)
    except WireDecodeError:
        raise
    except UnicodeDecodeError as error:
        raise WireDecodeError(f'{frame_name} frame is not UTF-8') from error
    except (json.JSONDecodeError, ValueError) as error:
        raise WireDecodeError(f'{frame_name} frame is malformed') from error
    if type(value) is not dict:
        raise WireDecodeError(f'{frame_name} frame must be an object')
    return value

def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise WireDecodeError('duplicate key in protocol frame')
        value[key] = item
    return value

def _reject_non_finite(value: str) -> None:
    raise WireDecodeError('non-finite JSON value')

def _require_string(value: Any, field_name: str, *, cap: int=MAX_STRING_BYTES) -> str:
    if type(value) is not str:
        raise WireDecodeError(f'{field_name} must be a string')
    if '\x00' in value:
        raise WireDecodeError(f'{field_name} contains NUL')
    try:
        byte_count = len(value.encode('utf-8', errors='strict'))
    except UnicodeEncodeError as error:
        raise WireDecodeError(f'{field_name} is not UTF-8 encodable') from error
    if byte_count > cap:
        raise WireDecodeError(f'{field_name} exceeds byte ceiling')
    return value

def _require_optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, field_name)

def _require_closed_string(value: Any, choices: frozenset[str], field_name: str) -> str:
    text = _require_string(value, field_name)
    if text not in choices:
        raise WireDecodeError(f'unsupported {field_name}')
    return text

def _require_path(value: Any, field_name: str) -> str:
    return _require_string(value, field_name, cap=MAX_PATH_BYTES)

def _require_positive_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise WireDecodeError(f'{field_name} must be a positive int')
    return value

def _require_nonnegative_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise WireDecodeError(f'{field_name} must be a non-negative int')
    return value

def _validate_directory_identity(value: Any, field_name: str) -> None:
    if type(value) is not dict:
        raise WireDecodeError(f'{field_name} must be an object')
    if set(value) != _IDENTITY_KEYS:
        raise WireDecodeError(f'{field_name} has invalid keys')
    for key in ('device', 'inode', 'mode'):
        _require_nonnegative_int(value[key], f'{field_name}.{key}')
    if type(value['reparse']) is not bool:
        raise WireDecodeError(f'{field_name}.reparse must be a bool')

def _validate_arguments(value: Any, *, operation: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise WireDecodeError('arguments must be an object')
    required, accepted = ARGUMENT_SCHEMAS[operation]
    if not required.issubset(value) or not set(value).issubset(accepted):
        raise WireDecodeError('invalid operation arguments')
    for key, argument in value.items():
        _validate_argument_value(argument, kind=accepted[key])
    return value

def _validate_argument_value(value: Any, *, kind: str) -> None:
    if kind == 'path':
        _require_string(value, 'argument path', cap=MAX_PATH_BYTES)
        return
    if kind == 'text':
        _require_string(value, 'argument text')
        return
    if kind == 'glob_pattern':
        validate_glob_pattern(value)
        return
    if kind == 'patch':
        _require_string(value, 'patch diff', cap=_PATCH_MAX_BYTES)
        return
    if kind == 'patch_targets':
        if type(value) is not list or not value or len(value) > _PATCH_MAX_FILES:
            raise WireDecodeError('invalid patch targets')
        for target in value:
            _require_path(target, 'patch target')
        return
    if kind == 'bool':
        if type(value) is not bool:
            raise WireDecodeError('argument must be a bool')
        return
    if kind == 'sha256':
        digest = _require_string(value, 'SHA-256 digest', cap=64)
        if len(digest) != 64 or any((character not in _SHA256_ALPHABET for character in digest)):
            raise WireDecodeError('invalid SHA-256 digest')
        return
    if kind == 'positive_int':
        _require_positive_int(value, 'argument')
        return
    if kind == 'nonnegative_int':
        _require_nonnegative_int(value, 'argument')
        return
    if kind == 'grep_mode':
        mode = _require_string(value, 'grep mode')
        if mode not in _GREP_MODES:
            raise WireDecodeError('invalid grep mode')
        return
    if kind == 'sensitive_exclusions':
        if type(value) is not list or len(value) > MAX_COLLECTION_ITEMS:
            raise WireDecodeError('invalid sensitive exclusions')
        for exclusion in value:
            if type(exclusion) is not dict or set(exclusion) != {'kind', 'value'}:
                raise WireDecodeError('invalid sensitive exclusions')
            kind_value = _require_closed_string(exclusion['kind'], _SENSITIVE_EXCLUSION_KINDS, 'sensitive exclusion kind')
            text = _require_path(exclusion['value'], 'sensitive exclusion value')
            if '\x00' in text or (kind_value == 'name' and ('/' in text or '\\' in text)):
                raise WireDecodeError('invalid sensitive exclusions')
        return
    raise WireDecodeError('invalid argument schema')

def validate_glob_pattern(value: Any) -> str:
    """Validate a platform-neutral, root-relative glob grammar.

    Public because the pinned dispatcher (stdlib-only import closure,
    Phase 0c) validates ``fs_glob`` patterns through this module rather
    than the parent's pydantic protocol module. Raises ``WireDecodeError``
    where the parent's ``WorkspaceProtocolError`` flavour would.
    """
    if type(value) is not str:
        raise WireDecodeError('glob pattern must be a string')
    if '\x00' in value:
        raise WireDecodeError('invalid glob pattern')
    pattern = _require_string(value, 'glob pattern')
    windows = Path(pattern.replace('\\', '/'))
    if pattern.startswith(('/', '\\')) or ':' in pattern.split('/')[0] or any((part == '..' for part in pattern.replace('\\', '/').split('/'))) or windows.is_absolute():
        raise WireDecodeError('invalid glob pattern')
    return pattern


# ===========================================================================
# Section: tldw_chatbook.Tools.worker_watchdog (extracted from the live module by the
# builder; regenerate rather than editing)
# ===========================================================================
"""Two-tier hard-timeout watchdog for the pinned workspace worker.

Part of the worker's stdlib-only import closure (Phase 0c; Task 8
flattens this module into the remote worker bundle). Shared by the
LOCAL worker and the shipped bundle: ``run_workspace_worker`` arms both
tiers with the request's ``timeout_seconds`` immediately after decoding
and disarms them once the exchange's frames are out (Task 12).

Why two tiers — the ``fs_grep`` catastrophic-regex case
------------------------------------------------------

``fs_grep`` runs caller-supplied patterns through ``re`` in the worker
process. A pattern like ``(a+)+$`` against a near-miss line (60KB of
``a`` plus one trailing ``b``) drives the C regex engine into quadratic-
or-worse backtracking, and that engine does not release the GIL while
it spins:

* **Tier 1 — graceful ``threading.Timer``** fires at the budget,
  sweeps every path in :data:`TEMP_REGISTRY` (best-effort ``unlink``),
  writes the fixed stderr line, and ``os._exit(75)``. It produces a
  clean, attributable death — but it is PYTHON code: it needs the GIL,
  and a starving regex can hold that GIL forever, so tier 1 alone can
  be starved.
* **Tier 2 — OS backstop ``signal.alarm(budget + 2)`` with the DEFAULT
  action**. No Python handler is installed (a handler would need the
  GIL and starve exactly like the Timer); the kernel terminates the
  process on delivery, GIL or no GIL. The two-second grace exists so
  the graceful tier wins whenever it can.

An optional RLIMIT_CPU ceiling (soft == hard == ``ceil(budget * 2)``)
backs both for CPU-spin cases on platforms that have ``resource``:
exceeding it raises SIGXCPU with its default action. It is guarded —
``resource`` and its limits are not universal — and NOT restored on
disarm: ``setrlimit`` is per-process and cannot be un-set, which is
fine because the worker is one-shot (one exchange per process).

The transport side (Task 11) buckets "admitted marker + exit 75" (tier
1) and "admitted marker + completion-deadline kill" (tier 2, seen by
the caller as death by signal) as the same failure kind, OP_TIMEOUT.

Exit-code reservation: ``os._exit(WATCHDOG_EXIT_CODE)`` below is the
ONLY place in the worker's closure that may hard-exit with 75; worker
failure paths exit 2. Task 11's transport keys on it.
"""
import math
import os
import signal
import threading
WATCHDOG_EXIT_CODE = 75
WATCHDOG_STDERR_MARKER = b'tldw-worker-watchdog\n'
TEMP_REGISTRY: list[str] = []
_armed_timer: threading.Timer | None = None

def register_temp(path: str) -> None:
    """Record one temp-file path for the watchdog's cleanup sweep."""
    if path not in TEMP_REGISTRY:
        TEMP_REGISTRY.append(path)

def unregister_temp(path: str) -> None:
    """Drop one temp-file path (its operation completed or cleaned up)."""
    if path in TEMP_REGISTRY:
        TEMP_REGISTRY.remove(path)

def _watchdog_fire(temp_registry: list[str]) -> None:
    """Tier-1 expiry: sweep temps, mark stderr, hard-exit 75.

    Everything here is best-effort by construction — the process is
    already past its budget, and ``os._exit`` skips ``finally`` blocks
    and interpreter cleanup, so THIS callback is the one place the
    sweep can run. It must never raise before the ``os._exit``.
    """
    for path in list(temp_registry):
        try:
            os.unlink(path)
        except OSError:
            pass
    try:
        os.write(2, WATCHDOG_STDERR_MARKER)
    except OSError:
        pass
    os._exit(WATCHDOG_EXIT_CODE)

def arm_watchdog(budget_seconds: float, temp_registry: list[str]) -> None:
    """Arm both hard-timeout tiers for one exchange.

    Args:
        budget_seconds: The request's ``timeout_seconds`` — over ssh the
            transport writes the REMAINING budget into that field; this
            function just consumes it. Values ``<= 0`` arm NOTHING: the
            wire decoder and the transport both require a positive
            budget, so reaching here with a non-positive one is already
            a caller bug, and skipping is the defensive choice (a 0
            budget would otherwise fire tier 1 instantly).
        temp_registry: The live registry tier 1 sweeps (by reference —
            registrations after arming are still seen; the worker passes
            ``TEMP_REGISTRY`` itself).
    """
    global _armed_timer
    disarm_watchdog()
    try:
        budget = float(budget_seconds)
    except (TypeError, ValueError):
        return
    if not math.isfinite(budget) or budget <= 0:
        return
    timer = threading.Timer(budget, _watchdog_fire, args=(temp_registry,))
    timer.daemon = True
    timer.start()
    _armed_timer = timer
    if hasattr(signal, 'alarm'):
        signal.alarm(max(1, int(budget) + 2))
    try:
        import resource
        cpu_seconds = max(1, math.ceil(budget * 2))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    except (ImportError, OSError, ValueError):
        pass

def disarm_watchdog() -> None:
    """Cancel tier 1 and zero tier 2 once the exchange completed.

    Called after the final response frame is emitted: a completed op
    must not die late. The RLIMIT_CPU ceiling is per-process and cannot
    be restored — it simply persists for the worker's remaining life,
    which is fine (the worker is one-shot: one exchange, then exit).
    """
    global _armed_timer
    if _armed_timer is not None:
        _armed_timer.cancel()
        _armed_timer = None
    if hasattr(signal, 'alarm'):
        signal.alarm(0)


# ===========================================================================
# Section: tldw_chatbook.Tools.local_tool_impls (extracted from the live module by the
# builder; regenerate rather than editing)
# ===========================================================================
"""Sync core implementations for workspace-local agent tools.

Plain functions, no async, no Textual, no event loop — callable from the
agent runtime's worker thread via Agents/local_tool_provider.py. Every
failure raises LocalToolError; the provider converts those (and any other
exception) into ToolResult error strings — nothing raises across the
provider boundary.

Path safety has TWO layers, and both are mandatory (TASK-19551):

1. **Confinement** to the configured ``[console] workspace_root``
   (ADR-032), enforced by ``validate_path`` inside
   ``resolve_workspace_path`` — the single choke point every path-taking
   function here (and in ``patch_tool_impls``/``git_tool_impls``) funnels
   through.
2. **The sensitive-path denylist** (``Utils/sensitive_paths.py``), also
   enforced inside that same choke point. Confinement alone is not enough:
   the shipped ``workspace_root`` default is the app's cwd at startup, so
   launching from ``$HOME`` makes ``$HOME`` the confinement root — and
   ``~/.ssh/id_rsa``, ``~/.aws/credentials``, this app's own ``config.toml``
   and ``mcp_permissions.json`` are all inside it. Reading them exfiltrates
   credentials into a transcript sent to a provider; WRITING
   ``mcp_permissions.json`` turns every ``ask`` into ``allow``, a one-step
   bypass of the permission gate that authorized the call.

The choke point covers a path the model NAMES. It cannot cover entries a
tool presents that the model never named, so the three ENUMERATING tools
(``list_directory``/``glob_files``/``grep_files``) — which resolve only the
workspace ROOT through it — each filter their own candidates against the
same denylist, resolving the sensitive-path context ONCE per invocation
(see ``Utils.sensitive_paths.resolve_sensitive_context``) rather than once
per candidate. ``grep_files`` is the sharpest of the three — it READS every
file it walks and prints matching lines — so its check runs before the
read, not after.

``Tools/git_tool_impls.py`` shares this choke point for its path arguments,
and ``path`` is optional on ``git_status``/``git_log``/``git_diff``: with
it omitted the choke point sees only the repository root, and ``git_diff``
returned the CONTENT of a denylisted file from a CLEAN worktree
(TASK-19632). That family solves the same problem a third way -- it cannot
filter candidates it never sees, because git enumerates the repository for
it, so it instead excludes denylisted paths from git's INPUT by pathspec
(``git_tool_impls._denylist_pathspecs``). Do not extend that family
assuming the choke point alone makes its output safe: whichever of the
three mechanisms fits, a tool that PRESENTS paths the model never named
needs one.
"""
import difflib
import hashlib
import heapq
import json
import os
import stat as stat_module
import threading
import uuid
from pathlib import Path
from typing import BinaryIO, Literal
MAX_LIST_ENTRIES = 200
MAX_SCAN_ENTRIES = 10000
MAX_READ_CHARS = 32 * 1024
MAX_READ_FILE_BYTES = 10 * 1024 * 1024
MAX_GLOB_RESULTS = 100
MAX_GREP_RESULTS = 100
_MAX_GREP_FILE_BYTES = 2 * 1024 * 1024
_MAX_WRITE_DIFF_CHARS = 12 * 1024
_WRITE_LOCKS_GUARD = threading.Lock()
_WRITE_LOCKS: dict[str, threading.Lock] = {}

class LocalToolError(ValueError):
    """Model-actionable failure from a local tool (path, not-found, …)."""
PathIntent = Literal['read', 'write', 'list']
_INTENT_VERBS: dict[str, str] = {'read': 'read', 'write': 'written', 'list': 'listed'}

def resolve_workspace_path(path: str, workspace_root: Path, *, intent: PathIntent='read', context: SensitivePathContext | None=None) -> Path:
    """Resolve ``path`` against ``workspace_root``, confined and denylisted.

    The single choke point for this tool family: every ``fs_*`` and
    ``git_*`` core function resolves a model-supplied path here (the git
    ones one hop away, via ``prepare_repository``/``_prepare_for_path``/
    ``_repo_relative_path``), so both path checks are enforced in ONE place
    rather than re-implemented per tool — which is exactly how the denylist
    came to be missing from all seven ``fs_*`` tools (TASK-19551). It
    governs the path a caller PASSES; it does not filter what a tool
    returns (see the module docstring for where that distinction bites).

    Two checks, in order:

    1. **Confinement** (``validate_path``). Hidden components (``.github/``,
       ``.gitignore``) are allowed under the root, per ADR-032: a coding
       agent that cannot read a repository's dotfile configuration is
       useless, and the ADR adopted ``allow_hidden`` for exactly this
       family. That parameter is deliberately KEPT — dotted names are how
       ``~/.ssh``/``~/.aws`` are spelled, but "starts with a dot" is a name
       heuristic, not a security boundary, and check 2 is the one designed
       to answer that question (by resolved ancestry, so a symlink or a
       ``~/.sshfoo`` lookalike cannot game it).
    2. **The sensitive-path denylist** (``is_sensitive_path``), matching
       what ``Tools/file_operation_tools.py``'s ``ReadFileTool``/
       ``WriteFileTool``/``ListDirectoryTool`` already do for the other
       file-tool family, message shape included, so agent-facing refusals
       stay consistent across the two.

    For ``intent="write"`` the denylist check is additionally applied to
    every not-yet-existing ancestor of the target
    (``refuses_new_directory_chain``), the same guard ``WriteFileTool``
    consults before ``mkdir(parents=True)``. No tool in this family creates
    directories today (a write target's parent must already exist), so this
    normally short-circuits on the first existing ancestor; it is here so
    that a future ``create_directories``-style option cannot reintroduce
    TASK-849's shadow-directory denial of service by forgetting it.

    Args:
        path: The user/model-supplied path, absolute or relative to
            ``workspace_root``.
        workspace_root: The confinement root the resolved path must stay
            within.
        intent: What the caller is about to do with the path — selects the
            refusal verb and enables the new-directory-chain guard for
            writes. Never weakens a check.
        context: Optional pre-resolved ``SensitivePathContext``. Callers
            that check many paths in one tool invocation (the enumerating
            tools, ``patch_files``' multi-file loop) resolve one with
            ``Utils.sensitive_paths.resolve_sensitive_context()`` and pass
            it through, so the ~11 config accessors behind the denylist are
            resolved once per CALL rather than once per path. ``None``
            resolves it fresh — that still enforces the denylist.

    Returns:
        The validated absolute ``Path`` inside ``workspace_root``.

    Raises:
        LocalToolError: If the path resolves outside ``workspace_root``, or
            is a protected credential/gate-state/app-state path.
    """
    try:
        resolved = validate_path(path, workspace_root, redact_paths=True, allow_hidden=True)
    except ValueError as exc:
        raise LocalToolError(f"Path '{path}' is outside the workspace root ({workspace_root})") from exc
    verb = _INTENT_VERBS.get(intent, 'accessed')
    if is_sensitive_path(resolved, context=context):
        raise LocalToolError(f"Refused: '{path}' is a protected path and cannot be {verb}")
    if intent == 'write' and is_git_metadata_write(resolved):
        raise LocalToolError(f"Refused: '{path}' is inside a repository's .git metadata and cannot be {verb}")
    if intent == 'write' and refuses_new_directory_chain(resolved.parent, context=context):
        raise LocalToolError(f"Refused: creating '{resolved.parent}' would collide with a protected path")
    return resolved

def list_directory(path: str, *, workspace_root: Path, max_entries: int=MAX_LIST_ENTRIES) -> str:
    """One-level listing of ``path``: ``name/`` for dirs, ``name`` for files.

    Directories sort before files, each group case-insensitively by name.
    Output is capped at ``max_entries`` with a trailing truncation notice.
    The directory SCAN itself is also capped at ``MAX_SCAN_ENTRIES`` — only
    the scanned entries are sorted (dirs-first contract preserved for the
    scanned set), and hitting the scan cap appends a "directory too large"
    notice instead of silently presenting a partial listing as complete.

    Individual denylisted ENTRIES are omitted from the listing (TASK-19551),
    mirroring ``ListDirectoryTool``'s per-entry check in
    ``Tools/file_operation_tools.py``: refusing the target directory alone
    would still disclose this app's own ``mcp_permissions.json`` or
    ``chachanotes.db`` by name and existence whenever an ordinary,
    listable ancestor happens to contain them.

    Args:
        path: Directory to list, absolute or relative to
            ``workspace_root``.
        workspace_root: The confinement root ``path`` must resolve within.
        max_entries: Maximum number of entries included in the output
            before a truncation notice is appended.

    Returns:
        The newline-joined listing, with a truncation and/or scan-cap
        notice appended when the directory exceeded either cap.

    Raises:
        LocalToolError: If ``path`` is not an existing directory, or
            resolves outside ``workspace_root``.
    """
    sensitive_ctx = resolve_sensitive_context()
    root = resolve_workspace_path(path, workspace_root, intent='list', context=sensitive_ctx)
    return _list_relative_directory(root.relative_to(Path(workspace_root).resolve()), workspace=Path(workspace_root).resolve(), max_entries=max_entries, sensitive_exclusions=sensitive_exclusions_under(Path(workspace_root).resolve(), sensitive_ctx), display_path=path)

def _list_relative_directory(relative: Path, *, workspace: Path, max_entries: int, sensitive_exclusions: tuple[SensitiveExclusion, ...], display_path: str | None=None) -> str:
    """List a pinned-root-relative directory without opening an absolute path."""
    target = workspace / relative
    if not _relative_target_is_safe(relative, workspace, sensitive_exclusions, is_directory=True) or not target.is_dir():
        raise LocalToolError(f'not a directory: {display_path or relative}')
    scanned: list[Path] = []
    scan_capped = False
    for index, entry in enumerate(target.iterdir()):
        if index >= MAX_SCAN_ENTRIES:
            scan_capped = True
            break
        entry_relative = _workspace_relative_path(entry, workspace)
        if not _relative_target_is_safe(entry_relative, workspace, sensitive_exclusions, is_directory=entry.is_dir()):
            continue
        scanned.append(entry)
    entries = sorted(scanned, key=lambda p: (p.is_file(), p.name.lower()))
    lines = [f'{p.name}/' if p.is_dir() else p.name for p in entries[:max_entries]]
    remaining = len(entries) - max_entries
    if remaining > 0:
        lines.append(f'… ({remaining} more entries, truncated)')
    if scan_capped:
        lines.append(f'… (directory too large; showing first {len(entries)} of many entries)')
    return '\n'.join(lines)

def read_file(path: str, *, workspace_root: Path, offset: int=1, limit: int | None=None) -> str:
    """Read ``path`` with 1-based line numbers, ``offset``/``limit`` paging.

    Lines are numbered from 1 (matching claude-code's Read). ``offset`` is
    the 1-based first line to return; ``limit`` caps the line count.
    Binary files (NUL byte in the first 8 KiB) and missing files raise
    LocalToolError with model-actionable messages. UTF-16 files trip the
    binary sniff; other non-UTF-8 text reads with U+FFFD replacement.
    """
    root = resolve_workspace_path(path, workspace_root, intent='read')
    return _read_relative_file(root.relative_to(Path(workspace_root).resolve()), workspace=Path(workspace_root).resolve(), offset=offset, limit=limit, sensitive_exclusions=sensitive_exclusions_under(Path(workspace_root).resolve()), display_path=path)

def _read_relative_file(relative: Path, *, workspace: Path, offset: int, limit: int | None, sensitive_exclusions: tuple[SensitiveExclusion, ...], display_path: str | None=None, content_stamps: bool=False) -> str:
    """Read a pinned-root-relative text file without reopening its resolved path.

    The file is opened exactly ONCE and fully read as bytes: the binary
    sniff, the decoded text, and (when ``content_stamps`` is set) the
    CAS stamp digest all come from that single read, so the rendered
    content and its sha256/size stamps can never be a torn pair under
    concurrent modification. (``str.splitlines`` splits untranslated
    ``\\r``/``\\r\\n`` exactly where the old universal-newline
    ``read_text`` translation produced splits, so the output is
    byte-identical to the previous two-open implementation.)

    Args:
        relative: Root-relative target, already safety-checked.
        workspace: The confinement root ``relative`` resolves against.
        offset: 1-based first line to render.
        limit: Optional cap on the rendered line count.
        sensitive_exclusions: Deny rules applied to the target.
        display_path: Caller-facing path for error messages.
        content_stamps: Append the worker-reported CAS tail
            (``\\nsha256: <hex>\\nsize: <n>``) computed from the same
            read — the pinned dispatch path sets this; the plain local
            ``read_file`` surface does not.

    Returns:
        The numbered (or notice) body, plus the stamp tail when
        requested.

    Raises:
        LocalToolError: If the target is missing/protected, oversized,
            or binary.
    """
    target = workspace / relative
    if not _relative_target_is_safe(relative, workspace, sensitive_exclusions, is_directory=False) or not target.is_file():
        raise LocalToolError(f'file not found: {display_path or relative}')
    file_size = target.stat().st_size
    if file_size > MAX_READ_FILE_BYTES:
        raise LocalToolError(f"'{display_path or relative}' is too large to read ({file_size} bytes; maximum {MAX_READ_FILE_BYTES})")
    data = target.read_bytes()
    if b'\x00' in data[:8192]:
        raise LocalToolError(f"'{display_path or relative}' appears to be binary; fs_read only reads text files")
    lines = data.decode('utf-8', errors='replace').splitlines()
    if not lines:
        body = '(empty file)'
    else:
        start = max(offset, 1) - 1
        if start >= len(lines):
            body = f'(offset {offset} is past end of file; {len(lines)} lines total)'
        else:
            window = lines[start:] if limit is None else lines[start:start + max(limit, 0)]
            body = '\n'.join((f'{i}\t{line}' for i, line in enumerate(window, start=start + 1)))
            if len(body) > MAX_READ_CHARS:
                body = body[:MAX_READ_CHARS] + '\n… [truncated]'
    if content_stamps:
        body += f'\nsha256: {hashlib.sha256(data).hexdigest()}\nsize: {len(data)}'
    return body

def stat_path(path: str, *, workspace_root: Path) -> str:
    """Return a small allowlisted metadata view for one workspace path.

    Args:
        path: File or directory to inspect.
        workspace_root: Confinement root the path must resolve within.

    Returns:
        Workspace-relative path, kind, size, nanosecond mtime, and mode.

    Raises:
        LocalToolError: If the path is outside the workspace or protected.
        OSError: If the resolved path cannot be inspected.
    """
    root = Path(workspace_root).resolve()
    resolved = resolve_workspace_path(path, root, intent='read')
    relative = resolved.relative_to(root)
    return _format_stat_result(relative, resolved.stat())

def _stat_relative_path(relative: Path) -> str:
    """Inspect one already-validated path relative to the pinned worker root."""
    if relative.is_absolute() or '..' in relative.parts:
        raise LocalToolError('stat path must be workspace-relative')
    return _format_stat_result(relative, relative.stat())

def _format_stat_result(relative: Path, info: os.stat_result) -> str:
    """Format the stable allowlisted stat fields for one relative path."""
    kind = 'directory' if stat_module.S_ISDIR(info.st_mode) else 'file' if stat_module.S_ISREG(info.st_mode) else 'other'
    return '\n'.join((f'path: {relative}', f'type: {kind}', f'size: {info.st_size}', f'modified_ns: {info.st_mtime_ns}', f'mode: {info.st_mode & 4095:04o}'))

def write_file(path: str, content: str, *, workspace_root: Path, dry_run: bool=False, expected_sha256: str | None=None, expected_absent: bool=False) -> str:
    """Create or overwrite ``path`` with ``content`` (full-file write).

    The parent directory must already exist (deliberate divergence from
    claude-code's Write, to catch model path typos early — spec §2).
    """
    workspace = Path(workspace_root).resolve()
    root = resolve_workspace_path(path, workspace, intent='write')
    return _write_relative_file(root.relative_to(workspace), content, workspace=workspace, display_path=path, dry_run=dry_run, expected_sha256=expected_sha256, expected_absent=expected_absent)

def _write_relative_file(relative: Path, content: str, *, workspace: Path, display_path: str | None=None, dry_run: bool=False, expected_sha256: str | None=None, expected_absent: bool=False, content_stamps: bool=False) -> str:
    """Preview or atomically write one admitted path with optional CAS.

    Args:
        relative: Root-relative target, already safety-checked.
        content: Full replacement content.
        workspace: The confinement root ``relative`` resolves against.
        display_path: Caller-facing path for messages.
        dry_run: Preview only; nothing is written and no stamp tail is
            appended (the preview is a JSON object).
        expected_sha256: Optional CAS precondition on the current bytes.
        expected_absent: Optional CAS precondition that no file exists.
        content_stamps: Append the worker-reported CAS tail
            (``\\nsha256: <hex>\\nsize: <n>``) of the bytes just written
            (Task 16 write-path parity with ``_read_relative_file``).
            Computed from ``data`` -- the exact in-memory bytes handed to
            the atomic writer -- so content and stamps can never be a
            torn pair; the plain local ``write_file`` surface does not
            set this and stays byte-identical.
    """
    target = workspace / relative
    shown = display_path or str(relative)
    if not target.parent.is_dir():
        raise LocalToolError(f'parent directory does not exist for: {shown}')
    if expected_sha256 is not None and expected_absent:
        raise LocalToolError('expected_sha256 and expected_absent are mutually exclusive')
    if expected_sha256 is not None and (not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or any((character not in '0123456789abcdef' for character in expected_sha256))):
        raise LocalToolError('expected_sha256 must be a lowercase SHA-256 digest')
    if type(expected_absent) is not bool or type(dry_run) is not bool:
        raise LocalToolError('write flags must be boolean')
    try:
        data = content.encode('utf-8')
    except UnicodeEncodeError as exc:
        raise LocalToolError(f'content is not UTF-8 encodable (lone surrogate?): {exc}') from exc
    canonical_key = os.path.normcase(str(target.absolute()))
    lock = _write_lock_for(canonical_key)
    with lock:
        current = _read_write_target(target, shown)
        current_digest = None if current is None else hashlib.sha256(current).hexdigest()
        if expected_absent and current is not None:
            raise LocalToolError('write precondition failed: target is present')
        if expected_sha256 is not None and current_digest != expected_sha256:
            raise LocalToolError('write precondition failed: target digest changed')
        if dry_run:
            return _write_preview_json(shown=shown, current=current, current_digest=current_digest, replacement=data)
        _atomic_write_target(target, data, shown=shown, expected_sha256=expected_sha256, expected_absent=expected_absent)
    summary = f'wrote {len(content)} characters to {shown}'
    if content_stamps and (not dry_run):
        summary += f'\nsha256: {hashlib.sha256(data).hexdigest()}\nsize: {len(data)}'
    return summary

def _write_lock_for(key: str) -> threading.Lock:
    with _WRITE_LOCKS_GUARD:
        return _WRITE_LOCKS.setdefault(key, threading.Lock())

def _read_write_target(target: Path, shown: str) -> bytes | None:
    try:
        before = os.lstat(target)
    except FileNotFoundError:
        return None
    if not stat_module.S_ISREG(before.st_mode) or stat_module.S_ISLNK(before.st_mode):
        raise LocalToolError(f'write target is not a regular file: {shown}')
    flags = os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0)
    if hasattr(os, 'O_NOFOLLOW'):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags)
        try:
            opened = os.fstat(descriptor)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            finished = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after = os.lstat(target)
    except OSError:
        raise LocalToolError(f'write target changed while reading: {shown}') from None
    identities = tuple(((value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns) for value in (before, opened, finished, after)))
    if len(set(identities)) != 1:
        raise LocalToolError(f'write target changed while reading: {shown}')
    return b''.join(chunks)

def _write_preview_json(*, shown: str, current: bytes | None, current_digest: str | None, replacement: bytes) -> str:
    before_text = '' if current is None else current.decode('utf-8', errors='replace')
    after_text = replacement.decode('utf-8')
    diff = ''.join(difflib.unified_diff(before_text.splitlines(keepends=True), after_text.splitlines(keepends=True), fromfile=shown if current is not None else '/dev/null', tofile=shown))
    if len(diff) > _MAX_WRITE_DIFF_CHARS:
        diff = diff[:_MAX_WRITE_DIFF_CHARS] + '\n… [truncated]'
    return json.dumps({'target_state': 'absent' if current is None else 'present', 'current_sha256': current_digest or 'absent', 'replacement_sha256': hashlib.sha256(replacement).hexdigest(), 'replacement_bytes': len(replacement), 'diff': diff}, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':'))

def _atomic_write_target(target: Path, data: bytes, *, shown: str, expected_sha256: str | None, expected_absent: bool) -> None:
    """Write through one pinned parent descriptor and same-directory temp.

    task-32808.5 AC#3: deliberately NOT routed through
    ``Utils/atomic_file_ops.atomic_write_text`` -- this path needs a pinned
    parent ``dir_fd`` (TOCTOU-safe rename), ``O_NOFOLLOW``, an sha256/absent
    precondition and a target lock, none of which the shared helper offers.
    """
    directory_flags = os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0)
    if hasattr(os, 'O_DIRECTORY'):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, 'O_NOFOLLOW'):
        directory_flags |= os.O_NOFOLLOW
    try:
        parent_fd = os.open(target.parent, directory_flags)
    except OSError:
        raise LocalToolError(f'write parent changed: {shown}') from None
    temp_name = f'.chatbook-write-{uuid.uuid4().hex}.tmp'
    temp_path = os.path.abspath(os.path.join(str(target.parent), temp_name))
    temp_created = False
    target_lock = None
    try:
        if expected_sha256 is not None:
            target_lock = _acquire_expected_target_lock(parent_fd, target.name, shown, expected_sha256)
        live, live_mode = _read_relative_descriptor(parent_fd, target.name, shown)
        live_digest = None if live is None else hashlib.sha256(live).hexdigest()
        if expected_absent and live is not None:
            raise LocalToolError('write precondition failed: target is present')
        if expected_sha256 is not None and live_digest != expected_sha256:
            raise LocalToolError('write precondition failed: target digest changed')
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_CLOEXEC', 0)
        if hasattr(os, 'O_NOFOLLOW'):
            flags |= os.O_NOFOLLOW
        temp_fd = os.open(temp_name, flags, 438, dir_fd=parent_fd)
        temp_created = True
        register_temp(temp_path)
        try:
            if live_mode is not None:
                os.fchmod(temp_fd, live_mode)
            view = memoryview(data)
            while view:
                written = os.write(temp_fd, view)
                view = view[written:]
            os.fsync(temp_fd)
        finally:
            os.close(temp_fd)
        if expected_absent:
            os.link(temp_name, target.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd, follow_symlinks=False)
            os.unlink(temp_name, dir_fd=parent_fd)
            temp_created = False
            unregister_temp(temp_path)
        else:
            if target_lock is not None:
                _assert_expected_target_is_current(parent_fd, target.name, shown, expected_sha256, target_lock)
            os.replace(temp_name, target.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            temp_created = False
            unregister_temp(temp_path)
        os.fsync(parent_fd)
    except FileExistsError:
        raise LocalToolError('write precondition failed: target is present') from None
    except LocalToolError:
        raise
    except OSError:
        raise LocalToolError(f'atomic write failed for: {shown}') from None
    finally:
        if temp_created:
            try:
                os.unlink(temp_name, dir_fd=parent_fd)
            except OSError:
                pass
            unregister_temp(temp_path)
        if target_lock is not None:
            _unlock_target_handle(target_lock)
            target_lock.close()
        os.close(parent_fd)

def _portalocker_module():
    """Return the ``portalocker`` module, or ``None`` where absent.

    The remote worker bundle (Phase 1d, Task 8) concatenates this module
    onto a bare remote interpreter that has no third-party packages, so
    the CAS write lock needs a stdlib fallback there. In the parent
    application ``portalocker`` is a pinned dependency and the fallback
    never runs.
    """
    try:
        raise ImportError('portalocker is not available inside the remote worker bundle')
    except ImportError:
        return None
    return portalocker

def _lock_expected_target(handle: BinaryIO) -> None:
    """Take one non-blocking exclusive advisory lock on an open target.

    Uses portalocker where installed; otherwise falls back to the same
    POSIX advisory lock portalocker itself delegates to (``fcntl.flock``,
    exclusive + non-blocking). Contention or a lock failure raises
    ``LocalToolError`` with the fixed precondition message callers map to.
    """
    portalocker = _portalocker_module()
    if portalocker is not None:
        try:
            portalocker.lock(handle, portalocker.LockFlags.EXCLUSIVE | portalocker.LockFlags.NON_BLOCKING)
        except portalocker.exceptions.LockException:
            handle.close()
            raise LocalToolError('write precondition failed: target is being modified') from None
        return
    import fcntl
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise LocalToolError('write precondition failed: target is being modified') from None

def _unlock_target_handle(handle: BinaryIO) -> None:
    """Best-effort release of the advisory lock taken by the helper above."""
    portalocker = _portalocker_module()
    try:
        if portalocker is not None:
            portalocker.unlock(handle)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass

def _acquire_expected_target_lock(parent_fd: int, filename: str, shown: str, expected_sha256: str) -> BinaryIO:
    """Acquire a non-blocking process lock on the expected target inode."""
    flags = os.O_RDWR | getattr(os, 'O_CLOEXEC', 0)
    if hasattr(os, 'O_NOFOLLOW'):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(filename, flags, dir_fd=parent_fd)
        handle = os.fdopen(descriptor, 'r+b', buffering=0)
    except OSError:
        raise LocalToolError('write precondition failed: target digest changed') from None
    _lock_expected_target(handle)
    try:
        _assert_expected_target_is_current(parent_fd, filename, shown, expected_sha256, handle)
    except Exception:
        _unlock_target_handle(handle)
        handle.close()
        raise
    return handle

def _assert_expected_target_is_current(parent_fd: int, filename: str, shown: str, expected_sha256: str | None, locked_handle: BinaryIO) -> None:
    """Verify the locked inode is still the named path with the expected bytes."""
    if expected_sha256 is None:
        raise LocalToolError('write precondition failed: target digest changed')
    locked_info = os.fstat(locked_handle.fileno())
    if not stat_module.S_ISREG(locked_info.st_mode):
        raise LocalToolError(f'write target is not a regular file: {shown}')
    locked_handle.seek(0)
    locked_digest = hashlib.sha256(locked_handle.read()).hexdigest()
    flags = os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0)
    if hasattr(os, 'O_NOFOLLOW'):
        flags |= os.O_NOFOLLOW
    try:
        current_fd = os.open(filename, flags, dir_fd=parent_fd)
    except OSError:
        raise LocalToolError('write precondition failed: target digest changed') from None
    try:
        current_info = os.fstat(current_fd)
    finally:
        os.close(current_fd)
    if (locked_info.st_dev, locked_info.st_ino) != (current_info.st_dev, current_info.st_ino) or locked_digest != expected_sha256:
        raise LocalToolError('write precondition failed: target digest changed')

def _read_relative_descriptor(parent_fd: int, filename: str, shown: str) -> tuple[bytes | None, int | None]:
    flags = os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0)
    if hasattr(os, 'O_NOFOLLOW'):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(filename, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        return (None, None)
    except OSError:
        raise LocalToolError(f'write target changed: {shown}') from None
    try:
        info = os.fstat(descriptor)
        if not stat_module.S_ISREG(info.st_mode):
            raise LocalToolError(f'write target is not a regular file: {shown}')
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return (b''.join(chunks), info.st_mode & 4095)
    finally:
        os.close(descriptor)

def edit_file(path: str, old_string: str, new_string: str, *, workspace_root: Path, replace_all: bool=False) -> str:
    """Replace exact ``old_string`` with ``new_string`` in ``path``.

    Fails unless the match is unique (or ``replace_all=True``); ambiguity
    errors include the match count so the model can self-correct. Exact
    semantics per spec §2 (claude-code Edit parity). Reads and writes with
    ``newline=""`` so CRLF files are not silently converted to LF. The
    result is encoded BEFORE the file is opened for writing, so an
    unencodable ``new_string`` (e.g. a lone surrogate from tool-call JSON)
    fails without truncating the file.
    """
    workspace = Path(workspace_root).resolve()
    root = resolve_workspace_path(path, workspace, intent='write')
    return _edit_relative_file(root.relative_to(workspace), old_string, new_string, workspace=workspace, replace_all=replace_all, display_path=path)

def _edit_relative_file(relative: Path, old_string: str, new_string: str, *, workspace: Path, replace_all: bool=False, display_path: str | None=None, content_stamps: bool=False) -> str:
    """Edit one already-admitted path relative to an I/O root.

    Args:
        relative: Root-relative target, already safety-checked.
        old_string: Exact text to replace.
        new_string: Replacement text.
        workspace: The confinement root ``relative`` resolves against.
        replace_all: Replace every occurrence instead of requiring a
            unique match.
        display_path: Caller-facing path for messages.
        content_stamps: Append the worker-reported CAS tail of the bytes
            just written (``\\nsha256: <hex>\\nsize: <n>``), computed
            from the same in-memory ``data`` handed to the atomic writer
            (Task 16 write-path parity with ``_read_relative_file``);
            the plain local ``edit_file`` surface does not set this and
            stays byte-identical.
    """
    shown = display_path or str(relative)
    if not old_string:
        raise LocalToolError('old_string must not be empty')
    if old_string == new_string:
        raise LocalToolError('old_string and new_string are identical')
    target = workspace / relative
    if not target.is_file():
        raise LocalToolError(f'file not found: {shown}')
    try:
        with open(target, encoding='utf-8', newline='') as fh:
            content = fh.read()
    except UnicodeDecodeError as exc:
        raise LocalToolError(f"'{shown}' is not valid UTF-8; fs_edit only edits text files") from exc
    count = content.count(old_string)
    if count == 0:
        raise LocalToolError(f'old_string not found in {shown}')
    if count > 1 and (not replace_all):
        raise LocalToolError(f'old_string appears {count} times in {shown}; provide more context to make it unique, or set replace_all=true')
    updated = content.replace(old_string, new_string)
    try:
        data = updated.encode('utf-8')
    except UnicodeEncodeError as exc:
        raise LocalToolError(f'new_string is not UTF-8 encodable (lone surrogate?): {exc}') from exc
    _atomic_write_target(target, data, shown=shown, expected_sha256=None, expected_absent=False)
    n = count if replace_all else 1
    summary = f"made {n} replacement{('s' if n != 1 else '')} in {shown}"
    if content_stamps:
        summary += f'\nsha256: {hashlib.sha256(data).hexdigest()}\nsize: {len(data)}'
    return summary

def glob_files(pattern: str, *, workspace_root: Path, max_results: int=MAX_GLOB_RESULTS) -> str:
    """Match ``pattern`` under the workspace, newest-mtime first, capped.

    Paths in the result are workspace-relative. Hidden files/dirs under the
    root ARE matched (workspace policy, ADR-032). Matches that escape the
    root via ``..`` pattern segments are excluded (lexical check only —
    symlinks are not resolved, per ADR-032 review). Denylisted matches are
    excluded too (TASK-19551): under a home-rooted workspace ``**/*`` would
    otherwise report ``.ssh/id_rsa`` back to the model by name.

    Memory is bounded at ``max_results``: a min-heap keeps only the newest N
    while the total is counted in one pass (no full-list materialization or
    sort of a huge workspace).

    Raises:
        LocalToolError: If ``max_results`` is below 1.
    """
    if max_results < 1:
        raise LocalToolError('max_results must be >= 1')
    sensitive_ctx = resolve_sensitive_context()
    root = resolve_workspace_path('.', workspace_root, intent='list', context=sensitive_ctx)
    return _glob_relative_files(pattern, workspace=Path(workspace_root).resolve(), max_results=max_results, sensitive_exclusions=sensitive_exclusions_under(root, sensitive_ctx))

def _glob_relative_files(pattern: str, *, workspace: Path, max_results: int, sensitive_exclusions: tuple[SensitiveExclusion, ...], validate_targets: bool=False) -> str:
    """Glob from the pinned working directory using only relative I/O paths."""
    heap: list[tuple[float, Path]] = []
    total = 0
    for p in workspace.glob(pattern):
        try:
            if not p.is_file():
                continue
            norm = Path(os.path.normpath(p))
            if not norm.is_relative_to(workspace):
                continue
            rendered = norm.relative_to(workspace)
            if validate_targets:
                if not _relative_target_is_safe(rendered, workspace, sensitive_exclusions, is_directory=False):
                    continue
            elif _is_relative_sensitive_path(rendered, sensitive_exclusions, is_directory=False):
                continue
            mtime = p.stat().st_mtime
        except OSError:
            continue
        total += 1
        if len(heap) < max_results:
            heapq.heappush(heap, (mtime, rendered))
        elif mtime > heap[0][0]:
            heapq.heapreplace(heap, (mtime, rendered))
    best = sorted(heap, key=lambda t: t[0], reverse=True)
    lines = [str(relative) for _, relative in best]
    if total > max_results:
        lines.append(f'… ({total - max_results} more, truncated)')
    return '\n'.join(lines) if lines else f'(no files matching {pattern!r})'

def grep_files(pattern: str, *, workspace_root: Path, mode: str='content', max_results: int=MAX_GREP_RESULTS) -> str:
    """Regex search under the workspace.

    Modes: ``content`` -> ``relpath:lineno:line``; ``files`` -> one relpath
    per matching file; ``count`` -> ``relpath:N``. Binary and >2 MiB files
    are skipped. Invalid regex raises LocalToolError. File order is
    filesystem order (no global sort — that would materialize the whole
    tree); within a file, lines are in order.

    Denylisted files are skipped BEFORE they are read (TASK-19551): this is
    the sharpest of the three enumerating tools, since ``content`` mode
    prints matching LINES — a home-rooted workspace and a pattern as bland
    as ``KEY`` would otherwise emit ``~/.ssh/id_rsa`` into the transcript.

    Raises:
        LocalToolError: If ``max_results`` is below 1.
    """
    import re
    try:
        re.compile(pattern)
    except re.error as exc:
        raise LocalToolError(f'invalid regex: {exc}') from exc
    if mode not in {'content', 'files', 'count'}:
        raise LocalToolError(f'unknown mode: {mode}')
    if max_results < 1:
        raise LocalToolError('max_results must be >= 1')
    sensitive_ctx = resolve_sensitive_context()
    root = resolve_workspace_path('.', workspace_root, intent='list', context=sensitive_ctx)
    return _grep_relative_files(pattern, workspace=Path(workspace_root).resolve(), mode=mode, max_results=max_results, sensitive_exclusions=sensitive_exclusions_under(root, sensitive_ctx))

def _grep_relative_files(pattern: str, *, workspace: Path, mode: str, max_results: int, sensitive_exclusions: tuple[SensitiveExclusion, ...]) -> str:
    """Grep pinned-root-relative files while refusing escaping symlink content."""
    import re
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        raise LocalToolError(f'invalid regex: {exc}') from exc
    if mode not in ('content', 'files', 'count'):
        raise LocalToolError(f'unknown mode: {mode}')
    if max_results < 1:
        raise LocalToolError('max_results must be >= 1')
    shown: list[str] = []
    total = 0
    for p in workspace.rglob('*'):
        try:
            if not p.is_file() or p.stat().st_size > _MAX_GREP_FILE_BYTES:
                continue
            relative = _workspace_relative_path(p, workspace)
            if not _relative_target_is_safe(relative, workspace, sensitive_exclusions, is_directory=False):
                continue
        except OSError:
            continue
        try:
            text = p.read_text(encoding='utf-8')
        except (UnicodeDecodeError, OSError):
            continue
        rel = str(relative)
        hits = [f'{i}:{line}' for i, line in enumerate(text.splitlines(), 1) if rx.search(line)]
        if not hits:
            continue
        if mode == 'content':
            for hit in hits:
                total += 1
                if len(shown) < max_results:
                    shown.append(f'{rel}:{hit}')
        else:
            total += 1
            if len(shown) < max_results:
                shown.append(rel if mode == 'files' else f'{rel}:{len(hits)}')
    if total > max_results:
        shown.append(f'… ({total - max_results} more, truncated)')
    return '\n'.join(shown) if shown else f'(no matches for {pattern!r})'

def _relative_target_is_safe(relative: Path, workspace: Path, exclusions: tuple[SensitiveExclusion, ...], *, is_directory: bool) -> bool:
    """Require both lexical and resolved targets to be admissible for I/O."""
    try:
        if _is_relative_sensitive_path(relative, exclusions, is_directory=is_directory):
            return False
        resolved_workspace = workspace.resolve()
        resolved = (workspace / relative).resolve()
        if not resolved.is_relative_to(resolved_workspace):
            return False
        return not _is_relative_sensitive_path(resolved.relative_to(resolved_workspace), exclusions, is_directory=is_directory)
    except OSError:
        return False

def _workspace_relative_path(path: Path, workspace: Path) -> Path:
    """Return a lexical candidate name relative to a workspace I/O base."""
    return path.relative_to(workspace)

def _is_relative_sensitive_path(relative: Path, exclusions: tuple[SensitiveExclusion, ...], *, is_directory: bool) -> bool:
    """Apply parent-derived sensitive exclusions to one relative candidate."""
    parts = tuple((part.casefold() for part in relative.parts))
    for kind, value in exclusions:
        value_parts = tuple((part.casefold() for part in Path(value).parts))
        if kind == 'subtree' and parts[:len(value_parts)] == value_parts:
            return True
        if kind == 'file' and parts == value_parts:
            return True
        if kind == 'direct_children' and (not is_directory and parts[:-1] == value_parts):
            return True
        if kind == 'name' and (not is_directory) and (relative.name.casefold() == value):
            return True
    return False


# ===========================================================================
# Section: tldw_chatbook.Tools.workspace_root_pin (extracted from the live module by the
# builder; regenerate rather than editing)
# ===========================================================================
"""Pin one admitted workspace root inside a single-threaded helper process."""
import ctypes
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterator

class WorkspaceRootPinError(RuntimeError):
    """Raised when an admitted root cannot be retained and verified safely."""

@dataclass(slots=True)
class PinnedWorkspaceRoot:
    """A verified helper-local current directory retained by an open handle."""
    canonical_locator: Path = field(repr=False)
    identity: DirectoryIdentity
    root_fd: int | None = field(default=None, repr=False)
    _previous_fd: int | None = field(default=None, repr=False)
    _windows_handle: int | None = field(default=None, repr=False)
    _previous_directory: str | None = field(default=None, repr=False)
    _closed: bool = field(default=False, repr=False)

    def relative_path(self, value: str) -> Path:
        """Return one lexical path relative to the retained helper root."""
        if type(value) is not str or not value or '\x00' in value:
            raise WorkspaceRootPinError('workspace operation requires a relative path')
        for lexical in (PurePosixPath(value), PureWindowsPath(value)):
            if lexical.drive or lexical.root or lexical.anchor:
                raise WorkspaceRootPinError('workspace operation requires a relative path')
        relative = Path(value)
        if relative.is_absolute() or '..' in relative.parts:
            raise WorkspaceRootPinError('workspace operation requires a relative path')
        return relative

    def close(self) -> None:
        """Restore the prior helper directory and release retained handles."""
        if self._closed:
            return
        self._closed = True
        if os.name == 'posix':
            self._close_posix()
        elif os.name == 'nt':
            self._close_windows()

    def _close_posix(self) -> None:
        restore_error = False
        try:
            if self._previous_fd is not None:
                os.fchdir(self._previous_fd)
        except OSError:
            restore_error = True
        finally:
            for descriptor in (self.root_fd, self._previous_fd):
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError:
                        restore_error = True
            self.root_fd = None
            self._previous_fd = None
        if restore_error:
            raise WorkspaceRootPinError('workspace root pin cleanup failed')

    def _close_windows(self) -> None:
        restore_error = False
        try:
            if self._previous_directory is not None:
                _windows_set_current_directory(self._previous_directory)
        except WorkspaceRootPinError:
            restore_error = True
        finally:
            if self._windows_handle is not None:
                try:
                    _windows_close_handle(self._windows_handle)
                except WorkspaceRootPinError:
                    restore_error = True
            self._windows_handle = None
        if restore_error:
            raise WorkspaceRootPinError('workspace root pin cleanup failed')

@contextmanager
def pin_workspace_root(canonical_locator: Path, chain: DirectoryChain) -> Iterator[PinnedWorkspaceRoot]:
    """Open, identity-check, chdir to, and retain one admitted root."""
    locator = Path(canonical_locator)
    try:
        locator = validate_path(locator, locator, redact_paths=True, allow_hidden=True)
    except ValueError as error:
        raise WorkspaceRootPinError('invalid admitted workspace root') from error
    if type(chain) is not DirectoryChain or not chain.identities or (not locator.is_absolute()) or (locator != chain.canonical_root):
        raise WorkspaceRootPinError('invalid admitted workspace root')
    if os.name == 'posix':
        pinned = _pin_posix(locator, chain.identities[0])
    elif os.name == 'nt':
        pinned = _pin_windows(locator, chain.identities[0])
    else:
        raise WorkspaceRootPinError('workspace root pinning is unsupported')
    try:
        yield pinned
    finally:
        pinned.close()

def _pin_posix(locator: Path, expected: DirectoryIdentity) -> PinnedWorkspaceRoot:
    required = ('O_DIRECTORY', 'O_NOFOLLOW', 'O_CLOEXEC')
    if not hasattr(os, 'fchdir') or any((not hasattr(os, name) for name in required)):
        raise WorkspaceRootPinError('workspace root pinning is unsupported')
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    root_fd: int | None = None
    previous_fd: int | None = None
    try:
        previous_fd = os.open('.', flags)
        root_fd = os.open(locator, flags)
        if _posix_identity(root_fd) != expected:
            raise WorkspaceRootPinError('workspace root identity mismatch')
        os.fchdir(root_fd)
        if directory_identity_from_stat(os.stat('.', follow_symlinks=False)) != expected:
            raise WorkspaceRootPinError('workspace root identity mismatch')
        return PinnedWorkspaceRoot(canonical_locator=locator, identity=expected, root_fd=root_fd, _previous_fd=previous_fd)
    except WorkspaceRootPinError:
        _restore_and_close_posix(previous_fd, root_fd)
        raise
    except (DirectoryIdentityError, OSError, ValueError) as error:
        _restore_and_close_posix(previous_fd, root_fd)
        raise WorkspaceRootPinError('workspace root pinning failed') from error

def _posix_identity(descriptor: int) -> DirectoryIdentity:
    try:
        return directory_identity_from_stat(os.fstat(descriptor))
    except (DirectoryIdentityError, OSError) as error:
        raise WorkspaceRootPinError('workspace root metadata unavailable') from error

def _restore_and_close_posix(previous_fd: int | None, root_fd: int | None) -> None:
    if previous_fd is not None:
        try:
            os.fchdir(previous_fd)
        except OSError:
            pass
    for descriptor in (root_fd, previous_fd):
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass

def _pin_windows(locator: Path, expected: DirectoryIdentity) -> PinnedWorkspaceRoot:
    if expected.reparse:
        raise WorkspaceRootPinError('unsafe workspace root metadata')
    handle: int | None = None
    previous_directory: str | None = None
    try:
        previous_directory = os.getcwd()
        handle, identity, reparse = _windows_open_directory(locator)
        if reparse:
            raise WorkspaceRootPinError('unsafe workspace root metadata')
        if (identity.device, identity.inode) != (expected.device, expected.inode):
            raise WorkspaceRootPinError('workspace root identity mismatch')
        _windows_set_current_directory(str(locator))
        verification, verified_identity, verified_reparse = _windows_open_directory(Path('.'))
        try:
            if verified_reparse or (verified_identity.device, verified_identity.inode) != (expected.device, expected.inode):
                raise WorkspaceRootPinError('workspace root identity mismatch')
        finally:
            _windows_close_handle(verification)
        return PinnedWorkspaceRoot(canonical_locator=locator, identity=expected, _windows_handle=handle, _previous_directory=previous_directory)
    except WorkspaceRootPinError:
        if previous_directory is not None:
            try:
                _windows_set_current_directory(previous_directory)
            except WorkspaceRootPinError:
                pass
        if handle is not None:
            try:
                _windows_close_handle(handle)
            except WorkspaceRootPinError:
                pass
        raise
    except (OSError, ValueError) as error:
        if handle is not None:
            try:
                _windows_close_handle(handle)
            except WorkspaceRootPinError:
                pass
        raise WorkspaceRootPinError('workspace root pinning failed') from error

def _windows_open_directory(path: Path) -> tuple[int, DirectoryIdentity, bool]:
    if os.name != 'nt':
        raise WorkspaceRootPinError('Windows root pinning is unavailable')
    from ctypes import wintypes

    class ByHandleFileInformation(ctypes.Structure):
        _fields_ = [('dwFileAttributes', wintypes.DWORD), ('ftCreationTime', wintypes.FILETIME), ('ftLastAccessTime', wintypes.FILETIME), ('ftLastWriteTime', wintypes.FILETIME), ('dwVolumeSerialNumber', wintypes.DWORD), ('nFileSizeHigh', wintypes.DWORD), ('nFileSizeLow', wintypes.DWORD), ('nNumberOfLinks', wintypes.DWORD), ('nFileIndexHigh', wintypes.DWORD), ('nFileIndexLow', wintypes.DWORD)]
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.GetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.POINTER(ByHandleFileInformation)]
    kernel32.GetFileInformationByHandle.restype = wintypes.BOOL
    file_read_attributes = 128
    share_all = 1 | 2 | 4
    open_existing = 3
    backup_semantics = 33554432
    open_reparse_point = 2097152
    raw_handle = kernel32.CreateFileW(str(path), file_read_attributes, share_all, None, open_existing, backup_semantics | open_reparse_point, None)
    invalid_handle = ctypes.c_void_p(-1).value
    handle = int(ctypes.cast(raw_handle, ctypes.c_void_p).value or 0)
    if not handle or handle == invalid_handle:
        raise WorkspaceRootPinError('workspace root open failed')
    information = ByHandleFileInformation()
    if not kernel32.GetFileInformationByHandle(handle, ctypes.byref(information)):
        _windows_close_handle(handle)
        raise WorkspaceRootPinError('workspace root metadata unavailable')
    inode = int(information.nFileIndexHigh) << 32 | int(information.nFileIndexLow)
    reparse = bool(int(information.dwFileAttributes) & 1024)
    return (handle, DirectoryIdentity(device=int(information.dwVolumeSerialNumber), inode=inode, mode=0, reparse=reparse), reparse)

def _windows_set_current_directory(path: str) -> None:
    if os.name != 'nt':
        raise WorkspaceRootPinError('Windows root pinning is unavailable')
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.SetCurrentDirectoryW.argtypes = [wintypes.LPCWSTR]
    kernel32.SetCurrentDirectoryW.restype = wintypes.BOOL
    if not kernel32.SetCurrentDirectoryW(path):
        raise WorkspaceRootPinError('workspace root current-directory pin failed')

def _windows_close_handle(handle: int) -> None:
    if os.name != 'nt':
        raise WorkspaceRootPinError('Windows root pinning is unavailable')
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    if not kernel32.CloseHandle(handle):
        raise WorkspaceRootPinError('workspace root handle cleanup failed')


# ===========================================================================
# Section: tldw_chatbook.Tools.git_tool_impls (extracted from the live module by the
# builder; regenerate rather than editing)
# ===========================================================================
"""Sync ``run_git`` wrapper + repository preparation for the read-only git tools.

Ported from tldw_server's
tldw_Server_API/app/core/MCP_unified/modules/implementations/git_module.py
@ 5605b9d9906322c2e6b5342b48c391ae674d315e
(https://github.com/rmusser01/tldw_server, GPL-3.0-only).

``run_git`` (argv validation, sanitized environment, bounded output reads)
is a near-verbatim sync port of ``AsyncGitCommandRunner`` (reference
:120-297). The async ``_communicate_bounded``/``_read_bounded_stream`` pair
becomes a Popen + reader-thread bounded read loop so output is killed AT the
cap rather than fully buffered — ``subprocess.run(capture_output=True)``
would let a runaway ``git log`` buffer unbounded memory before the 1 MB cap
applies, so capture-then-truncate is deliberately NOT used.

``prepare_repository`` adapts the reference's ``_prepare_repository``
(:1134+) to the chatbook shape: confinement via ``resolve_workspace_path``,
sync ``run_git``, and ``LocalToolError`` (shared model-actionable error)
instead of the reference's ``_GitToolError``. Deviations from the reference
(deliberate, per the phase-3b-ii plan):

1. A timeout raises ``LocalToolError("git command timed out ...")`` instead
   of returning a result with ``timed_out=True`` (the field is kept on
   ``GitCommandResult`` for shape fidelity with the reference).
2. ``GitCommandResult.duration_ms`` is dropped; truncation is surfaced with
   a human-readable marker appended to the affected stream.
3. The repo root must be the workspace root or INSIDE it; a repo root above
   the workspace root is refused (the reference's ``_path_inside`` rule,
   :2062-2063), so the model cannot read repo state outside confinement.

The tool cores (``git_status``/``git_branches``/``git_log``/``git_diff``/
``git_blame``) are sync adaptations of the reference's ``_execute_status``
(:570), ``_execute_branches`` (:621), ``_execute_log`` (:831),
``_execute_diff`` (:709) + ``_run_diff_command`` (:1049-1080 — the
``--no-ext-diff``/``--no-textconv``/``--no-color`` machine-safe flags are
ported), and ``_execute_blame`` (:892), returning plain text for the agent
provider instead of the reference's structured dicts. Disclosed deviations
from the reference (deliberate, per the phase-3b-ii plan):

(a) ``git_diff`` adds ``commit_range`` and ``stat`` modes the reference
    does not have; the reference's third ``working_tree`` scope is omitted
    (``staged=False`` maps to ``unstaged``, ``staged=True`` to ``staged``).
    ``commit_range`` is regex-validated (``^[A-Za-z0-9._/~^-]+$``) before
    entering argv to keep the fixed-argv guarantee meaningful.
(b) ``git_log`` defaults ``count=20`` (the reference has no default — an
    absent limit falls back to its max-100); both clamp to 1..100.
(c) ``_parse_blame_header`` accepts 3-field headers (``sha orig final``),
    not just 4-field group headers: git emits 3-field headers for the
    remaining lines of a commit group, and the reference's 4-field minimum
    silently drops those lines.
(d) ``git_blame``'s ``-L`` range is optional (the reference always passes
    one); the range is capped at ``GIT_BLAME_MAX_LINES`` lines.

TASK-19632 adds one thing the reference has no equivalent of: these tools
enumerate a repository on the model's behalf, so ``path`` being optional
meant no candidate ever reached the sensitive-path denylist and
``git_diff`` returned ``~/.ssh/id_rsa``'s CONTENT from a CLEAN worktree.
The fix constrains git's INPUT rather than filtering its OUTPUT -- see
``_denylist_pathspecs`` for the mechanism and ``Utils/sensitive_paths.py``
for why that direction was chosen. Two properties of pathspecs are
load-bearing there and are easy to lose in a later edit: an exclude-only
pathspec list applies to the whole tree (no positive pathspec is needed),
and pathspec MAGIC is honoured after ``--``, so every pathspec built here
carries explicit magic -- ``:(literal)`` for the one that SCOPES output,
``:(exclude,literal,icase)``/``:(exclude,glob,icase)`` for the ones that
DENY from it. The ``icase`` asymmetry is deliberate; see
``_literal_pathspec`` and ``_denylist_pathspecs``.
"""
import contextlib
import os
import queue
import re
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
GIT_TIMEOUT_SECONDS = 30.0
GIT_MAX_OUTPUT_BYTES = 1000000
REPOSITORY_DISCOVERY_TIMEOUT_SECONDS = 5.0
GIT_TRUNCATED_MARKER = '\n...[output truncated]'
_ALLOWED_GIT_SUBCOMMANDS = frozenset({'--version', 'blame', 'branch', 'diff', 'log', 'ls-files', 'rev-parse', 'status'})

@dataclass(frozen=True, slots=True)
class GitCommandResult:
    """Result returned by :func:`run_git` (sync port of the reference's)."""
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    truncated: bool = False

def _git_environment() -> dict[str, str]:
    env: dict[str, str] = {}
    path_value = os.environ.get('PATH')
    if path_value:
        env['PATH'] = path_value
    for key in ('SYSTEMROOT', 'WINDIR'):
        value = os.environ.get(key)
        if value:
            env[key] = value
    env.update({'GIT_TERMINAL_PROMPT': '0', 'GIT_OPTIONAL_LOCKS': '0', 'GIT_PAGER': 'cat', 'GIT_EXTERNAL_DIFF': '', 'GIT_CONFIG_COUNT': '1', 'GIT_CONFIG_KEY_0': 'core.fsmonitor', 'GIT_CONFIG_VALUE_0': 'false'})
    return env

def _extract_subcommand_and_validate_globals(argv: list[str]) -> str | None:
    index = 1
    while index < len(argv):
        value = argv[index]
        if value == '--version':
            if len(argv) != 2:
                raise LocalToolError('git global option --version must be used alone')
            return value
        if value == '--no-pager':
            index += 1
            continue
        if value.startswith('-'):
            raise LocalToolError(f'git global option is not allowlisted: {value}')
        return value
    return None

def _validate_argv(argv: list[str]) -> None:
    if not argv or argv[0] != 'git':
        raise LocalToolError('git runner only executes git commands')
    subcommand = _extract_subcommand_and_validate_globals(argv)
    if subcommand is None:
        raise LocalToolError('git requires a subcommand (e.g. status, diff, log)')
    if subcommand not in _ALLOWED_GIT_SUBCOMMANDS:
        raise LocalToolError(f'git subcommand is not allowlisted: {subcommand}')

def _read_bounded_stream(stream, max_output_bytes: int) -> tuple[bytes, bool]:
    """Read a stream until EOF or the byte cap; report cap hits."""
    output = bytearray()
    while len(output) <= max_output_bytes:
        read_size = min(8192, max_output_bytes + 1 - len(output))
        chunk = stream.read(read_size)
        if not chunk:
            return (bytes(output), False)
        output.extend(chunk)
        if len(output) > max_output_bytes:
            return (bytes(output[:max_output_bytes]), True)
    return (bytes(output[:max_output_bytes]), True)

def run_git(argv: list[str], *, cwd: Path | None=None, executable: Path | None=None, own_process_group: bool=True, timeout: float=GIT_TIMEOUT_SECONDS, max_output_bytes: int=GIT_MAX_OUTPUT_BYTES) -> GitCommandResult:
    """Run an allowlisted git command with bounded output and a timeout.

    Fixed argv only: ``argv[0]`` must be logical ``git``, the only permitted
    global option is ``--no-pager`` (``--version`` must stand alone), and the
    subcommand must be in ``_ALLOWED_GIT_SUBCOMMANDS``. The absolute executable
    replaces logical argv zero only at launch. The environment is sanitized
    (PATH + git safety vars only) and stdin is
    DEVNULL. Output per stream is capped at ``max_output_bytes`` — the
    process is killed when the cap is exceeded (never fully buffered) and a
    truncation marker is appended. Timeout kills the process and raises.

    NOTE: validation stops at the subcommand. Everything AFTER the
    subcommand is caller-constructed and NOT validated at this layer — e.g.
    ``["git", "diff", "--output=/tmp/pwn"]`` passes ``_validate_argv``.
    Callers (the tool cores below) must therefore build argv only from
    fixed literals plus values they have validated themselves (see the
    ``commit_range`` leading-dash/regex checks in :func:`git_diff`) and
    must never splice raw model-controlled strings into flag positions.

    Raises:
        LocalToolError: argv validation failure, git unavailable, or timeout.
    """
    _validate_argv(argv)
    resolved_executable = executable or (Path(found) if (found := shutil.which('git')) is not None else None)
    if resolved_executable is None:
        raise LocalToolError('git is not available on this system')
    resolved_executable = resolved_executable.resolve()
    if not isinstance(max_output_bytes, int) or isinstance(max_output_bytes, bool) or max_output_bytes <= 0:
        max_output_bytes = GIT_MAX_OUTPUT_BYTES
    process = subprocess.Popen([str(resolved_executable), *argv[1:]], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=_git_environment(), start_new_session=own_process_group and os.name == 'posix')
    results: dict[str, tuple[bytes, bool]] = {}
    done: queue.Queue[tuple[str, tuple[bytes, bool]]] = queue.Queue()

    def _reader(name: str, stream) -> None:
        try:
            done.put((name, _read_bounded_stream(stream, max_output_bytes)))
        except Exception:
            done.put((name, (b'', False)))
    for name, stream in (('stdout', process.stdout), ('stderr', process.stderr)):
        threading.Thread(target=_reader, args=(name, stream), daemon=True).start()
    deadline = time.monotonic() + float(timeout)
    while len(results) < 2:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _kill_process(process, own_process_group=own_process_group)
            with contextlib.suppress(Exception):
                process.wait()
            raise LocalToolError(f'git command timed out after {timeout} seconds: {list(argv)}')
        try:
            name, data = done.get(timeout=min(remaining, 0.05))
        except queue.Empty:
            continue
        results[name] = data
        if data[1]:
            _kill_process(process, own_process_group=own_process_group)
    with contextlib.suppress(Exception):
        process.wait(timeout=REPOSITORY_DISCOVERY_TIMEOUT_SECONDS)
    stdout_bytes, stdout_truncated = results.get('stdout', (b'', False))
    stderr_bytes, stderr_truncated = results.get('stderr', (b'', False))
    stdout = stdout_bytes.decode('utf-8', errors='replace')
    stderr = stderr_bytes.decode('utf-8', errors='replace')
    if stdout_truncated:
        stdout += GIT_TRUNCATED_MARKER
    if stderr_truncated:
        stderr += GIT_TRUNCATED_MARKER
    return GitCommandResult(argv=list(argv), returncode=int(process.returncode if process.returncode is not None else -1), stdout=stdout, stderr=stderr, timed_out=False, truncated=stdout_truncated or stderr_truncated)

def _kill_process(process: subprocess.Popen, *, own_process_group: bool=True) -> None:
    """Kill the whole process group on POSIX; fall back to the direct child."""
    if own_process_group and os.name == 'posix':
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(process.pid, signal.SIGKILL)
        return
    with contextlib.suppress(ProcessLookupError):
        process.kill()
_GLOB_METACHARACTERS = '\\*?['

def _glob_escape(value: str) -> str:
    """Escape ``value`` so a ``:(...,glob)`` pathspec matches it literally."""
    return ''.join((f'\\{char}' if char in _GLOB_METACHARACTERS else char for char in value))

def _literal_pathspec(relative_posix: str) -> str:
    """Render a repo-relative path as a pathspec that CANNOT carry magic.

    A pathspec is not a path. ``--`` ends git's OPTION parsing, not its
    pathspec-magic parsing, so a repository file legitimately named
    ``:(exclude)notes.txt`` -- resolved, confined and denylist-checked by
    the choke point exactly like any other filename -- inverted the scope
    of the diff it was spliced into and returned the rest of the
    repository, ``~/.ssh/id_rsa`` included (measured; TASK-19632).
    ``:(literal)`` disables magic AND wildcard interpretation for the
    remainder of the element, so the value is matched as the byte string
    it is.

    ``:(literal).`` behaves identically to a bare ``.`` (verified), so the
    repo-root case needs no special handling.

    Deliberately NOT given the ``icase`` magic the EXCLUSIONS carry. This
    pathspec SCOPES the output; theirs DENY from it, and the two fail in
    opposite directions -- exactly the asymmetry TASK-19800 records for
    ``_compare_key`` versus confinement. Folding case here would ADD files
    to what the model gets back (a ``README`` alongside the ``readme`` it
    asked for); folding it there only ever removes more.
    """
    return f':(literal){relative_posix}'

def _denylist_pathspecs(repo_root: Path, context: SensitivePathContext | None=None, *, workspace_root: Path | None=None, sensitive_exclusions: tuple[SensitiveExclusion, ...] | None=None) -> tuple[str, ...]:
    """Render the sensitive-path denylist as git exclude pathspecs.

    The TASK-19632 fix, in one place. ``git_status``/``git_diff`` hand a
    whole repository to git and return what comes back, so a path the
    model never named -- ``~/.ssh/id_rsa`` under a ``$HOME``-rooted
    workspace -- never reached ``Utils/sensitive_paths.py`` at all.

    Excluding by PATHSPEC rather than filtering git's output is the
    deliberate choice: git stays the authority on what matches, and no
    unified-diff or porcelain text is ever parsed to decide what to
    withhold -- a half-parsed diff is worse than none. Direct callers
    recompute the live denylist on each call. The pinned helper instead
    consumes the bounded snapshot admitted by its parent before the helper
    environment is scrubbed.

    Each :class:`~tldw_chatbook.Utils.sensitive_paths.SensitiveExclusion`
    kind maps to the pathspec form that expresses exactly that rule and
    no more, which is what keeps a legitimate diff intact:

    * ``subtree``/``file`` -> ``:(exclude,literal,icase)<rel>``. Literal
      magic, so a real filename containing ``*`` excludes only itself.
    * ``direct_children`` -> ``:(exclude,glob,icase)<rel>/*``. Under
      ``glob`` magic ``*`` does not cross ``/``, so this refuses the
      container's direct child FILES and leaves its subdirectories fully
      visible -- the same distinction ``is_sensitive_path``'s own
      container rule draws (``tool_sandbox/`` stays diffable; a loose file
      beside it does not).
    * ``name`` -> ``:(exclude,glob,icase)**/<name>``, which matches at
      every depth INCLUDING the repository root (verified).

    Every one of them carries ``icase``, for the reason TASK-19800 gives
    for folding the denylist itself: macOS and Windows filesystems are
    case-insensitive by default, git records whatever spelling a path was
    added under, and a denial that misses ``.SSH/id_rsa`` because the
    denylist says ``.ssh`` is a leak. Folding an EXCLUSION only ever
    removes more from the output, so it fails in the cheap direction --
    unlike folding a scoping pathspec (see ``_literal_pathspec``) or a
    confinement check.

    An unrecognized kind raises rather than being skipped: a denial added
    to the denylist that this renderer does not understand must fail the
    call, not silently pass through.

    Args:
        repo_root: The already-resolved repository root; every pathspec
            is rendered relative to it; git resolves pathspecs against the
            process's working directory.
        context: Optional pre-resolved ``SensitivePathContext``, so one
            tool call resolves the denylist once.

    Returns:
        Exclude pathspecs, possibly empty of location-based entries but
        never empty overall (the name rule always applies). They may be
        passed as the ONLY pathspecs after ``--``: git applies an
        exclude-only list to the whole tree (verified).

    Raises:
        LocalToolError: The repository root is itself a protected path,
            or the denylist produced an exclusion kind this renderer does
            not know how to express.
    """
    if sensitive_exclusions is None:
        exclusions = sensitive_exclusions_under(repo_root, context=context)
    else:
        if workspace_root is None:
            raise LocalToolError('workspace root is required for admitted exclusions')
        exclusions = _repo_relative_exclusions(Path(workspace_root).resolve(), repo_root, sensitive_exclusions)
    specs: list[str] = []
    for kind, value in exclusions:
        if kind in {'subtree', 'file'}:
            if not value:
                raise LocalToolError(f'repository root ({repo_root}) is a protected path; refusing')
            specs.append(f':(exclude,literal,icase){value}')
        elif kind == 'direct_children':
            prefix = f'{_glob_escape(value)}/' if value else ''
            specs.append(f':(exclude,glob,icase){prefix}*')
        elif kind == 'name':
            specs.append(f':(exclude,glob,icase)**/{_glob_escape(value)}')
        else:
            raise LocalToolError(f'unsupported sensitive-path exclusion kind: {kind!r}')
    return tuple(specs)

def _repo_relative_exclusions(workspace_root: Path, repo_root: Path, exclusions: tuple[SensitiveExclusion, ...]) -> tuple[SensitiveExclusion, ...]:
    """Translate parent-admitted workspace exclusions to repository-relative."""
    repo_parts = tuple((part.casefold() for part in repo_root.relative_to(workspace_root).parts))
    translated: list[SensitiveExclusion] = []
    for exclusion in exclusions:
        if exclusion.kind == 'name':
            translated.append(exclusion)
            continue
        value_parts = PurePosixPath(exclusion.value).parts if exclusion.value else ()
        folded_value = tuple((part.casefold() for part in value_parts))
        if folded_value[:len(repo_parts)] == repo_parts:
            relative_parts = value_parts[len(repo_parts):]
            translated.append(SensitiveExclusion(exclusion.kind, PurePosixPath(*relative_parts).as_posix() if relative_parts else ''))
            continue
        if repo_parts[:len(folded_value)] != folded_value:
            continue
        if exclusion.kind in {'subtree', 'file'}:
            translated.append(SensitiveExclusion(exclusion.kind, ''))
        elif exclusion.kind == 'direct_children' and len(repo_parts) == len(folded_value):
            translated.append(SensitiveExclusion('direct_children', ''))
    return tuple(translated)

def prepare_repository(workspace_root: Path, path: str='.', *, context: SensitivePathContext | None=None, sensitive_exclusions: tuple[SensitiveExclusion, ...] | None=None, executable: Path | None=None, own_process_group: bool=True) -> Path:
    """Resolve the git repo root for ``path``, confined to ``workspace_root``.

    Refuses (LocalToolError) when git is unavailable, ``path`` escapes the
    workspace, no repository is found, or the discovered repo root is ABOVE
    the workspace root (the repo root must be the workspace root or inside
    it — a workspace nested inside a repo is refused so the model cannot
    read repo state outside the confinement).

    Args:
        workspace_root: The confinement root ``path`` must resolve inside.
        path: The workspace-relative (or absolute-but-confined) location to
            discover a repository from; ``"."`` (the default) discovers
            from ``workspace_root`` itself.
        context: Optional pre-resolved ``SensitivePathContext``. Passed
            through to both denylist checks this function makes (the
            ``path`` argument, via ``resolve_workspace_path``, and the
            DISCOVERED repo root below) so a caller that also needs the
            same context for e.g. ``_denylist_pathspecs`` resolves the
            ~11 config accessors behind the denylist once per tool call
            instead of once per check. ``None`` resolves fresh each time —
            still enforces the denylist, just not shared.

    Returns:
        The resolved repository root path.
    """
    if executable is None and shutil.which('git') is None:
        raise LocalToolError('git is not available on this system')
    workspace_root = Path(workspace_root).resolve()
    target = _resolve_git_path(path, workspace_root, context=context, sensitive_exclusions=sensitive_exclusions)
    result = run_git(['git', 'rev-parse', '--show-toplevel'], cwd=_git_cwd(workspace_root, target, own_process_group=own_process_group), executable=executable, own_process_group=own_process_group, timeout=REPOSITORY_DISCOVERY_TIMEOUT_SECONDS)
    if result.returncode != 0:
        gist = (result.stderr or result.stdout).strip().splitlines()
        gist_text = gist[0][:200] if gist else 'unknown error'
        combined = (result.stderr + result.stdout).lower()
        if 'not a git repository' in combined:
            raise LocalToolError(f"'{path}' is not a git repository: {gist_text}")
        raise LocalToolError(f'git repository discovery failed: {gist_text}')
    if result.truncated:
        raise LocalToolError('git returned truncated repository information')
    first_line = result.stdout.strip().splitlines()
    if not first_line:
        raise LocalToolError('git returned invalid repository information')
    repo_root = Path(first_line[0]).expanduser()
    if not repo_root.is_absolute():
        raise LocalToolError('git returned invalid repository information')
    repo_root = repo_root.resolve()
    if not (repo_root == workspace_root or workspace_root in repo_root.parents):
        raise LocalToolError(f'repository root ({repo_root}) is outside the workspace root ({workspace_root}); refusing')
    if sensitive_exclusions is None:
        repo_is_sensitive = is_sensitive_path(repo_root, context=context)
    else:
        repo_relative = repo_root.relative_to(workspace_root)
        repo_is_sensitive = not _relative_target_is_safe(repo_relative, workspace_root, sensitive_exclusions, is_directory=True)
    if repo_is_sensitive:
        raise LocalToolError(f'repository root ({repo_root}) is a protected path; refusing')
    return repo_root
GIT_LOG_DEFAULT_COUNT = 20
GIT_LOG_MAX_COUNT = 100
GIT_STATUS_MAX_ENTRIES = 200
GIT_BLAME_MAX_LINES = 500
_COMMIT_RANGE_PATTERN = re.compile('^[A-Za-z0-9._/~^-]+$')
_EMAIL_PATTERN = re.compile('<[^<>\\s@]+@[^<>\\s@]+>|\\b\\S+@\\S+\\b')

def _stderr_gist(result: GitCommandResult) -> str:
    for line in (result.stderr or result.stdout).strip().splitlines():
        if line.strip():
            return line.strip()[:200]
    return f'exit code {result.returncode}'

def _run_git_checked(argv: list[str], *, subcommand: str, cwd: Path, executable: Path | None, own_process_group: bool) -> GitCommandResult:
    result = run_git(argv, cwd=cwd, executable=executable, own_process_group=own_process_group)
    if result.returncode != 0 and (not result.truncated):
        raise LocalToolError(f'git {subcommand} failed: {_stderr_gist(result)}')
    return result

def _git_cwd(workspace_root: Path, target: Path, *, own_process_group: bool) -> Path:
    """Return absolute compatibility cwd or helper-root-relative cwd."""
    if own_process_group:
        return target
    relative = target.relative_to(workspace_root)
    return relative if relative.parts else Path('.')

def _resolve_git_path(path: str, workspace_root: Path, *, context: SensitivePathContext | None, sensitive_exclusions: tuple[SensitiveExclusion, ...] | None) -> Path:
    """Resolve a Git path using either live or parent-admitted exclusions."""
    workspace_root = Path(workspace_root).resolve()
    if sensitive_exclusions is None:
        return resolve_workspace_path(path, workspace_root, context=context)
    candidate = Path(path)
    try:
        if candidate.is_absolute():
            relative = candidate.resolve().relative_to(workspace_root)
        else:
            relative = candidate
        target = workspace_root / relative
        if not _relative_target_is_safe(relative, workspace_root, sensitive_exclusions, is_directory=target.is_dir()):
            raise LocalToolError(f"path '{path}' is outside the workspace or protected")
        return target.resolve()
    except (OSError, ValueError):
        raise LocalToolError(f"path '{path}' is outside the workspace or protected") from None

def _repo_relative_path(workspace_root: Path, repo_root: Path, path: str, *, context: SensitivePathContext | None=None, sensitive_exclusions: tuple[SensitiveExclusion, ...] | None=None) -> str:
    """Resolve ``path`` confined to the workspace, rendered repo-relative.

    Args:
        workspace_root: The confinement root ``path`` must resolve inside.
        repo_root: The already-discovered repository root.
        path: The workspace-relative (or absolute-but-confined) path.
        context: Optional pre-resolved ``SensitivePathContext``, threaded
            through to ``resolve_workspace_path`` so a caller that has
            already resolved one for the same tool call does not pay for
            it again here.
    """
    resolved = _resolve_git_path(path, Path(workspace_root).resolve(), context=context, sensitive_exclusions=sensitive_exclusions)
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError:
        raise LocalToolError(f"path '{path}' is outside the repository root ({repo_root})") from None
    return relative.as_posix() or '.'

def _prepare_for_path(workspace_root: Path, path: str | None, *, context: SensitivePathContext | None=None, sensitive_exclusions: tuple[SensitiveExclusion, ...] | None=None, executable: Path | None=None, own_process_group: bool=True) -> Path:
    """Repo discovery that tolerates ``path`` being a file (uses its parent).

    Args:
        workspace_root: The confinement root ``path`` must resolve inside.
        path: The workspace-relative (or absolute-but-confined) path to
            discover a repository from, or ``None`` to discover from
            ``workspace_root`` itself.
        context: Optional pre-resolved ``SensitivePathContext``, threaded
            through to every denylist check this function (and
            ``prepare_repository`` beneath it) makes, so a caller that
            resolves one per tool call — rather than letting each check
            resolve its own — pays the ~11 config-accessor cost once.
    """
    if path is None:
        return prepare_repository(workspace_root, '.', context=context, sensitive_exclusions=sensitive_exclusions, executable=executable, own_process_group=own_process_group)
    resolved = _resolve_git_path(path, Path(workspace_root).resolve(), context=context, sensitive_exclusions=sensitive_exclusions)
    discovery = resolved if resolved.is_dir() else resolved.parent
    relative = os.path.relpath(discovery, Path(workspace_root).resolve())
    return prepare_repository(workspace_root, relative, context=context, sensitive_exclusions=sensitive_exclusions, executable=executable, own_process_group=own_process_group)

def _sanitize_author_name(value: str) -> str:
    return ' '.join(_EMAIL_PATTERN.sub('', value).split())

def _nul_records(stdout: str) -> list[str]:
    return [record for record in stdout.split('\x00') if record]

def git_status(workspace_root: Path, path: str='.', *, sensitive_exclusions: tuple[SensitiveExclusion, ...] | None=None, executable: Path | None=None, own_process_group: bool=True) -> str:
    """Branch header + staged/unstaged/untracked/conflicted entries as text.

    Sync adaptation of the reference's ``_execute_status`` (:570): porcelain
    v2 ``-z`` output with the ``--branch`` header, parsed and rendered as
    ``category: XY path`` lines capped at ``GIT_STATUS_MAX_ENTRIES``.

    Denylisted paths are excluded by pathspec (``_denylist_pathspecs``):
    this tool named ``~/.ssh/id_rsa`` on a dirty ``$HOME``-rooted
    workspace (TASK-19632). Existence and a name are all a status entry
    carries, so excluding them is the whole refusal.

    Args:
        workspace_root: The confinement root ``path`` must resolve inside.
        path: Used ONLY to discover which repository to report on (a file
            or directory anywhere inside the target repo works, since
            discovery walks up to the repo root). It is NOT applied as a
            scoping pathspec: unlike ``git_diff``/``git_log``, this
            function's argv carries no positive pathspec for ``path``, so
            the status returned always covers the WHOLE repository —
            asking for a subdirectory's status still returns every
            changed file in the repo, not just that subdirectory's.

    Returns:
        The branch header line, one ``category: XY path`` (or
        ``untracked: path``) line per entry up to ``GIT_STATUS_MAX_ENTRIES``,
        ``"(working tree clean)"`` when there are none, and a truncation
        note appended if the repository has more entries than the cap.
    """
    workspace_root = Path(workspace_root).resolve()
    context = None if sensitive_exclusions is not None else resolve_sensitive_context()
    repo_root = _prepare_for_path(workspace_root, path, context=context, sensitive_exclusions=sensitive_exclusions, executable=executable, own_process_group=own_process_group)
    result = _run_git_checked(['git', '--no-pager', 'status', '--porcelain=v2', '-z', '--branch', '--untracked-files=all', '--', *_denylist_pathspecs(repo_root, context=context, workspace_root=workspace_root, sensitive_exclusions=sensitive_exclusions)], subcommand='status', cwd=_git_cwd(workspace_root, repo_root, own_process_group=own_process_group), executable=executable, own_process_group=own_process_group)
    branch, entries, truncated = _parse_status_porcelain_v2(result.stdout, limit=GIT_STATUS_MAX_ENTRIES)
    lines = [_format_branch_header(branch)]
    lines.extend((_format_status_entry(entry) for entry in entries))
    if not entries:
        lines.append('(working tree clean)')
    if truncated:
        lines.append('… (more entries, truncated)')
    return '\n'.join(lines)

def _parse_status_porcelain_v2(stdout: str, *, limit: int) -> tuple[dict[str, object], list[dict[str, object]], bool]:
    branch: dict[str, object] = {'branch': None, 'upstream': None, 'ahead': None, 'behind': None}
    entries: list[dict[str, object]] = []
    total = 0
    for record in _nul_records(stdout):
        if record.startswith('# '):
            _parse_status_branch_header(record, branch)
            continue
        if record.startswith('! '):
            continue
        entry = _parse_status_entry(record)
        if entry is None:
            continue
        total += 1
        if len(entries) < limit:
            entries.append(entry)
    return (branch, entries, total > limit)

def _parse_status_branch_header(record: str, branch: dict[str, object]) -> None:
    if record.startswith('# branch.head '):
        value = record.removeprefix('# branch.head ').strip()
        branch['branch'] = None if value == '(detached)' else value or None
        return
    if record.startswith('# branch.upstream '):
        branch['upstream'] = record.removeprefix('# branch.upstream ').strip() or None
        return
    if record.startswith('# branch.ab '):
        for part in record.removeprefix('# branch.ab ').split():
            if part.startswith('+'):
                with contextlib.suppress(ValueError):
                    branch['ahead'] = int(part[1:])
            elif part.startswith('-'):
                with contextlib.suppress(ValueError):
                    branch['behind'] = int(part[1:])

def _parse_status_entry(record: str) -> dict[str, object] | None:
    if record.startswith('? '):
        path = record[2:].strip()
        if not path:
            return None
        return {'path': path, 'xy': '??', 'category': 'untracked'}
    if record.startswith('1 '):
        parts = record.split(' ', 8)
        if len(parts) < 9:
            return None
        return _status_entry_from_xy(parts[1], parts[8])
    if record.startswith('2 '):
        parts = record.split(' ', 9)
        if len(parts) < 10:
            return None
        return _status_entry_from_xy(parts[1], parts[9])
    if record.startswith('u '):
        parts = record.split(' ', 10)
        if len(parts) < 11 or not parts[10].strip():
            return None
        return {'path': parts[10].strip(), 'xy': parts[1], 'category': 'conflicted'}
    return None

def _status_entry_from_xy(xy: str, path_raw: str) -> dict[str, object] | None:
    path = path_raw.strip()
    if not path or len(xy) < 2:
        return None
    staged = xy[0] not in {'.', '?', '!'}
    unstaged = xy[1] not in {'.', '?', '!'}
    if staged and unstaged:
        category = 'staged+unstaged'
    elif staged:
        category = 'staged'
    elif unstaged:
        category = 'unstaged'
    else:
        category = 'clean'
    return {'path': path, 'xy': xy, 'category': category}

def _format_branch_header(branch: dict[str, object]) -> str:
    name = branch.get('branch') or '(detached)'
    extras: list[str] = []
    if branch.get('upstream'):
        extras.append(f"upstream: {branch['upstream']}")
    if branch.get('ahead') is not None:
        extras.append(f"ahead: {branch['ahead']}")
    if branch.get('behind') is not None:
        extras.append(f"behind: {branch['behind']}")
    suffix = f" ({', '.join(extras)})" if extras else ''
    return f'branch: {name}{suffix}'

def _format_status_entry(entry: dict[str, object]) -> str:
    category = entry['category']
    if category == 'untracked':
        return f"untracked: {entry['path']}"
    return f"{category}: {entry['xy']} {entry['path']}"

def git_branches(workspace_root: Path, *, sensitive_exclusions: tuple[SensitiveExclusion, ...] | None=None, executable: Path | None=None, own_process_group: bool=True) -> str:
    """Verbose branch list with the current branch marked by ``*``.

    Sync adaptation of the reference's ``_execute_branches`` (:621).
    """
    workspace_root = Path(workspace_root).resolve()
    context = None if sensitive_exclusions is not None else resolve_sensitive_context()
    repo_root = prepare_repository(workspace_root, '.', context=context, sensitive_exclusions=sensitive_exclusions, executable=executable, own_process_group=own_process_group)
    result = _run_git_checked(['git', '--no-pager', 'branch', '--format=%(HEAD)%00%(refname:short)%00%(upstream:short)%00%(objectname)'], subcommand='branch', cwd=_git_cwd(workspace_root, repo_root, own_process_group=own_process_group), executable=executable, own_process_group=own_process_group)
    lines: list[str] = []
    for record in result.stdout.splitlines():
        if not record:
            continue
        parts = record.split('\x00')
        if len(parts) < 4:
            continue
        marker, name, upstream, commit = (part.strip() for part in parts[:4])
        if not name:
            continue
        extras: list[str] = []
        if commit:
            extras.append(commit[:12])
        if upstream:
            extras.append(f'upstream: {upstream}')
        suffix = f" ({', '.join(extras)})" if extras else ''
        lines.append(f'* {name}{suffix}' if marker == '*' else f'  {name}{suffix}')
    return '\n'.join(lines) if lines else '(no branches)'

def git_log(workspace_root: Path, *, count: int=GIT_LOG_DEFAULT_COUNT, path: str | None=None, sensitive_exclusions: tuple[SensitiveExclusion, ...] | None=None, executable: Path | None=None, own_process_group: bool=True) -> str:
    """Bounded commit log, newest first; ``count`` is clamped to 1..100.

    Sync adaptation of the reference's ``_execute_log`` (:831). Deviation:
    ``count`` defaults to 20 here (the reference has no default and falls
    back to its max-100 when no limit is given).

    Deliberately NOT given the denylist exclusions ``git_status``/
    ``git_diff`` carry: the ``--format`` below emits commit metadata only
    -- no paths, no content -- so this tool was measured leaking nothing,
    and excluding denied paths here would silently drop commits from a
    legitimate history instead of protecting anything (TASK-19632). Its
    ``path`` pathspec IS rendered literally, for the same reason
    ``git_diff``'s is: the value is a model-supplied filename and a bare
    one would be parsed as pathspec magic.

    Args:
        workspace_root: The confinement root ``path`` must resolve inside.
        count: Maximum number of commits to return, clamped to 1..100.
        path: When given, scopes the log to commits touching this file or
            directory (rendered as a literal pathspec — see the note
            above); ``None`` (the default) returns the log for the whole
            repository ``path`` (or ``workspace_root`` when ``path`` is
            also omitted) discovers.

    Returns:
        One ``short_hash date author: subject`` line per commit, newest
        first, or ``"(no commits)"`` when there are none.
    """
    count = min(max(int(count), 1), GIT_LOG_MAX_COUNT)
    workspace_root = Path(workspace_root).resolve()
    context = None if sensitive_exclusions is not None else resolve_sensitive_context()
    repo_root = _prepare_for_path(workspace_root, path, context=context, sensitive_exclusions=sensitive_exclusions, executable=executable, own_process_group=own_process_group)
    argv = ['git', '--no-pager', 'log', '--format=%H%x1f%h%x1f%an%x1f%aI%x1f%s%x1e', '-n', str(count)]
    if path is not None:
        argv.extend(['--', _literal_pathspec(_repo_relative_path(workspace_root, repo_root, path, context=context, sensitive_exclusions=sensitive_exclusions))])
    result = _run_git_checked(argv, subcommand='log', cwd=_git_cwd(workspace_root, repo_root, own_process_group=own_process_group), executable=executable, own_process_group=own_process_group)
    lines: list[str] = []
    for record in result.stdout.split('\x1e'):
        fields = record.strip('\n').split('\x1f', 4)
        if len(fields) < 5:
            continue
        _commit_hash, short_hash, author_name, author_date, subject = fields
        lines.append(f'{short_hash} {author_date} {_sanitize_author_name(author_name)}: {subject}')
    return '\n'.join(lines) if lines else '(no commits)'

def git_diff(workspace_root: Path, *, staged: bool=False, commit_range: str | None=None, path: str | None=None, stat: bool=False, sensitive_exclusions: tuple[SensitiveExclusion, ...] | None=None, executable: Path | None=None, own_process_group: bool=True) -> str:
    """Unified diff of the worktree (default) or the index (``staged=True``).

    Sync adaptation of the reference's ``_execute_diff`` (:709) +
    ``_run_diff_command`` (:1049-1080 — ``--no-ext-diff``/``--no-textconv``/
    ``--no-color`` ported). Disclosed deviations: adds ``commit_range``
    (regex-validated before entering argv) and ``stat`` modes; the
    reference's third ``working_tree`` scope is omitted.

    Denylisted paths are excluded by pathspec (``_denylist_pathspecs``)
    in every mode — worktree, index and ``commit_range`` alike, since the
    leak this closes was reachable from a CLEAN worktree by reading the
    credential out of history (TASK-19632). Exclusions apply whether or
    not the caller supplied ``path``: a denylisted ``path`` is refused by
    the choke point, but the leak was in the no-``path`` case, where the
    model names nothing and git enumerates the repository.

    Nothing announces that an exclusion took effect, deliberately. The
    only honest note would state that this repository contains a protected
    path — which is the same disclosure ``stat=True``/``git_status`` were
    leaking. A model that names the path still gets a "protected path"
    refusal, which is the case where the information is actionable.

    Args:
        workspace_root: The confinement root ``path`` must resolve inside.
        staged: When True, diff the index against ``HEAD`` (``--cached``)
            instead of the worktree.
        commit_range: When given, diff across this ref/range instead of
            against the worktree or index (validated against
            ``^[A-Za-z0-9._/~^-]+$`` and refused if it starts with ``-``).
        path: When given, scopes the diff to this file or directory
            (rendered as a literal pathspec, so a magic-shaped filename is
            matched literally); ``None`` (the default) diffs the whole
            repository ``path`` (or ``workspace_root`` when ``path`` is
            also omitted) discovers.
        stat: When True, return a ``--stat`` summary instead of a unified
            patch.

    Returns:
        The unified diff (or ``--stat`` summary) text, or ``"(no
        changes)"`` when the diff is empty.
    """
    if commit_range is not None:
        if commit_range.startswith('-') or not _COMMIT_RANGE_PATTERN.match(commit_range):
            raise LocalToolError(f"invalid commit_range {commit_range!r}: must be a ref/range matching [A-Za-z0-9._/~^-] and not start with '-'")
    workspace_root = Path(workspace_root).resolve()
    context = None if sensitive_exclusions is not None else resolve_sensitive_context()
    repo_root = _prepare_for_path(workspace_root, path, context=context, sensitive_exclusions=sensitive_exclusions, executable=executable, own_process_group=own_process_group)
    argv = ['git', '--no-pager', 'diff', '--no-ext-diff', '--no-textconv', '--no-color']
    if stat:
        argv.append('--stat')
    else:
        argv.append('--unified=3')
    if staged:
        argv.append('--cached')
    if commit_range is not None:
        argv.append(commit_range)
    pathspecs: list[str] = []
    if path is not None:
        pathspecs.append(_literal_pathspec(_repo_relative_path(workspace_root, repo_root, path, context=context, sensitive_exclusions=sensitive_exclusions)))
    pathspecs.extend(_denylist_pathspecs(repo_root, context=context, workspace_root=workspace_root, sensitive_exclusions=sensitive_exclusions))
    argv.extend(['--', *pathspecs])
    result = _run_git_checked(argv, subcommand='diff', cwd=_git_cwd(workspace_root, repo_root, own_process_group=own_process_group), executable=executable, own_process_group=own_process_group)
    return result.stdout if result.stdout.strip() else '(no changes)'

def git_blame(workspace_root: Path, path: str, *, start_line: int | None=None, end_line: int | None=None, sensitive_exclusions: tuple[SensitiveExclusion, ...] | None=None, executable: Path | None=None, own_process_group: bool=True) -> str:
    """Per-line blame for ``path``; optional 1-based inclusive line range.

    Sync adaptation of the reference's ``_execute_blame`` (:892) — line
    porcelain parse — except the ``-L`` range is optional here (omitted when
    neither bound is given) and the range is capped at
    ``GIT_BLAME_MAX_LINES`` lines.
    """
    workspace_root = Path(workspace_root).resolve()
    context = None if sensitive_exclusions is not None else resolve_sensitive_context()
    resolved = _resolve_git_path(path, Path(workspace_root).resolve(), context=context, sensitive_exclusions=sensitive_exclusions)
    if not resolved.is_file():
        raise LocalToolError(f'file not found: {path}')
    repo_root = _prepare_for_path(workspace_root, path, context=context, sensitive_exclusions=sensitive_exclusions, executable=executable, own_process_group=own_process_group)
    try:
        repo_relative = resolved.relative_to(repo_root).as_posix()
    except ValueError:
        raise LocalToolError(f"path '{path}' is outside the repository root ({repo_root})") from None
    argv = ['git', '--no-pager', 'blame', '--line-porcelain', '--no-textconv']
    if start_line is not None or end_line is not None:
        start = int(start_line) if start_line is not None else 1
        if start < 1:
            raise LocalToolError(f'start_line must be >= 1, got {start}')
        end = int(end_line) if end_line is not None else start + GIT_BLAME_MAX_LINES - 1
        if end < start:
            raise LocalToolError(f'end_line ({end}) is before start_line ({start})')
        end = min(end, start + GIT_BLAME_MAX_LINES - 1)
        argv.extend(['-L', f'{start},{end}'])
    argv.extend(['--', repo_relative])
    result = _run_git_checked(argv, subcommand='blame', cwd=_git_cwd(workspace_root, repo_root, own_process_group=own_process_group), executable=executable, own_process_group=own_process_group)
    lines = [f'{ln}: {author}: {text}' for ln, author, text in _parse_blame(result.stdout)]
    return '\n'.join(lines) if lines else '(no blame output)'

def _parse_blame(stdout: str) -> list[tuple[int, str, str]]:
    """Parse ``blame --line-porcelain`` into (line_number, author, text)."""
    lines: list[tuple[int, str, str]] = []
    current: dict[str, object] | None = None
    commit_metadata: dict[str, dict[str, object]] = {}
    for raw_line in stdout.splitlines():
        if raw_line.startswith('\t'):
            if current is None:
                continue
            author = _sanitize_author_name(str(current.get('author_name') or ''))
            lines.append((int(current['line_number']), author, raw_line[1:]))
            current = None
            continue
        header = _parse_blame_header(raw_line)
        if header is not None:
            cached = commit_metadata.get(str(header['commit']))
            if cached:
                header.update(cached)
            current = header
            continue
        if current is None:
            continue
        if raw_line.startswith('author '):
            author_name = raw_line.removeprefix('author ')
            current['author_name'] = author_name
            commit_metadata.setdefault(str(current['commit']), {})['author_name'] = author_name
    return lines

def _parse_blame_header(raw_line: str) -> dict[str, object] | None:
    parts = raw_line.split()
    if len(parts) < 3:
        return None
    commit_hash = parts[0]
    if len(commit_hash) < 8 or not all((character in '0123456789abcdefABCDEF' for character in commit_hash)):
        return None
    with contextlib.suppress(ValueError):
        return {'commit': commit_hash, 'line_number': int(parts[2]), 'author_name': None}
    return None


# ===========================================================================
# Section: tldw_chatbook.Tools.patch_tool_impls (extracted from the live module by the
# builder; regenerate rather than editing)
# ===========================================================================
"""fs_patch core: unified-diff parser/applier + workspace wrapper.

The parser/applier below (``parse_unified_diff``, ``apply_patch_to_text``
and their helpers/dataclasses, ``FilesystemPatchError``) is a near-verbatim
port of tldw_server's
tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_diff.py
@ 5605b9d9906322c2e6b5342b48c391ae674d315e
(https://github.com/rmusser01/tldw_server, GPL-3.0-only). Reason codes are
kept exactly so error handling stays in lockstep with the reference.

The workspace wrapper (``patch_files``) is written fresh for tldw_chatbook:
it enforces ADR-032 confinement via resolve_workspace_path and phase-2
write discipline (encode-before-write, newline-preserving reads), and
translates FilesystemPatchError into the shared LocalToolError.

Deviations from reference (deliberate fixes; reference kept otherwise):

1. Pure-insertion hunks (``@@ -N,0 +M,K @@``, N>0) apply AFTER line N per
   unified-diff semantics (verified against GNU/BSD ``diff -U0`` +
   ``patch``). The reference used ``max(0, old_start - 1)`` for all hunks,
   inserting one line early with no context to catch it. Here,
   ``old_count == 0`` uses ``hunk_start = old_start``; ``old_count > 0``
   keeps ``old_start - 1``.
2. Real multi-file ``git diff`` output parses: in the per-file hunk-section
   loop, a line that is neither a hunk header nor a ``--- `` file header
   ends the section once at least one hunk has been parsed (so
   ``diff --git``/``index``/``new file mode`` preamble lines are skipped
   by the outer loop). With no hunks parsed yet it still raises
   ``invalid_diff``. The reference raised ``invalid_diff`` on any such
   line, making real git diffs unparseable.
3. ``_parse_hunk``'s body loop terminates when the header line counts are
   satisfied (accepting only the ``\\ No newline at end of file`` marker
   afterwards), instead of only on ``@@ ``/``--- `` sentinels. The
   reference misread a removal of content starting with ``-- `` (e.g. a
   SQL comment) as a file-header sentinel. A ``--- `` line followed by a
   ``+++ `` line is still treated as the next file's header pair, so
   truncated hunks keep raising ``invalid_hunk_line_count``.
4. A leading U+FEFF (BOM) is stripped from the diff text before parsing;
   the reference rejected BOM-prefixed diffs as ``invalid_diff``.
"""
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
PATCH_MAX_BYTES = 256 * 1024
PATCH_MAX_FILES = 20
PATCH_MAX_HUNKS = 200
PatchLineKind = Literal['context', 'add', 'remove']
PatchFileAction = Literal['modify', 'create']
_HUNK_HEADER = re.compile('^@@ -(\\d+)(?:,(\\d+))? \\+(\\d+)(?:,(\\d+))? @@(?: .*)?$')
_NO_NEWLINE_MARKER = '\\ No newline at end of file'
_HEADER_TIMESTAMP_METADATA = re.compile('\\s+\\d{4}-\\d{2}-\\d{2}(?:[ T]\\d{2}:\\d{2}:\\d{2}(?:\\.\\d+)?)?(?:\\s*(?:[+-]\\d{4}|[+-]\\d{2}:?\\d{2}|Z))?$')

class FilesystemPatchError(ValueError):
    """Raised when a unified diff cannot be parsed or applied safely."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code

@dataclass(frozen=True, slots=True)
class PatchHunkLine:
    """One context, addition, or removal line inside a unified-diff hunk."""
    kind: PatchLineKind
    text: str
    has_trailing_newline: bool = True

@dataclass(frozen=True, slots=True)
class PatchHunk:
    """Parsed unified-diff hunk with old and new line ranges."""
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: tuple[PatchHunkLine, ...]

@dataclass(frozen=True, slots=True)
class PatchFile:
    """One file-level patch from a unified diff."""
    old_path: str | None
    new_path: str | None
    action: PatchFileAction
    hunks: tuple[PatchHunk, ...]

def parse_unified_diff(diff_text: str, *, max_files: int, max_hunks: int, max_bytes: int) -> tuple[PatchFile, ...]:
    """Parse bounded unified diff text into file-level patch plans."""
    if not isinstance(diff_text, str):
        raise FilesystemPatchError('invalid_diff')
    diff_text = diff_text.removeprefix('\ufeff')
    if not diff_text.strip():
        raise FilesystemPatchError('invalid_diff')
    if len(diff_text.encode('utf-8')) > max(1, int(max_bytes)):
        raise FilesystemPatchError('diff_too_large')
    lines = diff_text.splitlines()
    files: list[PatchFile] = []
    hunk_count = 0
    index = 0
    while index < len(lines):
        if not lines[index].startswith('--- '):
            index += 1
            continue
        old_path = _parse_header_path(lines[index][4:])
        index += 1
        if index >= len(lines) or not lines[index].startswith('+++ '):
            raise FilesystemPatchError('invalid_diff')
        new_path = _parse_header_path(lines[index][4:])
        index += 1
        if old_path is None and new_path is None:
            raise FilesystemPatchError('invalid_patch_path')
        if new_path is None:
            raise FilesystemPatchError('delete_not_supported')
        if old_path is None:
            action: PatchFileAction = 'create'
        else:
            if old_path != new_path:
                raise FilesystemPatchError('rename_not_supported')
            action = 'modify'
        hunks: list[PatchHunk] = []
        while index < len(lines) and (not lines[index].startswith('--- ')):
            if not lines[index].startswith('@@ '):
                if hunks:
                    break
                raise FilesystemPatchError('invalid_diff')
            hunk, index = _parse_hunk(lines, index)
            hunks.append(hunk)
            hunk_count += 1
            if hunk_count > max(1, int(max_hunks)):
                raise FilesystemPatchError('diff_hunk_limit_exceeded')
        if not hunks:
            raise FilesystemPatchError('invalid_diff')
        files.append(PatchFile(old_path=old_path, new_path=new_path, action=action, hunks=tuple(hunks)))
        if len(files) > max(1, int(max_files)):
            raise FilesystemPatchError('diff_file_limit_exceeded')
    if not files:
        raise FilesystemPatchError('invalid_diff')
    return tuple(files)

def parse_patch_targets(diff_text: str) -> tuple[PatchFile, ...]:
    """Parse the bounded create/modify plans shared by preflight and execution."""
    return parse_unified_diff(diff_text, max_files=PATCH_MAX_FILES, max_hunks=PATCH_MAX_HUNKS, max_bytes=PATCH_MAX_BYTES)

def apply_patch_to_text(original: str, patch_file: PatchFile) -> str:
    """Apply one parsed file patch to original text without touching the filesystem."""
    original_lines = original.splitlines(keepends=True)
    newline = _detect_output_newline(original_lines)
    output: list[str] = []
    cursor = 0
    for hunk in patch_file.hunks:
        if hunk.old_count == 0:
            hunk_start = hunk.old_start
        else:
            hunk_start = hunk.old_start - 1
        hunk_start = max(0, hunk_start)
        if hunk_start < cursor or hunk_start > len(original_lines):
            raise FilesystemPatchError('patch_context_mismatch')
        output.extend(original_lines[cursor:hunk_start])
        cursor = hunk_start
        for hunk_line in hunk.lines:
            if hunk_line.kind == 'add':
                output.append(hunk_line.text)
                if hunk_line.has_trailing_newline:
                    output.append(newline)
                continue
            if cursor >= len(original_lines):
                raise FilesystemPatchError('patch_context_mismatch')
            if _line_body(original_lines[cursor]) != hunk_line.text:
                raise FilesystemPatchError('patch_context_mismatch')
            if _line_has_trailing_newline(original_lines[cursor]) != hunk_line.has_trailing_newline:
                raise FilesystemPatchError('patch_context_mismatch')
            if hunk_line.kind == 'context':
                output.append(original_lines[cursor])
            cursor += 1
    output.extend(original_lines[cursor:])
    return ''.join(output)

def _parse_hunk(lines: list[str], start_index: int) -> tuple[PatchHunk, int]:
    """Parse one unified-diff hunk and return the next unread line index."""
    match = _HUNK_HEADER.match(lines[start_index])
    if match is None:
        raise FilesystemPatchError('invalid_hunk_header')
    old_start = int(match.group(1))
    old_count = int(match.group(2) or '1')
    new_start = int(match.group(3))
    new_count = int(match.group(4) or '1')
    hunk_lines: list[PatchHunkLine] = []
    old_seen = 0
    new_seen = 0
    index = start_index + 1
    while index < len(lines):
        raw_line = lines[index]
        if raw_line == _NO_NEWLINE_MARKER:
            index += 1
            if not hunk_lines:
                raise FilesystemPatchError('invalid_no_newline_marker')
            previous = hunk_lines[-1]
            if not previous.has_trailing_newline:
                raise FilesystemPatchError('invalid_no_newline_marker')
            hunk_lines[-1] = PatchHunkLine(kind=previous.kind, text=previous.text, has_trailing_newline=False)
            continue
        if old_seen == old_count and new_seen == new_count:
            break
        if raw_line.startswith('@@ '):
            break
        if raw_line.startswith('--- ') and index + 1 < len(lines) and lines[index + 1].startswith('+++ '):
            break
        index += 1
        if not raw_line:
            raise FilesystemPatchError('invalid_hunk_line')
        prefix = raw_line[0]
        text = raw_line[1:]
        if prefix == ' ':
            hunk_lines.append(PatchHunkLine(kind='context', text=text))
            old_seen += 1
            new_seen += 1
        elif prefix == '-':
            hunk_lines.append(PatchHunkLine(kind='remove', text=text))
            old_seen += 1
        elif prefix == '+':
            hunk_lines.append(PatchHunkLine(kind='add', text=text))
            new_seen += 1
        else:
            raise FilesystemPatchError('invalid_hunk_line')
    if old_seen != old_count or new_seen != new_count:
        raise FilesystemPatchError('invalid_hunk_line_count')
    return (PatchHunk(old_start=old_start, old_count=old_count, new_start=new_start, new_count=new_count, lines=tuple(hunk_lines)), index)

def _parse_header_path(raw_path: str) -> str | None:
    """Normalize a unified-diff file header path while stripping safe metadata.

    Tab-separated metadata is removed first because GNU/Git-style diffs commonly
    place timestamps after a tab. When no tab exists, only a trailing
    timestamp-shaped suffix is stripped so paths containing spaces remain intact.
    Returns None for `/dev/null` create/delete sentinels.
    """
    candidate = raw_path.rstrip()
    if '\t' in candidate:
        candidate = candidate.split('\t', 1)[0]
    else:
        candidate = _strip_space_separated_header_metadata(candidate)
    candidate = candidate.strip()
    if candidate == '/dev/null':
        return None
    if candidate.startswith('a/') or candidate.startswith('b/'):
        candidate = candidate[2:]
    return _normalize_patch_path(candidate)

def _strip_space_separated_header_metadata(candidate: str) -> str:
    """Strip common space-separated timestamp metadata from a diff header path."""
    stripped = _HEADER_TIMESTAMP_METADATA.sub('', candidate).rstrip()
    return stripped or candidate

def _normalize_patch_path(raw_path: str) -> str:
    candidate = raw_path.strip().replace('\\', '/')
    if not candidate or candidate in {'.', '/'}:
        raise FilesystemPatchError('invalid_patch_path')
    if candidate.startswith('/') or candidate.startswith('//'):
        raise FilesystemPatchError('invalid_patch_path')
    if len(candidate) >= 2 and candidate[1] == ':' and candidate[0].isalpha():
        raise FilesystemPatchError('invalid_patch_path')
    parts = candidate.split('/')
    if any((part in {'', '.', '..'} for part in parts)):
        raise FilesystemPatchError('invalid_patch_path')
    return '/'.join(parts)

def _detect_output_newline(lines: list[str]) -> str:
    for line in lines:
        if line.endswith('\r\n'):
            return '\r\n'
        if line.endswith('\n'):
            return '\n'
        if line.endswith('\r'):
            return '\r'
    return '\n'

def _line_body(line: str) -> str:
    if line.endswith('\r\n'):
        return line[:-2]
    if line.endswith('\n') or line.endswith('\r'):
        return line[:-1]
    return line

def _line_has_trailing_newline(line: str) -> bool:
    """Return whether a split line retained an LF, CRLF, or CR terminator."""
    return line.endswith(('\n', '\r'))

def patch_files(diff_text: str, *, workspace_root: Path, dry_run: bool=False) -> str:
    """Parse and apply a unified diff to workspace files.

    Every target is confined AND denylist-checked via
    ``resolve_workspace_path`` — this tool owns no path resolution of its
    own, so it inherits the sensitive-path guard from that one choke point
    (TASK-19551; without it, a diff against ``mcp_permissions.json`` was a
    one-step permission-gate bypass). ``dry_run`` is checked identically:
    it still reads the target, and reporting "would patch
    mcp_permissions.json" is itself a disclosure.

    Modify targets must exist; create targets must not. dry_run validates
    and reports without writing. Returns a per-file summary ("patched X",
    "would patch X"). Files are applied sequentially; if a later file
    fails, earlier files stay patched — the error names the failed file so
    the model can recover (atomic multi-file apply is a documented non-goal
    for this phase).
    """
    try:
        parsed = parse_patch_targets(diff_text)
    except FilesystemPatchError as exc:
        raise LocalToolError(f'fs_patch failed [{exc.reason_code}]') from exc
    sensitive_ctx = resolve_sensitive_context()
    summaries: list[str] = []
    for patch_file in parsed:
        rel_path = patch_file.new_path
        assert rel_path is not None
        try:
            target = resolve_workspace_path(rel_path, workspace_root, intent='write', context=sensitive_ctx)
            _patch_relative_file(patch_file, target.relative_to(Path(workspace_root).resolve()), workspace=Path(workspace_root).resolve(), dry_run=dry_run)
        except FilesystemPatchError as exc:
            raise LocalToolError(f'fs_patch failed [{exc.reason_code}]: {rel_path}') from exc
        summaries.append(f"{('would patch' if dry_run else 'patched')} {rel_path}")
    return '\n'.join(summaries)

def patch_validated_files(plans: tuple[PatchFile, ...], *, root: PinnedWorkspaceRoot, dry_run: bool=False, content_stamps: bool=False) -> str:
    """Apply parent-admitted plans through one retained workspace root pin.

    Args:
        plans: Parent-admitted patch plans, in order.
        root: The retained workspace root pin every target resolves through.
        dry_run: Preview only; nothing is written and no stamp lines are
            appended.
        content_stamps: Append one worker-reported CAS stamp line per
            written target (``sha256 <relpath>: <64 hex> size: <n>``,
            Task 16 write-path parity with fs_read/fs_write/fs_edit).
            A patch may touch MANY targets, so unlike the single-path
            tools each stamp line names its target; the digest is the
            exact in-memory ``data`` each target's atomic write received,
            never a post-write re-read.
    """
    summaries: list[str] = []
    stamp_lines: list[str] = []
    for patch_file in plans:
        rel_path = patch_file.new_path
        if rel_path is None:
            raise LocalToolError('fs_patch failed [invalid_patch_path]')
        try:
            relative = root.relative_path(rel_path)
            written = _patch_relative_file(patch_file, relative, workspace=Path('.'), dry_run=dry_run)
        except WorkspaceRootPinError as exc:
            raise LocalToolError('fs_patch failed [invalid_patch_path]') from exc
        except FilesystemPatchError as exc:
            raise LocalToolError(f'fs_patch failed [{exc.reason_code}]: {rel_path}') from exc
        summaries.append(f"{('would patch' if dry_run else 'patched')} {rel_path}")
        if content_stamps and (not dry_run):
            stamp_lines.append(f'sha256 {relative.as_posix()}: {written[0]} size: {written[1]}')
    return '\n'.join([*summaries, *stamp_lines])

def _patch_relative_file(patch_file: PatchFile, relative: Path, *, workspace: Path, dry_run: bool) -> 'tuple[str, int] | None':
    """Apply one parsed patch plan using only root-relative I/O.

    Returns the ``(sha256_hex, size)`` of the exact bytes the patch
    produced (the same in-memory ``data`` the atomic write receives, so
    preview and real run stamp identically).
    """
    rel_path = patch_file.new_path
    assert rel_path is not None
    target = workspace / relative
    if patch_file.action == 'modify':
        if not target.is_file():
            raise LocalToolError(f'file not found: {rel_path}')
        try:
            with open(target, encoding='utf-8', newline='') as fh:
                original = fh.read()
        except UnicodeDecodeError as exc:
            raise LocalToolError(f"'{rel_path}' is not valid UTF-8; fs_patch only patches text files") from exc
    else:
        if target.exists():
            raise LocalToolError(f'file already exists: {rel_path}')
        if not target.parent.is_dir():
            raise LocalToolError(f'parent directory does not exist for: {rel_path}')
        original = ''
    updated = apply_patch_to_text(original, patch_file)
    try:
        data = updated.encode('utf-8')
    except UnicodeEncodeError as exc:
        raise LocalToolError(f"patched content for '{rel_path}' is not UTF-8 encodable (lone surrogate?): {exc}") from exc
    if not dry_run:
        _atomic_write_target(target, data, shown=str(rel_path), expected_sha256=None, expected_absent=patch_file.action != 'modify')
    return (hashlib.sha256(data).hexdigest(), len(data))


# ===========================================================================
# Section: tldw_chatbook.Tools.workspace_tool_dispatch (extracted from the live module by the
# builder; regenerate rather than editing)
# ===========================================================================
"""Dispatch closed workspace operations against one retained root pin.

Part of the pinned worker's stdlib-only import closure (Phase 0c): nothing
here may import the parent's pydantic protocol module. Request frames
arrive already decoded/validated; dispatch consumes the attribute surface
``_PinnedOperationRequest`` describes (the parent's
``WorkspaceToolRequest`` dataclass and the worker's decoded-request view
both satisfy it).
"""
import shutil
from pathlib import Path
from typing import Any, Protocol

class WorkspaceToolDispatchError(RuntimeError):
    """A fixed-code refusal from the pinned worker dispatcher."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code

class _PinnedOperationRequest(Protocol):
    """The attribute surface dispatch consumes from one admitted request."""
    operation: str
    arguments: dict[str, Any]

def _remote_home_denylist_exclusions() -> tuple[SensitiveExclusion, ...]:
    """Map the worker-side remote-home denylist into root-relative space.

    ``REMOTE_SENSITIVE_PATHS`` (``Tools/remote_sensitive_paths.py``) is
    the set of REMOTE-HOME-relative subtrees the pinned worker must never
    expose regardless of which root a binding pins. This module is shared
    by the LOCAL pinned worker and the flattened remote bundle; the
    denylist NAME exists only in the bundle's flat namespace (Task 8
    embeds the data below the dispatch section), so the live local worker
    resolves it to ``()`` through ``globals().get`` — byte-identical
    local behavior — while the bundle picks the tuple up at call time.

    Mapping rule (Task 17, ADR-174 floor): each home-relative entry is
    joined onto the resolved home directory, then re-expressed relative
    to the pinned root (``Path('.')`` after the root pin's chdir). An
    entry outside the pinned root maps to nothing — the worker is root-
    confined, so it is unreachable — and a root ABOVE (or at) the remote
    home maps every entry into a 'subtree' refusal the existing matcher
    enforces for reads, writes, and enumeration. An unresolvable home
    degrades to ``()``: the request-carried exclusions still enforce, and
    a missing ``HOME`` must not crash unrelated operations.
    """
    entries = globals().get('REMOTE_SENSITIVE_PATHS', ())
    if not entries:
        return ()
    try:
        home = Path.home().resolve()
        root = Path('.').resolve()
        relative_root = root.relative_to(home)
    except (RuntimeError, OSError, ValueError):
        return ()
    return tuple((SensitiveExclusion('subtree', (relative_root / entry).as_posix()) for entry in entries))

def execute_pinned_operation(request: _PinnedOperationRequest, root: PinnedWorkspaceRoot) -> str:
    """Execute one supported request relative to ``root`` or refuse it."""
    if request.operation == 'stat_path':
        relative = _request_relative_path(request, root)
        denylist = _remote_home_denylist_exclusions()
        if denylist and (not _relative_target_is_safe(relative, Path('.'), denylist, is_directory=True)):
            raise WorkspaceToolDispatchError('invalid_request', 'workspace path is invalid')
        return _stat_relative_path(relative)
    if request.operation == 'fs_write':
        return _write_relative_file(_request_mutation_path(request, root), request.arguments['content'], workspace=Path('.'), display_path=request.arguments['path'], dry_run=request.arguments.get('dry_run', False), expected_sha256=request.arguments.get('expected_sha256'), expected_absent=request.arguments.get('expected_absent', False), content_stamps=True)
    if request.operation == 'fs_edit':
        return _edit_relative_file(_request_mutation_path(request, root), request.arguments['old_string'], request.arguments['new_string'], workspace=Path('.'), replace_all=request.arguments.get('replace_all', False), display_path=request.arguments['path'], content_stamps=True)
    if request.operation == 'fs_patch':
        return _patch_request(request, root)
    if request.operation in {'git_status', 'git_diff', 'git_log', 'git_blame', 'git_branches'}:
        return _git_request(request)
    if request.operation not in {'fs_list', 'fs_read', 'fs_glob', 'fs_grep'}:
        raise WorkspaceToolDispatchError('unsupported_operation', 'workspace operation is not implemented')
    exclusions = _request_exclusions(request, 'sensitive_exclusions')
    if request.operation == 'fs_list':
        return _list_relative_directory(_request_relative_path(request, root), workspace=Path('.'), max_entries=MAX_LIST_ENTRIES, sensitive_exclusions=exclusions)
    if request.operation == 'fs_read':
        return _read_relative_file(_request_relative_path(request, root), workspace=Path('.'), offset=request.arguments.get('offset', 1), limit=request.arguments.get('limit'), sensitive_exclusions=exclusions, content_stamps=True)
    if request.operation == 'fs_glob':
        try:
            pattern = validate_glob_pattern(request.arguments['pattern'])
        except WireDecodeError:
            raise WorkspaceToolDispatchError('invalid_request', 'workspace glob pattern is invalid') from None
        return _glob_relative_files(pattern, workspace=Path('.'), max_results=request.arguments.get('max_results', MAX_GLOB_RESULTS), sensitive_exclusions=exclusions, validate_targets=True)
    if request.operation == 'fs_grep':
        return _grep_relative_files(request.arguments['pattern'], workspace=Path('.'), mode=request.arguments.get('mode', 'content'), max_results=request.arguments.get('max_results', MAX_GREP_RESULTS), sensitive_exclusions=_request_exclusions(request, 'content_exclusions'))

def _git_request(request: _PinnedOperationRequest) -> str:
    """Run one closed read-only Git operation beneath the retained root."""
    discovered = shutil.which('git')
    if discovered is None:
        raise WorkspaceToolDispatchError('tool_failure', 'git is not available on this system')
    executable = Path(discovered).resolve()
    exclusions = _request_exclusions(request, 'sensitive_exclusions')
    execution = {'executable': executable, 'own_process_group': False, 'sensitive_exclusions': exclusions}
    arguments = request.arguments
    if request.operation == 'git_status':
        return git_status(Path('.'), arguments.get('path', '.'), **execution)
    if request.operation == 'git_diff':
        return git_diff(Path('.'), staged=arguments.get('staged', False), commit_range=arguments.get('commit_range'), path=arguments.get('path'), stat=arguments.get('stat', False), **execution)
    if request.operation == 'git_log':
        return git_log(Path('.'), count=arguments.get('count', 20), path=arguments.get('path'), **execution)
    if request.operation == 'git_blame':
        return git_blame(Path('.'), arguments['path'], start_line=arguments.get('start_line'), end_line=arguments.get('end_line'), **execution)
    return git_branches(Path('.'), **execution)

def _request_relative_path(request: _PinnedOperationRequest, root: PinnedWorkspaceRoot) -> Path:
    """Return one request path validated as lexical root-relative text."""
    try:
        return root.relative_path(request.arguments['path'])
    except WorkspaceRootPinError:
        raise WorkspaceToolDispatchError('invalid_request', 'workspace operation path is invalid') from None

def _request_exclusions(request: _PinnedOperationRequest, field: str) -> tuple[SensitiveExclusion, ...]:
    """Decode the parent's fixed bounded exclusions without filesystem discovery.

    Task 17: the worker-side remote-home denylist joins the request's
    serialized exclusions here — the single decode point every read,
    write, patch, and git consumer passes through — so ``REMOTE_SENSITIVE_PATHS``
    is enforced on every operation even when the parent serialized none
    (the local pinned worker's denylist contribution is ``()``, keeping
    its behavior byte-identical).
    """
    return tuple((SensitiveExclusion(item['kind'], item['value']) for item in request.arguments[field])) + _remote_home_denylist_exclusions()

def _request_mutation_path(request: _PinnedOperationRequest, root: PinnedWorkspaceRoot) -> Path:
    """Validate a mutation target's live lexical and resolved location."""
    relative = _request_relative_path(request, root)
    if not _relative_target_is_safe(relative, Path('.'), _request_exclusions(request, 'sensitive_exclusions'), is_directory=False):
        raise WorkspaceToolDispatchError('invalid_request', 'workspace mutation target is invalid')
    return relative

def _patch_request(request: _PinnedOperationRequest, root: PinnedWorkspaceRoot) -> str:
    """Reparse a bounded patch and require its exact parent-admitted targets."""
    try:
        plans = parse_patch_targets(request.arguments['diff'])
        parsed_paths = tuple((root.relative_path(plan.new_path) for plan in plans if plan.new_path is not None))
        parsed_targets = tuple((path.as_posix() for path in parsed_paths))
        requested_targets = tuple((root.relative_path(target).as_posix() for target in request.arguments.get('targets', ())))
    except (FilesystemPatchError, WorkspaceRootPinError, TypeError):
        raise WorkspaceToolDispatchError('invalid_request', 'workspace patch request is invalid') from None
    if len(parsed_targets) != len(plans) or parsed_targets != requested_targets:
        raise WorkspaceToolDispatchError('invalid_request', 'workspace patch targets changed after admission')
    exclusions = _request_exclusions(request, 'sensitive_exclusions')
    if not all((_relative_target_is_safe(relative, Path('.'), exclusions, is_directory=False) for relative in parsed_paths)):
        raise WorkspaceToolDispatchError('invalid_request', 'workspace patch target is invalid')
    return patch_validated_files(plans, root=root, dry_run=request.arguments.get('dry_run', False), content_stamps=True)


# ===========================================================================
# Section: tldw_chatbook.Tools.workspace_tool_worker (extracted from the live module by the
# builder; regenerate rather than editing)
# ===========================================================================
"""Fixed one-shot stdin/stdout worker for pinned workspace operations.

Import closure (Phase 0c): this module's transitive imports must stay
stdlib-only — Task 8 concatenates the closure into a remote worker bundle,
and ``Tests/Tools/test_worker_import_closure.py`` is the gate. Frames are
decoded by ``Tools/workspace_wire_decode.py`` (the stdlib counterpart of
the parent's pydantic serde in ``Tools/workspace_tool_protocol.py``) and
responses are emitted by ``workspace_wire_decode.encode_response``, whose
byte layout the conformance tests pin to the parent's
``WorkspaceToolResponse.to_bytes``.
"""
import json
import os
import platform
import stat
import sys
import time
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO
_MAX_DOMAIN_ERROR_CHARS = 300

@dataclass(frozen=True, slots=True)
class _DecodedRequest:
    """Attribute view over one decoded frame, for pinning and dispatch.

    Field-for-field the decoded payload of ``decode_request``; the wire
    decoder has already applied every admission check the parent's
    ``WorkspaceToolRequest.from_bytes`` applies.
    """
    operation_id: str
    operation: str
    root_locator: Path = field(repr=False)
    root_identity: DirectoryIdentity
    ancestor_identities: tuple[DirectoryIdentity, ...]
    arguments: dict[str, Any] = field(repr=False)
    timeout_seconds: int

def run_workspace_worker(stdin: BinaryIO, stdout: BinaryIO, stderr: BinaryIO, *, bundle_sha256: str='') -> int:
    """Read, pin, dispatch, respond once, and return a process exit code.

    Args:
        stdin: The buffered request stream (already positioned past the
            remote bundle bytes when running as the shipped bundle).
        stdout: Frame sink for the admitted/terminal response contract.
        stderr: Reserved for fixed diagnostics; no request-derived text
            is written.
        bundle_sha256: The remote bundle's identity stamp, threaded in
            by the bundle IO adapter so ``ping`` can echo it. The LOCAL
            worker has no bundle and reports the empty default.
    """
    del stderr
    started = time.monotonic()
    raw = stdin.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        _emit(stdout, _failure('unknown', 'invalid_request', started))
        return 2
    try:
        request = _decode_request(raw)
    except WireDecodeError:
        _emit(stdout, _failure('unknown', 'invalid_request', started))
        return 2
    arm_watchdog(request.timeout_seconds, TEMP_REGISTRY)
    try:
        return _run_admitted_exchange(stdout, request, started, bundle_sha256=bundle_sha256)
    finally:
        disarm_watchdog()

def _run_admitted_exchange(stdout: BinaryIO, request: _DecodedRequest, started: float, *, bundle_sha256: str) -> int:
    """Pin, dispatch, and respond for one watchdog-armed request."""
    if request.operation == 'ping':
        return _run_ping(stdout, request, started, bundle_sha256=bundle_sha256)
    chain = DirectoryChain(canonical_root=request.root_locator, identities=(request.root_identity, *request.ancestor_identities[1:]))
    try:
        with pin_workspace_root(request.root_locator, chain) as root:
            _emit(stdout, _frame(request.operation_id, outcome='admitted', code='root_pinned', result=None, error=None, started=started))
            result = execute_pinned_operation(request, root)
        _emit(stdout, _frame(request.operation_id, outcome='success', code='ok', result=result, error=None, started=started))
        return 0
    except WorkspaceToolDispatchError as error:
        _emit(stdout, _failure(request.operation_id, error.code, started, message=str(error)))
        return 2
    except WorkspaceRootPinError:
        _emit(stdout, _failure(request.operation_id, 'root_pin_failed', started))
        return 2
    except LocalToolError as error:
        _emit(stdout, _failure(request.operation_id, 'tool_failure', started, message=_sanitized_domain_error(error, request.root_locator)))
        return 2
    except (OSError, ValueError):
        _emit(stdout, _failure(request.operation_id, 'tool_failure', started))
        return 2
    except BaseException:
        _emit(stdout, _failure(request.operation_id, 'worker_failure', started))
        return 2

def _decode_request(raw: bytes) -> _DecodedRequest:
    """Decode one admitted frame into the worker's pinned-request view."""
    payload = decode_request(raw)
    return _DecodedRequest(operation_id=payload['operation_id'], operation=payload['operation'], root_locator=Path(payload['root_locator']), root_identity=_identity(payload['root_identity']), ancestor_identities=tuple((_identity(item) for item in payload['ancestor_identities'])), arguments=payload['arguments'], timeout_seconds=payload['timeout_seconds'])

def _identity(payload: Mapping[str, Any]) -> DirectoryIdentity:
    return DirectoryIdentity(device=payload['device'], inode=payload['inode'], mode=payload['mode'], reparse=payload['reparse'])

def _run_ping(stdout: BinaryIO, request: _DecodedRequest, started: float, *, bundle_sha256: str) -> int:
    """Capture and report the root's full identity chain without pinning.

    Ping is the bootstrap probe: the ONE operation dispatched before the
    root pin, because its purpose is to capture the identity chain every
    other operation's request must carry (over ssh the parent cannot
    stat the remote root). The request's own identity fields are
    therefore advisory for ping — a first-contact ping has no identity
    to verify against. A root that cannot be captured (missing, itself a
    symlink, unsafe metadata) fails with the pin-failure code and emits
    NO admitted marker — the no-marker bucket the parent's status cache
    reads as a transport/setup-class failure.
    """
    try:
        locator_metadata = os.lstat(request.root_locator)
        if not stat.S_ISDIR(locator_metadata.st_mode) or stat.S_ISLNK(locator_metadata.st_mode):
            raise DirectoryIdentityError('unsafe directory metadata')
        chain = capture_directory_chain(request.root_locator)
        payload = _ping_payload(chain, bundle_sha256=bundle_sha256)
    except (DirectoryIdentityError, OSError, ValueError):
        _emit(stdout, _failure(request.operation_id, 'root_pin_failed', started))
        return 2
    _emit(stdout, _frame(request.operation_id, outcome='success', code='ok', result=payload, error=None, started=started))
    return 0

def _ping_payload(chain: DirectoryChain, *, bundle_sha256: str) -> str:
    """Serialize the ping result: chain, canonical path, python, stamp.

    The identity chain is root-first and covers every ancestor to ``/``,
    exactly the shape ``DirectoryChain(identities=(root_identity,
    *ancestor_identities[1:]))`` reconstruction consumes — a root-only
    stat cannot build a request the pinned dispatcher accepts. Rendered
    as a JSON document inside the response's string ``result`` field
    (the frame schema itself is unchanged).
    """
    paths = (chain.canonical_root, *chain.canonical_root.parents)
    payload = {'identity_chain': [[str(path_text), identity.device, identity.inode, identity.mode] for path_text, identity in zip(paths, chain.identities)], 'canonical_path': str(chain.canonical_root), 'python_version': platform.python_version(), 'bundle_sha256': bundle_sha256}
    return json.dumps(payload, allow_nan=False, ensure_ascii=False, separators=(',', ':'))

def _frame(operation_id: str, *, outcome: str, code: str, result: str | None, error: str | None, started: float) -> dict[str, Any]:
    """Build one response payload in the parent's fixed field order."""
    return {'version': WIRE_VERSION, 'operation_id': operation_id, 'outcome': outcome, 'code': code, 'result': result, 'error': error, 'elapsed_ms': _elapsed_ms(started), 'truncated': False, 'cleanup_proven': True}

def _failure(operation_id: str, code: str, started: float, *, message: str='workspace operation failed') -> dict[str, Any]:
    return _frame(operation_id, outcome='failure', code=code, result=None, error=message, started=started)

def _sanitized_domain_error(error: LocalToolError, root_locator: object) -> str:
    """Return bounded model-actionable text from one audited domain type."""
    message = str(error)
    root_text = str(root_locator)
    for separator in ('/', '\\'):
        message = message.replace(root_text + separator, '')
    message = message.replace(root_text, '.')
    message = ''.join((character for character in message if unicodedata.category(character) != 'Cc'))
    return message[:_MAX_DOMAIN_ERROR_CHARS] or 'workspace operation failed'

def _elapsed_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))

def _emit(stdout: BinaryIO, payload: dict[str, Any]) -> None:
    frame = encode_response(payload)
    decode_response(frame)
    stdout.write(frame + b'\n')
    stdout.flush()


# ---------------------------------------------------------------------------
# Bundle IO adapter (builder-emitted; the LOCAL worker has no counterpart)
# ---------------------------------------------------------------------------

#: The response magic: exactly 16 bytes (controller ruling; the earlier
#: 15-byte literal was a miscount). Prefixed to every response frame this
#: bundle emits so stdout noise on a shared remote channel can never be
#: mistaken for a response frame. The LOCAL pinned worker does NOT add
#: this prefix: its pipe has no noise source. This is the single
#: definition site; transport tests (Task 11) mirror the literal.
RESPONSE_MAGIC = b"TLDW-REMOTE-0001"


def split_magic(raw: bytes) -> tuple[bool, bytes]:
    """Strip ``RESPONSE_MAGIC`` from one raw response line before parsing.

    Args:
        raw: One captured response line, magic-prefixed or not.

    Returns:
        ``(had_magic, stripped)`` — ``had_magic`` is ``False`` (and
        ``stripped`` is ``raw`` unchanged) when the prefix is absent.
    """
    if raw.startswith(RESPONSE_MAGIC):
        return True, raw[len(RESPONSE_MAGIC) :]
    return False, raw


class _MagicPrefixStdout:
    """Minimal stdout shim prefixing every write with ``RESPONSE_MAGIC``."""

    __slots__ = ("_stream",)

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def write(self, data: bytes) -> int:
        return self._stream.write(RESPONSE_MAGIC + data)

    def flush(self) -> None:
        return self._stream.flush()


def main(stream: Any, *, bundle_sha256: str = "") -> int:
    """Run one isolated protocol exchange over an already-positioned stdin.

    ``stream`` is the buffered stdin AFTER the remote bootstrap consumed
    this bundle's own bytes (Task 9's harness positions it). Responses
    are written to the process stdout buffer with ``RESPONSE_MAGIC``
    prefixed to each frame; the process exit code follows the local
    worker's contract (0 success, 2 refused/failed). ``bundle_sha256``
    defaults to the empty string for direct/in-process callers; the
    bootstrap entry path always supplies the artifact stamp.
    """
    return run_workspace_worker(
        stream,
        _MagicPrefixStdout(sys.stdout.buffer),
        sys.stderr.buffer,
        bundle_sha256=bundle_sha256,
    )


def _enter_worker_exchange(stamp: str) -> str:
    """Bootstrap entry seam — run the exchange when exec'd as ``__main__``.

    The builder emits the artifact's FINAL line as
    ``BUNDLE_SHA256 = _enter_worker_exchange("<digest>")``. Under the
    fixed remote bootstrap this module executes inside the interpreter's
    own ``__main__`` namespace, so evaluating that assignment fires the
    one exchange (and the ``SystemExit`` aborts module execution before
    anything uncovered could follow the line). In-process loaders run
    under their own module name and simply get the stamp bound.

    Defined ABOVE the stamp assignment on purpose: the entry logic must
    sit inside the region ``BUNDLE_SHA256`` attests, so a rewritten tail
    cannot launch divergent code while echoing a matching stamp.
    """
    if __name__ == "__main__":
        raise SystemExit(main(sys.stdin.buffer, bundle_sha256=stamp))
    return stamp



# ---------------------------------------------------------------------------
# Remote denylist (embedded from Tools/remote_sensitive_paths.py)
# ---------------------------------------------------------------------------
#: Remote-home-relative paths the worker must never read, list, or
#: write, regardless of the pinned root. Enforced (Task 17) by the
#: dispatch section above: _remote_home_denylist_exclusions maps
#: these onto the pinned root and joins them into every operation's
#: exclusion set. The local pinned worker resolves no such name and
#: keeps its byte-identical behavior.
REMOTE_SENSITIVE_PATHS: tuple[str, ...] = (
    '.ssh',
    '.aws',
    '.gnupg',
    '.config/gcloud',
    '.kube',
    '.docker',
    '.netrc',
)


# ---------------------------------------------------------------------------
# Bundle identity stamp (builder-emitted; ping echoes this value)
# ---------------------------------------------------------------------------
#: SHA-256 of this file's bytes ABOVE this assignment line — which is the
#: file's FINAL line, so every executable byte (bootstrap entry included)
#: is covered; only this assignment's own line falls outside the digest.
#: A full-file digest is not self-embeddable (the stamp would change its
#: own input); derive the same value from the artifact with
#: ``build_remote_worker_bundle.expected_bundle_stamp``. The remote
#: worker's ``ping`` echoes it so callers can confirm which bundle the
#: remote actually executed.
BUNDLE_SHA256 = _enter_worker_exchange("609ef17a10ce56f1f1dd8098097679f4cdc9e32467b7e55546de5e87c36e3445")
