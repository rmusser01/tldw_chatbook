"""Exact Chat_Dictionary_Lib import/export/parser IO; external paths are not inventory."""

from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass
from functools import wraps
import os
import ctypes
import errno
from pathlib import Path
import re
import shutil
import stat
import sys
import threading
import weakref

from . import bootstrap, profile_paths, storage_admission as storage

ROUTE = "dictionary_files"
_local = threading.local()
_bindings = weakref.WeakKeyDictionary()
_errors = weakref.WeakKeyDictionary()


@dataclass
class _Selection:
    module: object
    config: object
    config_path: Path
    root: Path
    selected: Path
    inputs: tuple
    output: Path | None
    db: object
    record: object = None
    token: object = None
    core: object = None
    input_stats: dict | None = None
    create_parent: bool = False
    published: bool = False
    destination_identity: tuple | None = None
    directory: bool = False
    metadata_supported: bool = True


def _flags_function():
    if hasattr(os, "fchflags"):
        return os.fchflags
    if sys.platform != "darwin":
        return None
    try:
        function = ctypes.CDLL(None, use_errno=True).fchflags
    except (OSError, AttributeError):
        return None
    function.argtypes = (ctypes.c_int, ctypes.c_uint32)
    function.restype = ctypes.c_int

    def apply(descriptor, flags):
        if function(descriptor, flags) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))

    return apply


def _copy_flags(descriptor, flags):
    function = _flags_function()
    if function is None:
        raise bootstrap.RecoveryRequired("dictionary_metadata_unqualified")
    try:
        function(descriptor, flags)
    except OSError as error:
        # Match shutil.copystat's deliberately narrow unsupported-flags policy.
        if error.errno not in {errno.EOPNOTSUPP, errno.ENOTSUP}:
            raise


def _module():
    return sys.modules["tldw_chatbook.Character_Chat.Chat_Dictionary_Lib"]


def _check_selection(plan):
    config = plan.config
    if (
        plan.module is not _module()
        or config is not sys.modules.get("tldw_chatbook.config")
        or config._CONFIG_CACHE_SOURCE != plan.config_path
        or profile_paths.lexical_path(config._get_effective_config_path())
        != plan.config_path
        or profile_paths.user_data_dir(config._CONFIG_CACHE) / "chat_dicts" != plan.root
    ):
        raise bootstrap.RecoveryRequired("dictionary_file_selection_changed")
    if plan.db is not None and getattr(plan, "installed_db", False):
        if (
            profile_paths.database_path(config._CONFIG_CACHE, "chachanotes_db_path")
            != plan.db.db_path
        ):
            raise bootstrap.RecoveryRequired("dictionary_file_selection_changed")


def binding(source):
    bound = _bindings.get(source)
    if bound is None:
        return None
    _check_selection(bound)
    return "chat.dictionaries", bound.root, True


def selection(source):
    plan = getattr(_local, "plan", None)
    if plan is None or plan.module is not source or source is not _module():
        raise bootstrap.RecoveryRequired("dictionary_file_source_invalid")
    _check_selection(plan)
    binding(source)
    return plan.selected, getattr(plan, "installed", False), plan.directory


def members(source):
    plan = _local.plan
    selection(source)
    paths = (plan.selected,)
    if plan.output is not None:
        paths += (plan.output.with_suffix(plan.output.suffix + ".tmp"),)
    return paths + tuple(path for path in plan.inputs if path not in paths)


def pin_inputs(state):
    from ..Utils.private_paths import _open_verified_parent
    from . import raw_participants as raw

    plan = _local.plan
    plan.input_stats = {}
    if plan.output is not None:
        try:
            info = plan.output.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("dictionary_output_not_regular")
            plan.destination_identity = (info.st_dev, info.st_ino)
        except FileNotFoundError:
            plan.destination_identity = None
    for path in plan.inputs:
        if path.parent not in state.pins and state.pinned:
            descriptor, _ = _open_verified_parent(
                path,
                missing_leaf_allowed=False,
                _close=lambda fd: raw._close_descriptor(state, fd),
            )
            state.pins[path.parent] = descriptor
        if not state.pinned:
            info = path.parent.stat()
            state.identities[path.parent] = (info.st_dev, info.st_ino)
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("dictionary_input_not_regular")
        state.observed_files[path] = (info.st_dev, info.st_ino)
        plan.input_stats[path] = _stamp(info)


def _stamp(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _current():
    from . import raw_participants as raw

    plan = getattr(_local, "plan", None)
    if plan is None or plan.token is None:
        return None
    _check_selection(plan)
    raw._check(plan.token)
    if plan.core is not None:
        storage._check_operation(plan.core, plan.db.db_path)
    return plan


@contextmanager
def _operation(
    db,
    inputs,
    output,
    *,
    record=None,
    root=None,
    config=None,
    create_parent=False,
    directory=False,
):
    from .. import config as current_config
    from ..DB.ChaChaNotes_DB import CharactersRAGDB
    from . import raw_participants as raw
    from .participants import _core_operation

    previous_plan = getattr(_local, "plan", None)
    previous_raw = getattr(raw._local, "operation", None)
    previous_core = getattr(storage._operation_local, "operation", None)
    raw._local.operation = None
    storage._operation_local.operation = None
    _local.plan = None
    try:
        with closing(storage._Acquisition()) as attempt:
            config = config or current_config
            data = config.load_cli_config_and_ensure_existence()
            root = root or profile_paths.user_data_dir(data) / "chat_dicts"
            inputs = tuple(profile_paths.lexical_path(path) for path in inputs)
            output = profile_paths.lexical_path(output) if output is not None else None
            plan = _Selection(
                _module(),
                config,
                profile_paths.lexical_path(config._get_effective_config_path()),
                root,
                root if directory else (output or inputs[0]),
                inputs,
                output,
                db,
                record,
            )
            plan.create_parent = create_parent
            plan.directory = directory
            plan.installed_db = (
                type(db) is CharactersRAGDB
                and not db.is_memory_db
                and profile_paths.database_path(data, "chachanotes_db_path")
                == db.db_path
            )
            plan.metadata_supported = not (
                inputs
                and output is not None
                and hasattr(os.stat(inputs[-1]), "st_flags")
                and _flags_function() is None
            )
            plan.installed = (
                db is None or plan.installed_db
            ) and plan.metadata_supported
            binding(plan.module)  # A previously installed module cannot demote.
            if plan.installed:
                if plan.module not in _bindings:
                    _bindings[plan.module] = plan
                _check_selection(plan)
            _local.plan = plan
            with ExitStack() as scopes:
                if isinstance(db, CharactersRAGDB):
                    scopes.enter_context(_core_operation(db))
                    plan.core = getattr(storage._operation_local, "operation", None)
                token = scopes.enter_context(
                    raw._scope(plan.module, ROUTE, writing=output is not None)
                )
                plan.token = token
                attempt.check()
                _check_selection(plan)
                raw._check(token)
                qualified = (
                    plan.installed
                    and all(raw._states[token].holds)
                    and (plan.core is None or plan.core.hold is not None)
                )
                if not qualified:
                    _errors[plan.module] = "dictionary_file_scope_unqualified"
                with storage._changed:
                    attempt.check()
                    participant = raw._states[token].participant
                    if (
                        participant is not None
                        and raw._participant_state(participant).closed
                    ):
                        raise bootstrap.RecoveryRequired("storage_locally_paused")
                    if plan.core is not None and plan.core.participant.closed:
                        raise bootstrap.RecoveryRequired("storage_locally_paused")
                    if qualified:
                        storage._operation_local.operation = plan.core
                try:
                    yield plan
                finally:
                    storage._operation_local.operation = None
            if token in raw._states:
                raise bootstrap.RecoveryRequired("raw_resources_not_retired")
    finally:
        _local.plan = previous_plan
        raw._local.operation = previous_raw
        storage._operation_local.operation = previous_core
        if previous_raw is not None:
            raw._check(previous_raw)
        if previous_core is not None:
            storage._check_operation(previous_core, previous_core.path)


def folder():
    from .. import config
    from . import config_participants, raw_participants as raw

    plan = _current()
    if plan is not None:
        return plan.root
    selected = config.get_user_data_dir() / "chat_dicts"
    with config_participants.operation(
        config, route="config_chat_dicts", target=selected
    ) as token:
        raw._mkdirs(token)
    return selected


def _function(function, guarded, name):
    if (
        function.__module__ != _module().__name__
        or function.__name__ != name
        or getattr(_module(), name) is not guarded
    ):
        raise bootstrap.RecoveryRequired("dictionary_file_source_invalid")


def listing(function):
    @wraps(function)
    def guarded():
        _function(function, guarded, "list_available_dictionary_files")
        with closing(storage._Acquisition()) as attempt:
            root = folder()
            attempt.check()
            with _operation(None, (), None, root=root, directory=True):
                return function()

    return guarded


def parser(function):
    @wraps(function)
    def guarded(file_path, base_directory=None):
        _function(function, guarded, "parse_user_dict_markdown_file")
        plan = _current()
        if plan is not None:
            selected = profile_paths.lexical_path(
                _module().validate_path(file_path, base_directory or plan.root)
            )
            if selected not in plan.inputs:
                raise bootstrap.RecoveryRequired("dictionary_input_outside_operation")
            return function(file_path, base_directory)
        with closing(storage._Acquisition()) as attempt:
            base = base_directory or _module()._default_dictionary_import_directory()
            try:
                selected = _module().validate_path(file_path, base)
            except ValueError:
                return function(file_path, base_directory)
            attempt.check()
            try:
                with _operation(None, (selected,), None):
                    return function(file_path, base_directory)
            except OSError:
                return {}

    return guarded


def importing(function):
    @wraps(function)
    def guarded(db, file_path, name=None, description=None):
        _function(function, guarded, "import_dictionary_from_file")
        try:
            return invoke(db, file_path, name, description)
        except OSError:
            return None

    def invoke(db, file_path, name, description):
        from .. import config

        with closing(storage._Acquisition()) as attempt:
            data = config.load_cli_config_and_ensure_existence()
            root = profile_paths.user_data_dir(data) / "chat_dicts"
            try:
                validated = _module().validate_path(file_path, root)
            except ValueError:
                return None
            original = profile_paths.lexical_path(file_path)
            destination = root / Path(file_path).name
            output = (
                destination if original.resolve() != destination.resolve() else None
            )
            attempt.check()
            with _operation(
                db,
                tuple(dict.fromkeys((Path(validated), original))),
                output,
                root=root,
                config=config,
            ) as plan:
                try:
                    result = function(db, file_path, name, description)
                except BaseException:
                    if plan.published:
                        _errors[plan.module] = (
                            "dictionary_import_publication_incomplete"
                        )
                    raise
                if result is None and plan.published:
                    _errors[plan.module] = "dictionary_import_publication_incomplete"
                return result

    return guarded


def exporting(function):
    @wraps(function)
    def guarded(db, dict_id, export_path=None):
        _function(function, guarded, "export_dictionary_to_file")
        try:
            return invoke(db, dict_id, export_path)
        except OSError:
            return None

    def invoke(db, dict_id, export_path):
        from .. import config
        from .participants import _core_operation
        from ..DB.ChaChaNotes_DB import CharactersRAGDB
        from contextlib import nullcontext

        with closing(storage._Acquisition()) as attempt:
            with (
                _core_operation(db)
                if isinstance(db, CharactersRAGDB)
                else nullcontext()
            ):
                record = _module().load_chat_dictionary(db, dict_id)
                if not record:
                    return None
                data = config.load_cli_config_and_ensure_existence()
                root = profile_paths.user_data_dir(data) / "chat_dicts"
                safe_name = re.sub(r"[^\w\s-]", "", record["name"])
                safe_name = re.sub(r"[-\s]+", "-", safe_name)
                selected = export_path or str(root / f"{safe_name}.md")
                attempt.check()
                with _operation(
                    db,
                    (),
                    selected,
                    record=record,
                    root=root,
                    config=config,
                    create_parent=not export_path,
                ):
                    return function(db, dict_id, export_path)

    return guarded


def export_record(db, dict_id):
    plan = _current()
    if plan is not None and plan.db is db and plan.record is not None:
        if int(plan.record["id"]) != int(dict_id):
            raise bootstrap.RecoveryRequired("dictionary_record_changed")
        return plan.record
    return _module().load_chat_dictionary(db, dict_id)


@contextmanager
def opened(path, mode, *, encoding="utf-8"):
    from . import raw_participants as raw

    plan = _current()
    if plan is None or encoding != "utf-8":
        raise bootstrap.RecoveryRequired("dictionary_file_source_invalid")
    path = profile_paths.lexical_path(path)
    if mode == "r":
        if path not in plan.inputs:
            raise bootstrap.RecoveryRequired("dictionary_input_outside_operation")
        with raw._file(plan.token, path, "r") as stream:
            before = _stamp(os.fstat(stream.fileno()))
            if before != plan.input_stats[path]:
                raise bootstrap.RecoveryRequired("dictionary_input_changed")
            yield stream
            if (
                _stamp(os.fstat(stream.fileno())) != before
                or _stamp(path.stat()) != before
            ):
                raise bootstrap.RecoveryRequired("dictionary_input_changed")
        return
    if mode != "w" or path != plan.output:
        raise bootstrap.RecoveryRequired("dictionary_output_outside_operation")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if plan.create_parent:
        raw._mkdirs(plan.token)
    try:
        with raw._file(plan.token, temporary, "w") as stream:
            yield stream
        check_destination(plan)
        raw._replace(plan.token, temporary, path)
        plan.published = True
    finally:
        state = raw._states[plan.token]
        if temporary in state.created_files and not state.uncertain:
            raw._remove_temporary(plan.token, temporary)


def copy_file(source, destination):
    """Preserve copy2 byte/metadata behavior on the two admitted native files."""
    with opened(source, "r") as incoming:
        with opened(destination, "w") as outgoing:
            shutil.copyfileobj(incoming.buffer, outgoing.buffer)
            info = os.fstat(incoming.fileno())
            outgoing.flush()
            if _current().token is not None:
                from . import raw_participants as raw

                state = raw._check(_current().token)
                if state.pinned and _current().metadata_supported:
                    os.utime(outgoing.fileno(), ns=(info.st_atime_ns, info.st_mtime_ns))
                    shutil._copyxattr(incoming.fileno(), outgoing.fileno())
                    os.fchmod(outgoing.fileno(), stat.S_IMODE(info.st_mode))
                    if hasattr(info, "st_flags"):
                        _copy_flags(outgoing.fileno(), info.st_flags)
                else:
                    temporary = Path(destination).with_suffix(
                        Path(destination).suffix + ".tmp"
                    )
                    shutil.copystat(source, temporary)


def check_destination(plan):
    try:
        info = plan.output.lstat()
        identity = (info.st_dev, info.st_ino)
    except FileNotFoundError:
        identity = None
    if identity != plan.destination_identity:
        raise bootstrap.RecoveryRequired("dictionary_output_identity_changed")


def drain_ready(source):
    binding(source)
    return source not in _errors
