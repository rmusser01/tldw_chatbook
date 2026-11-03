"""Private evidence for the original first-send config publication.

The retained views are process-local and can contain secrets.  They are never
logged, serialized, or used to authorize a new operation.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field
from pathlib import PosixPath
from types import FunctionType

from tldw_chatbook import config


def _native_config_current():
    if sys.modules.get("tldw_chatbook.config") is not config:
        return False
    namespace, rows = config._FIRST_SEND_RETIREMENT_SOURCE
    result_type, status_type, created = config._FIRST_SEND_RETIREMENT_TYPES
    if (
        namespace is not vars(config)
        or config.PrivatePathResult is not result_type
        or config.PrivatePathStatus is not status_type
        or status_type.CREATED_PRIVATE is not created
    ):
        return False
    for owner, name, original, records in rows:
        if owner is not namespace or owner.get(name) is not original or not records:
            return False
        for (
            fn,
            code,
            globals_,
            defaults,
            kwdefaults,
            entries,
            closure,
            cells,
            wrapped,
        ) in records:
            if (
                type(fn) is not FunctionType
                or fn.__code__ is not code
                or fn.__globals__ is not globals_
                or fn.__defaults__ is not defaults
                or fn.__kwdefaults__ is not kwdefaults
                or fn.__closure__ is not closure
                or vars(fn).get("__wrapped__") is not wrapped
                or (
                    kwdefaults is not None
                    and (
                        len(kwdefaults) != len(entries)
                        or any(
                            kwdefaults.get(key) is not value for key, value in entries
                        )
                    )
                )
                or any(cell.cell_contents is not value for cell, value in cells)
            ):
                return False
    return True


def _plain(value):
    if value is None or type(value) in (str, int, float, bool, bytes, PosixPath):
        return True
    if type(value) in (list, tuple):
        return all(_plain(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _plain(item) for key, item in value.items())
    return False


def _same(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return len(left) == len(right) and all(
            key in right and _same(value, right[key]) for key, value in left.items()
        )
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _set_flag(value, prefix=()):
    current = value
    for key in (*prefix, "console", "onboarding"):
        child = current.get(key)
        if child is None and key not in current:
            child = current[key] = {}
        if type(child) is not dict:
            return False
        current = child
    current["first_send_completed"] = True
    return True


@dataclass(frozen=True)
class _Views:
    identity: tuple
    raw_object: object = field(repr=False)
    normalized_object: object = field(repr=False)
    raw: object = field(repr=False)
    normalized: object = field(repr=False)


@dataclass(frozen=True)
class _FirstSendSource:
    views: _Views
    app_config: object = field(repr=False)
    aliases: tuple[bool, bool]
    bootstrap: object = field(repr=False)


@dataclass(frozen=True)
class _FirstSendReceipt:
    source: object = field(repr=False)
    original: _FirstSendSource = field(repr=False)
    published: _Views = field(repr=False)


def _views():
    if not _native_config_current():
        return None
    with config._settings_rebuild_lock(), config._config_file_lock():
        cache_lock = config._SETTINGS_CACHE_LOCK
        if cache_lock is None:
            return None
        with cache_lock:
            raw, normalized = config._CONFIG_CACHE, config.settings
            path = config._CONFIG_CACHE_SOURCE
            if (
                type(raw) is not dict
                or type(normalized) is not dict
                or config._SETTINGS_CACHE is not normalized
                or config._SETTINGS_CACHE_SOURCE != path
                or path is None
                or not _plain(raw)
                or not _plain(normalized)
            ):
                return None
            identity = config.current_config_identity()
            if identity[1] != str(path):
                return None
            return _Views(
                identity, raw, normalized, copy.deepcopy(raw), copy.deepcopy(normalized)
            )


def capture_first_send_source(source):
    """Retain only an already-published, exact native source view."""
    views = _views()
    app_config = source.app_config
    if (
        views is None
        or views.identity != source.config_identity
        or type(app_config) is not dict
    ):
        return None
    app_console = app_config.get("console")
    app_onboarding = (
        app_console.get("onboarding") if type(app_console) is dict else None
    )

    def aliases(root):
        console = root.get("console") if type(root) is dict else None
        onboarding = console.get("onboarding") if type(console) is dict else None
        return (
            root is app_config
            or type(console) is dict
            and console is app_console
            or type(onboarding) is dict
            and onboarding is app_onboarding
        )

    normalized = views.normalized_object
    bootstrap = config._FIRST_BOOTSTRAP_CREATION
    if (
        type(bootstrap) is not tuple
        or len(bootstrap) != 3
        or str(bootstrap[0]) != views.identity[1]
        or bootstrap[1] is not views.raw_object
        or type(bootstrap[2]) is not config.PrivatePathResult
        or bootstrap[2].status is not config.PrivatePathStatus.CREATED_PRIVATE
        or bootstrap[2].lexical_path != bootstrap[0]
    ):
        bootstrap = None
    return _FirstSendSource(
        views,
        app_config,
        (aliases(normalized), aliases(normalized.get("COMPREHENSIVE_CONFIG_RAW"))),
        bootstrap,
    )


def publish_first_send(source):
    """Run the original fixed-key transaction and retain only its proven result."""
    original = source.first_send_source
    if original is None or not _native_config_current():
        config.save_setting_to_cli_config(
            "console.onboarding", "first_send_completed", True
        )
        return None
    initial = original.views
    eligible = []
    expected_original_raw = copy.deepcopy(initial.raw)
    expected_original_normalized = copy.deepcopy(initial.normalized)
    bootstrap = original.bootstrap
    transient = (
        bootstrap is not None
        and expected_original_raw.get("_first_run") is True
        and type(expected_original_normalized.get("COMPREHENSIVE_CONFIG_RAW")) is dict
        and expected_original_normalized["COMPREHENSIVE_CONFIG_RAW"].get("_first_run")
        is True
    )
    if transient:
        del expected_original_raw["_first_run"]
        del expected_original_normalized["COMPREHENSIVE_CONFIG_RAW"]["_first_run"]

    def build(snapshot):
        current = _views()
        expected = copy.deepcopy(initial.normalized)
        valid = all(
            not alias or _set_flag(expected, prefix)
            for alias, prefix in zip(
                original.aliases, ((), ("COMPREHENSIVE_CONFIG_RAW",))
            )
        )
        eligible.append(
            bool(
                valid
                and current is not None
                and snapshot.generation == initial.identity[0]
                and str(snapshot.config_path) == initial.identity[1]
                and current.identity == initial.identity
                and current.raw_object is initial.raw_object
                and current.normalized_object is initial.normalized_object
                and _same(current.raw, initial.raw)
                and _same(snapshot.effective_values, expected_original_raw)
                and (
                    not transient
                    or "_first_run" not in snapshot.raw_values
                    and config._FIRST_BOOTSTRAP_CREATION is bootstrap
                )
                and _same(current.normalized, expected)
                and source.app.app_config is original.app_config
                and _native_config_current()
            )
        )
        return config.LiteralSettingsMutation(
            {("console", "onboarding"): {"first_send_completed": True}}, {}
        )

    result = config._apply_literal_settings_transaction_locked(
        build,
        mutation_precondition=None,
        publish_noop=False,
        validate_literal_targets=False,
    )
    current = _views()
    expected_raw = copy.deepcopy(expected_original_raw)
    expected_normalized = copy.deepcopy(expected_original_normalized)
    if not (
        len(eligible) == 1
        and eligible[0]
        and result.file_replaced
        and result.caches_reloaded
        and result.failure_phase is None
        and current is not None
        and _set_flag(expected_raw)
        and _set_flag(expected_normalized)
        and _set_flag(expected_normalized, ("COMPREHENSIVE_CONFIG_RAW",))
        and current.identity == (initial.identity[0] + 1, initial.identity[1])
        and _same(current.raw, expected_raw)
        and _same(current.normalized, expected_normalized)
        and _same(result.settings_view, current.normalized)
        and (not transient or config._FIRST_BOOTSTRAP_CREATION is bootstrap)
        and source.app.app_config is original.app_config
        and _native_config_current()
    ):
        return None
    return _FirstSendReceipt(source, original, current)


def first_send_receipt_current(source, receipt):
    if type(receipt) is not _FirstSendReceipt or receipt.source is not source:
        return False
    expected = receipt.published
    current = _views()
    return bool(
        current is not None
        and current.identity == expected.identity
        and current.raw_object is expected.raw_object
        and current.normalized_object is expected.normalized_object
        and _same(current.raw, expected.raw)
        and _same(current.normalized, expected.normalized)
        and (
            receipt.original.bootstrap is None
            or config._FIRST_BOOTSTRAP_CREATION is receipt.original.bootstrap
        )
        and source.app.app_config is receipt.original.app_config
        and _native_config_current()
    )


def _stop_producers_current():
    for name, record in _NATIVE_STOP_RECEIPT_FUNCTIONS:
        function, code, defaults, kwdefaults, entries, closure, cells = record
        if (
            globals().get(name) is not function
            or function.__code__ is not code
            or function.__defaults__ is not defaults
            or function.__kwdefaults__ is not kwdefaults
            or function.__closure__ is not closure
            or kwdefaults is not None
            and (
                len(kwdefaults) != len(entries)
                or any(kwdefaults.get(key) is not value for key, value in entries)
            )
            or any(cell.cell_contents is not value for cell, value in cells)
        ):
            return False
    return True


from .console_ordinary_tree import _function_record as _record_retirement_function

_NATIVE_STOP_RECEIPT_FUNCTIONS = tuple(
    (name, _record_retirement_function(globals()[name]))
    for name in (
        "capture_first_send_source",
        "publish_first_send",
        "first_send_receipt_current",
        "_stop_producers_current",
        "_views",
        "_native_config_current",
        "_plain",
        "_same",
        "_set_flag",
    )
)
_NATIVE_STOP_RECEIPT_TYPES = (_Views, _FirstSendSource, _FirstSendReceipt)
_NATIVE_STOP_PRODUCER_FENCE = (
    _stop_producers_current,
    _stop_producers_current.__code__,
    _stop_producers_current.__defaults__,
    _stop_producers_current.__kwdefaults__,
    _stop_producers_current.__closure__,
)
