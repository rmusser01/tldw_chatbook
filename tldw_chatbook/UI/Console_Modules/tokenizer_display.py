"""Finite, display-only preparation of the original tokenizer backends.

The caller owns worker scheduling and retirement. A successful result is only
readiness for the captured backend; it never replaces a storage admission.
"""

# Exact types prevent qualification from invoking custom containers/callbacks.
# ruff: noqa: E721

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from types import FunctionType, ModuleType

from ...Utils import custom_tokenizers as custom
from ...Utils import token_counter as counter


_MANAGER_TYPE = custom.CustomTokenizerManager
_MANAGER_METHODS = (
    "__init__",
    "_load_mappings",
    "has_tokenizers",
    "get_tokenizer_for_model",
    "load_tokenizer",
    "count_tokens",
)
_FUNCTIONS = tuple(
    (
        namespace,
        name,
        function,
        function.__code__,
        function.__defaults__,
        function.__kwdefaults__,
    )
    for namespace, names in (
        (
            vars(counter),
            (
                "estimate_tokens",
                "count_tokens_messages",
                "get_tiktoken_encoding",
                "count_tokens_tiktoken",
            ),
        ),
        (
            vars(custom),
            (
                "get_tokenizer_manager",
                "custom_tokenizers_available",
                "count_tokens_with_custom",
            ),
        ),
        (vars(_MANAGER_TYPE), _MANAGER_METHODS),
    )
    for name in names
    for function in (namespace[name],)
)
_CUSTOM_TOKENIZER_TYPE = getattr(custom, "Tokenizer", None)
_PREPARATION_LOCK = threading.Lock()


def _sources_current() -> bool:
    return (
        sys.modules.get(counter.__name__) is counter
        and sys.modules.get(custom.__name__) is custom
        and custom.CustomTokenizerManager is _MANAGER_TYPE
        and getattr(custom, "Tokenizer", None) is _CUSTOM_TOKENIZER_TYPE
        and counter.custom_tokenizers_available is custom.custom_tokenizers_available
        and counter.count_tokens_with_custom is custom.count_tokens_with_custom
        and all(
            namespace.get(name) is function
            and function.__code__ is code
            and function.__defaults__ is defaults
            and function.__kwdefaults__ is kwdefaults
            for namespace, name, function, code, defaults, kwdefaults in _FUNCTIONS
        )
    )


def _manager_snapshot(manager):
    if type(manager) is not _MANAGER_TYPE:
        return None
    fields = vars(manager)
    if any(name in fields for name in _MANAGER_METHODS):
        return None
    mappings, loaded = fields.get("_model_mappings"), fields.get("_tokenizers")
    directory = fields.get("tokenizers_dir")
    if (
        type(directory) is not str
        or type(mappings) is not dict
        or type(loaded) is not dict
        or any(
            type(key) is not str or type(value) is not str
            for key, value in mappings.items()
        )
        or any(
            type(key) is not str or type(value) is not _CUSTOM_TOKENIZER_TYPE
            for key, value in loaded.items()
        )
    ):
        return None
    return directory, mappings, tuple(mappings.items()), loaded, tuple(loaded.items())


def _same_manager_snapshot(manager, snapshot) -> bool:
    current = _manager_snapshot(manager)
    return current is not None and (
        current[0] == snapshot[0]
        and current[1] is snapshot[1]
        and current[2] == snapshot[2]
        and current[3] is snapshot[3]
        and len(current[4]) == len(snapshot[4])
        and all(
            key == old_key and value is old_value
            for (key, value), (old_key, old_value) in zip(current[4], snapshot[4])
        )
    )


def _tiktoken_snapshot():
    """Inspect only resident package callbacks; importing stays on the worker."""
    module = sys.modules.get("tiktoken")
    if module is None:
        return ()
    if type(module) is not ModuleType:
        return None
    records = []
    for name, defining_name in (
        ("get_encoding", "tiktoken.registry"),
        ("encoding_for_model", "tiktoken.model"),
    ):
        function = vars(module).get(name)
        defining = sys.modules.get(defining_name)
        if (
            type(function) is not FunctionType
            or type(defining) is not ModuleType
            or function.__globals__ is not vars(defining)
            or function.__code__.co_filename != vars(defining).get("__file__")
            or vars(defining).get(name) is not function
        ):
            return None
        records.append((module, name, function, function.__code__))
    registry = sys.modules.get("tiktoken.registry")
    core = sys.modules.get("tiktoken.core")
    if type(registry) is not ModuleType or type(core) is not ModuleType:
        return None
    encoding_type = vars(core).get("Encoding")
    encodings = vars(registry).get("ENCODINGS")
    if (
        vars(module).get("Encoding") is not encoding_type
        or type(encodings) is not dict
        or any(type(value) is not encoding_type for value in encodings.values())
    ):
        return None
    return tuple(records)


def _encoding_mapping_snapshot():
    mappings = [counter.TIKTOKEN_MODEL_ENCODINGS]
    model_module = sys.modules.get("tiktoken.model")
    if model_module is not None:
        if type(model_module) is not ModuleType:
            return None
        mappings.extend(
            vars(model_module).get(name)
            for name in (
                "MODEL_TO_ENCODING",
                "MODEL_PREFIX_TO_ENCODING",
            )
        )
    records = []
    for mapping in mappings:
        if type(mapping) is not dict:
            return None
        items = tuple(mapping.items())
        if any(type(key) is not str or type(value) is not str for key, value in items):
            return None
        records.append((mapping, items))
    return tuple(records)


def _same_encoding_mappings(before, after) -> bool:
    # A cold worker may import tiktoken, adding its two resident model maps.
    return (
        after is not None
        and len(after) >= len(before)
        and all(
            old_mapping is mapping and old_items == items
            for (old_mapping, old_items), (mapping, items) in zip(before, after)
        )
    )


@dataclass(eq=False)
class _Preparation:
    model: str
    provider: str
    flags: tuple[bool, bool, bool]
    manager: object = None
    manager_state: tuple | None = None
    encoding_sources: tuple = ()
    encoding_mappings: tuple = ()
    encoding: object = None
    ready: bool = False
    _heuristic_selection: bool = False

    def current(self) -> bool:
        if self._heuristic_selection:
            return False
        flags = (
            counter.CUSTOM_TOKENIZERS_AVAILABLE,
            custom.TOKENIZERS_AVAILABLE,
            counter.TIKTOKEN_AVAILABLE,
        )
        if (
            any(type(flag) is not bool for flag in flags)
            or not _sources_current()
            or self.flags != flags
        ):
            return False
        if not _same_encoding_mappings(
            self.encoding_mappings, _encoding_mapping_snapshot()
        ):
            return False
        if self.flags[0] and self.flags[1]:
            manager = custom._tokenizer_manager
            if manager is not self.manager:
                return False
            if manager is not None and not _same_manager_snapshot(
                manager, self.manager_state
            ):
                return False
        current_encoding = _tiktoken_snapshot()
        if current_encoding is None:
            return False
        if self.encoding_sources and (
            len(current_encoding) != len(self.encoding_sources)
            or any(
                any(left is not right for left, right in zip(row, original))
                for row, original in zip(current_encoding, self.encoding_sources)
            )
        ):
            return False
        if self.encoding is not None:
            registry = sys.modules.get("tiktoken.registry")
            encodings = (
                vars(registry).get("ENCODINGS")
                if type(registry) is ModuleType
                else None
            )
            if type(encodings) is not dict or not any(
                value is self.encoding for value in encodings.values()
            ):
                return False
        return True

    def __call__(self) -> bool:
        # Only our finite preparations share this lock. An unrelated original
        # getter can race construction; its published owner invalidates ours.
        with _PREPARATION_LOCK:
            if not self.current():
                return False
            self.ready = False
            try:
                if self.flags[0] and self.flags[1]:
                    manager = custom.get_tokenizer_manager()
                    state = _manager_snapshot(manager)
                    if state is None or custom._tokenizer_manager is not manager:
                        return False
                    self.manager, self.manager_state = manager, state
                    if state[2]:
                        # Heuristic directory discovery has no stable in-memory
                        # selection. Keep that route on its original caller.
                        if self.model not in state[1]:
                            # Cold construction can discover a mapping shape
                            # capture could not inspect. Recapture next time so
                            # it takes the same direct path as a warm manager.
                            self._heuristic_selection = True
                            return False
                        tokenizer = manager.get_tokenizer_for_model(
                            self.model, self.provider.strip().lower()
                        )
                        if type(tokenizer) is not _CUSTOM_TOKENIZER_TYPE:
                            return False
                        loaded_state = _manager_snapshot(manager)
                        if loaded_state is None or (
                            loaded_state[0] != state[0]
                            or loaded_state[1] is not state[1]
                            or loaded_state[2] != state[2]
                        ):
                            return False
                        self.manager_state = loaded_state
                        self.ready = self.current()
                        return self.ready
                if self.flags[2]:
                    self.encoding = counter.get_tiktoken_encoding(self.model)
                    if self.encoding is None:
                        return False
                    sources = _tiktoken_snapshot()
                    mappings = _encoding_mapping_snapshot()
                    if sources is None or not _same_encoding_mappings(
                        self.encoding_mappings, mappings
                    ):
                        return False
                    self.encoding_sources = sources
                    self.encoding_mappings = mappings
                self.ready = self.current()
                return self.ready
            except Exception:
                # No ready publication on admission/loading failure; a later
                # finite request may retry, never the synchronous display path.
                return False


def capture_tokenizer_preparation(
    model: str, provider: str
) -> Callable[[], bool] | None:
    """Capture stock backend preparation without loading or touching storage.

    Returns None for custom callbacks or an already-known heuristic mapping.
    The callable returns False on stale sources or a failed backend load.
    """
    if type(model) is not str or type(provider) is not str or not _sources_current():
        return None
    flags = (
        counter.CUSTOM_TOKENIZERS_AVAILABLE,
        custom.TOKENIZERS_AVAILABLE,
        counter.TIKTOKEN_AVAILABLE,
    )
    if any(type(flag) is not bool for flag in flags):
        return None
    manager = custom._tokenizer_manager if flags[0] and flags[1] else None
    state = _manager_snapshot(manager) if manager is not None else None
    if manager is not None and (state is None or state[2] and model not in state[1]):
        return None
    encoding_sources = _tiktoken_snapshot()
    encoding_mappings = _encoding_mapping_snapshot()
    if encoding_sources is None or encoding_mappings is None:
        return None
    return _Preparation(
        model, provider, flags, manager, state, encoding_sources, encoding_mappings
    )


def preparation_current(preparation: Callable[[], bool]) -> bool:
    """Check captured sources without asserting the callable has succeeded."""
    return type(preparation) is _Preparation and preparation.current()
