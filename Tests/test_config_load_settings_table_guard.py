"""Guard: every top-level config table the app reads from its loaded settings
survives ``load_settings()`` (TASK-34000.5, review finding L-02).

``load_settings()`` builds its return dict by hand, table by table. A table
that the shipped default config writes and that a feature reads through
``app_config.get("<table>")`` can therefore be silently dropped on the way
from the TOML to ``app.app_config`` -- the Media Analysis tab said "No
analysis provider is configured" for every user because
``[analysis_defaults]`` was never carried over, although the raw TOML had it
and the readiness resolver accepts it.

The set of tables this guard checks is DERIVED from the package source: an
AST pass over every ``.get("<key>")`` / ``["<key>"]`` read whose receiver is
the settings object (``*.app_config``, a bare ``app_config`` parameter, a
``load_settings()`` call, or a local bound from one of those). A hand-written
list of tables would go stale the day a new reader lands; this one grows with
the code. Two explicit, pinned lists bound it:

* :data:`DERIVATION_BOUNDARY` -- reads the pass finds that are not TOML
  tables the loader could carry (each entry must still be read, or it fails);
* :data:`PRE_EXISTING_DROPS` -- tables the loader already dropped when this
  guard was written, measured on the wave base. The guard fails on any drop
  outside this register, and it fails the other way too: an entry whose
  table is carried through later, or whose reader disappears, must be
  removed. The register can only shrink.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Mapping

import pytest

import tldw_chatbook
from Tests.Backup_Recovery.config_test_support import install_config_source

PACKAGE_ROOT = Path(tldw_chatbook.__file__).resolve().parent

#: The attribute / callable the derivation treats as "the settings object".
SETTINGS_ATTRIBUTE = "app_config"
SETTINGS_LOADER = "load_settings"

#: Reads the derivation finds that are NOT top-level TOML tables the loader
#: could carry, each with the reason. Pinned exact by
#: ``test_boundary_entries_are_still_read``.
DERIVATION_BOUNDARY: dict[str, str] = {
    # TTS/TTS_Backends.py reads the dotted string "api_settings.openai" as a
    # flat key; load_settings() exposes [api_settings.openai] as
    # settings["api_settings"]["openai"]. A reader quirk, not a table.
    "api_settings.openai": "dotted key read as a flat key by a reader",
    # app.py: runtime flag the bootstrap injects into the loaded mapping
    # (config.py ``loaded_config["_first_run"] = True``), never a TOML table.
    "_first_run": "runtime flag injected by the bootstrap, not a TOML table",
}

#: Tables ``load_settings()`` already dropped on the wave base (3027850197),
#: with one read site each. They are the same defect class as L-02 and are
#: NOT fixed by TASK-34000.5 (several change subsystem behaviour -- TLS
#: trust, TTS backends, RAG service constants -- and need their own
#: verification). Pinned exact by ``test_pre_existing_drops_register_is_exact``:
#: carry a table through, then delete its row here.
PRE_EXISTING_DROPS: dict[str, str] = {
    "API": "LLM_Calls/LLM_API_Calls.py:4148 (legacy [API] fallback)",
    "AppRAGSearchConfig": "UI/Screens/settings_library_rag_defaults.py:256",
    "HiggsSettings": "TTS/TTS_Backends.py:288",
    "OmniVoiceSettings": "TTS/TTS_Backends.py:352",
    "app_tts": "Event_Handlers/TTS_Events/tts_events.py:1553",
    "canvas": "UI/Screens/settings_screen.py:31703",
    "chat": "UI/Screens/settings_screen.py:7450",
    "custom_endpoints": "Chat/custom_endpoint_registry.py:250",
    "custom_openai_2_api": "LLM_Calls/LLM_API_Calls_Local.py:2348 (misspelt legacy fallback)",
    "encryption": "UI/Screens/settings_privacy_security.py:139",
    "global_tts_settings": "TTS/TTS_Backends.py:194",
    "home": "UI/Screens/home_screen.py:817",
    "llm_management": "UI/LLM_Management_Window.py:673",
    "media_creation": "Media_Creation/swarmui_client.py:108",
    "network": "UI/Screens/settings_network_defaults.py:50",
    "provider_setup": "Chat/provider_setup_persistence.py:1131",
    "rag": "RAG_Search/simplified/rag_service.py:97",
    "tldw_api": "config.py:1532",
    "transcription": "UI/Wizards/first_run_speech_step.py:179",
}

#: Marker key written into every table the guard's TOML defines.
GUARD_SENTINEL = "_guard_sentinel"


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


def _callee_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _own_statements(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """Statements of one scope: descend into compound statements, never into
    nested function or class bodies (those are their own scope)."""
    for stmt in body:
        yield stmt
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for field in ("body", "orelse", "finalbody", "handlers"):
            sub = getattr(stmt, field, None)
            if not isinstance(sub, list):
                continue
            for item in sub:
                if isinstance(item, ast.ExceptHandler):
                    yield from _own_statements(item.body)
                elif isinstance(item, ast.stmt):
                    yield from _own_statements([item])
        if isinstance(stmt, ast.Match):
            for case in stmt.cases:
                yield from _own_statements(case.body)


class _SettingsReadFinder(ast.NodeVisitor):
    """Collect ``<settings>.get("<key>")`` and ``<settings>["<key>"]`` reads."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.reads: dict[str, list[str]] = defaultdict(list)
        self._scopes: list[set[str]] = []

    def _is_settings(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Attribute) and node.attr == SETTINGS_ATTRIBUTE:
            return True
        if isinstance(node, ast.Name):
            if node.id == SETTINGS_ATTRIBUTE:
                return True
            return bool(self._scopes) and node.id in self._scopes[-1]
        if isinstance(node, ast.Call) and _callee_name(node.func) == SETTINGS_LOADER:
            return True
        return False

    def _bind_aliases(self, body: list[ast.stmt]) -> None:
        scope = self._scopes[-1]
        changed = True
        while changed:
            changed = False
            for stmt in _own_statements(body):
                if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                    continue
                value = stmt.value
                if value is None or not self._is_settings(value):
                    continue
                targets = (
                    stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                )
                for target in targets:
                    if isinstance(target, ast.Name) and target.id not in scope:
                        scope.add(target.id)
                        changed = True

    def visit_Module(self, node: ast.Module) -> None:
        self._scopes.append(set())
        self._bind_aliases(node.body)
        self.generic_visit(node)
        self._scopes.pop()

    def _visit_scope(self, node) -> None:
        self._scopes.append(set(self._scopes[0]) if self._scopes else set())
        self._bind_aliases(node.body)
        self.generic_visit(node)
        self._scopes.pop()

    visit_FunctionDef = _visit_scope
    visit_AsyncFunctionDef = _visit_scope

    def _record(self, key: str, lineno: int) -> None:
        self.reads[key].append(f"{self.label}:{lineno}")

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "get"
            and self._is_settings(func.value)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            self._record(node.args[0].value, node.lineno)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if (
            self._is_settings(node.value)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            self._record(node.slice.value, node.lineno)
        self.generic_visit(node)


def settings_reads_in(source: str, label: str = "<source>") -> dict[str, list[str]]:
    """Top-level settings keys one module's source reads, with read sites."""
    finder = _SettingsReadFinder(label)
    finder.visit(ast.parse(source, filename=label))
    return {key: sorted(sites) for key, sites in finder.reads.items()}


def derive_settings_reads(package_root: Path = PACKAGE_ROOT) -> dict[str, list[str]]:
    """Every top-level key the package reads from the loaded settings.

    Returns:
        Mapping of key -> sorted ``path:line`` read sites (paths relative to
        the repository root).
    """
    reads: dict[str, list[str]] = defaultdict(list)
    for path in sorted(package_root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        # Sound prefilter: every receiver form the finder accepts spells one
        # of these two names, and an alias is bound from one of them in the
        # same file. Parsing only those files is the difference between ~15 s
        # and ~40 s over this 1.9M-line package.
        if SETTINGS_ATTRIBUTE not in source and SETTINGS_LOADER not in source:
            continue
        label = str(path.relative_to(package_root.parent))
        for key, sites in settings_reads_in(source, label).items():
            reads[key].extend(sites)
    return {key: sorted(sites) for key, sites in reads.items()}


def missing_tables(
    settings: Mapping, reads: Mapping[str, list[str]]
) -> dict[str, list[str]]:
    """The read keys absent from ``settings`` outside the two pinned lists.

    Args:
        settings: What ``load_settings()`` returned.
        reads: Output of :func:`derive_settings_reads`.

    Returns:
        key -> read sites, for every unexpected missing table.
    """
    return {
        key: sites
        for key, sites in reads.items()
        if key not in DERIVATION_BOUNDARY
        and key not in PRE_EXISTING_DROPS
        and key not in settings
    }


def _toml_defining(tables: list[str]) -> str:
    """A user config that defines every named table (with one sentinel key)."""
    return "".join(f"[{table}]\n{GUARD_SENTINEL} = true\n\n" for table in tables)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def derived_reads() -> dict[str, list[str]]:
    reads = derive_settings_reads()
    assert "analysis_defaults" in reads, "derivation lost the L-02 reader itself"
    assert len(reads) > 20, f"derivation collapsed to {sorted(reads)}"
    return reads


def _load(monkeypatch: pytest.MonkeyPatch, config_path: Path, toml_text: str) -> dict:
    """``load_settings()`` over ``toml_text`` via the fresh-module recipe.

    Under the per-test sandbox the shared config module is already bound to
    the bootstrap profile, and admission refuses a redirected
    ``load_settings(force_reload=True)`` on it with
    ``RecoveryRequired(raw_source_selection_changed)``; ``install_config_source``
    imports a fresh module bound to this file instead.
    """
    config_path.write_text(toml_text, encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    fresh = install_config_source(monkeypatch)
    return fresh.load_settings(force_reload=True)


@pytest.fixture(scope="module")
def loaded_settings(tmp_path_factory, derived_reads) -> Iterator[dict]:
    """``load_settings()`` over a file that DEFINES every table it might drop.

    Two passes. The first loads an empty user file: a key present then is
    produced unconditionally and needs no table in the file. The second
    defines only the keys the first pass did not produce -- a loader that
    carries a table only when the file has it (``buddy_interaction``) is
    correct, and only visible this way; and never defining the heavy,
    always-present tables (``[database]``, ``[logging]``...) keeps the load
    to seconds rather than minutes.
    """
    root = tmp_path_factory.mktemp("load-settings-guard")
    candidates = [
        key
        for key in sorted(derived_reads)
        if key not in DERIVATION_BOUNDARY
    ]
    with pytest.MonkeyPatch.context() as monkeypatch:
        unconditional = _load(monkeypatch, root / "empty.toml", "")
        # The L-02 table is always defined: the shipped defaults carry one,
        # so pass 1 sees it, but the defect was the USER's table not
        # surviving -- the sentinel proves that shape.
        defined = ["analysis_defaults"] + [
            key
            for key in candidates
            if key not in unconditional and key != "analysis_defaults"
        ]
        settings = _load(monkeypatch, root / "defined.toml", _toml_defining(defined))
        raw = settings.get("COMPREHENSIVE_CONFIG_RAW", {})
        for key in defined:
            assert raw.get(key, {}).get(GUARD_SENTINEL) is True, (
                f"the temp TOML's [{key}] did not reach load_settings()"
            )
        yield settings


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_every_table_the_app_reads_survives_load_settings(loaded_settings, derived_reads):
    missing = missing_tables(loaded_settings, derived_reads)
    detail = "\n".join(
        f"  {key}: read at {', '.join(sites[:3])}"
        for key, sites in sorted(missing.items())
    )
    assert not missing, (
        "load_settings() drops top-level config table(s) the app reads from "
        f"app_config -- carry each through in config.py's config_dict:\n{detail}"
    )


def test_analysis_defaults_is_carried_through(loaded_settings):
    """The L-02 table itself, named explicitly so the failure reads as L-02."""
    assert loaded_settings.get("analysis_defaults", {}).get(GUARD_SENTINEL) is True


def test_guard_reports_a_dropped_table(loaded_settings, derived_reads):
    """Negative control: the guard names a table that goes missing again."""
    dropped = dict(loaded_settings)
    del dropped["analysis_defaults"]

    missing = missing_tables(dropped, derived_reads)

    assert list(missing) == ["analysis_defaults"]
    assert any("media_viewer_panel.py" in site for site in missing["analysis_defaults"])


def test_pre_existing_drops_register_is_exact(loaded_settings, derived_reads):
    """Each registered table is still read AND still dropped; otherwise remove it."""
    unread = sorted(key for key in PRE_EXISTING_DROPS if key not in derived_reads)
    now_carried = sorted(
        key for key in PRE_EXISTING_DROPS if key in derived_reads and key in loaded_settings
    )
    assert not unread, f"PRE_EXISTING_DROPS lists tables nothing reads any more: {unread}"
    assert not now_carried, (
        "PRE_EXISTING_DROPS lists tables load_settings() now carries -- delete "
        f"their rows: {now_carried}"
    )


def test_boundary_entries_are_still_read(derived_reads):
    stale = sorted(key for key in DERIVATION_BOUNDARY if key not in derived_reads)
    assert not stale, f"DERIVATION_BOUNDARY lists keys nothing reads any more: {stale}"


def test_derivation_sees_every_reader_shape():
    """The receiver forms the package uses; losing one would shrink the guard silently."""
    source = (
        "from tldw_chatbook.config import load_settings\n"
        "class W:\n"
        "    def a(self):\n"
        "        return self.app_config.get('t_attr')\n"
        "    def b(self):\n"
        "        return self.app_instance.app_config['t_chain']\n"
        "def c(app_config):\n"
        "    return app_config.get('t_param', {})\n"
        "def d():\n"
        "    return load_settings().get('t_call')\n"
        "def e():\n"
        "    cfg = load_settings()\n"
        "    loaded = cfg\n"
        "    return loaded['t_alias']\n"
        "def f(other):\n"
        "    return other.get('t_not_settings')\n"
        "def g():\n"
        "    cfg = load_settings()\n"
        "    def inner(cfg):\n"
        "        return cfg\n"
        "    return cfg.get('t_outer')\n"
        "def h(cfg):\n"
        "    return cfg.get('t_other_scope')\n"
    )
    reads = settings_reads_in(source)
    assert set(reads) == {
        "t_attr", "t_chain", "t_param", "t_call", "t_alias", "t_outer"
    }, sorted(reads)
