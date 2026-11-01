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
AST pass over every ``.get("<key>")`` / ``["<key>"]`` read (load context
only) whose receiver is the settings object. A hand-written list of tables
would go stale the day a new reader lands; this one grows with the code.

Derivation boundary -- what the pass SEES as the settings object:

* ``<anything>.app_config`` (any attribute chain) and the bare name
  ``app_config`` (a parameter or local of that name);
* a ``load_settings()`` call, and ``getattr(x, "app_config", ...)``;
* ``a or b`` / ``a if c else b`` whose operand / branch is one of the above
  (the ``(app_config or {}).get(...)`` idiom);
* a local bound from one of the above in the same scope, to a fixpoint
  (``cfg = load_settings(); loaded = cfg; loaded["x"]``);
* ONE level of call-site propagation: a function's parameter counts as the
  settings object when some call in the package passes one of the above into
  it, positionally or by keyword, matched by function NAME
  (``webhook_config_from_settings(load_settings())`` makes ``settings``
  inside that function a receiver).

What it does NOT see: a receiver reached through a helper's RETURN value
(``cfg = self._app_config_mapping(); cfg.get(...)``), propagation deeper
than one call, a key that is not a string literal, and reads of the raw
bootstrap mapping (``load_cli_config_and_ensure_existence()``), which is a
different object this guard does not cover. ``test_derivation_sees_every_
reader_shape`` pins every seen shape with negatives for the unseen ones.

Two explicit, pinned lists bound the result:

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
#: verification; TASK-34000.53 owns them). Pinned exact by
#: ``test_pre_existing_drops_register_is_exact``: carry a table through,
#: then delete its row here.
PRE_EXISTING_DROPS: dict[str, str] = {
    "API": "LLM_Calls/LLM_API_Calls.py:4148 (legacy [API] fallback)",
    "AppRAGSearchConfig": "UI/Screens/settings_library_rag_defaults.py:256",
    "HiggsSettings": "TTS/TTS_Backends.py:288",
    "OmniVoiceSettings": "TTS/TTS_Backends.py:352",
    "app_tts": "Event_Handlers/TTS_Events/tts_events.py:1553",
    # ``canvas`` left this register in fix round 1, but it IS read from
    # the loaded settings: settings_screen.py:6393 feeds app_config to
    # build_canvas_config_policy -> _normalize_canvas_execution, which
    # reads it at config.py:210. That is a two-level helper read, outside
    # this derivation's one-level boundary, so the guard cannot see it.
    # Its direct accesses (settings_screen.py:31700-31703) are a
    # setdefault and a store. TASK-34000.53 still lists it.
    "chat": "UI/Screens/settings_screen.py:7450",
    "custom_endpoints": "Chat/custom_endpoint_registry.py:250",
    "custom_openai_2_api": "LLM_Calls/LLM_API_Calls_Local.py:2348 (misspelt legacy fallback)",
    "encryption": "UI/Screens/settings_privacy_security.py:139",
    "global_tts_settings": "TTS/TTS_Backends.py:194",
    "home": "UI/Screens/home_screen.py:817",
    "llm_management": "UI/LLM_Management_Window.py:673",
    "media_creation": "Media_Creation/swarmui_client.py:108",
    # (app_config or {}).get(...) -- the BoolOp receiver shape (fix round 1).
    "model_capabilities": "UI/Screens/settings_context_memory.py:305",
    "network": "UI/Screens/settings_network_defaults.py:50",
    "provider_setup": "Chat/provider_setup_persistence.py:1131",
    "rag": "RAG_Search/simplified/rag_service.py:97",
    "tldw_api": "config.py:1532",
    "transcription": "UI/Wizards/first_run_speech_step.py:179",
    # Helper-parameter read reached by call-site propagation (fix round 1):
    # the Settings screen builds its Canvas policy from app_config
    # (UI/Screens/settings_screen.py:6393 -> build_canvas_config_policy),
    # whose ``values.get("web_server")`` therefore misses the user's table;
    # the runtime policy (get_canvas_config_policy) reads the raw TOML.
    "web_server": "config.py:374 via UI/Screens/settings_screen.py:6393",
    # Helper-parameter read reached by call-site propagation (fix round 1):
    # webhook_config_from_settings(load_settings()) at Agents/agent_service.py
    # :3882. A configured [webhooks] url never fires today.
    "webhooks": "Agents/run_webhooks.py:317 via Agents/agent_service.py:3882",
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


#: (defining module label, function name) -> positional indexes / keyword
#: names that receive the settings object somewhere in the package. The
#: module label is ``None`` while a finder runs (= the caller's own file);
#: the driver resolves it.
SettingsParams = Mapping[tuple, set]


def _module_labels(dotted: str) -> tuple[str, ...]:
    """Source labels a dotted module may live at (``a/b.py`` or ``a/b/__init__.py``)."""
    base = dotted.replace(".", "/")
    return (f"{base}.py", f"{base}/__init__.py")


def _resolve_relative(label: str, module: str, level: int) -> str:
    """Absolute dotted module for ``from <level dots><module> import ...`` in ``label``."""
    if level == 0:
        return module
    package = label[: -len("/__init__.py")] if label.endswith("/__init__.py") else label.rpartition("/")[0]
    parts = package.split("/")
    parts = parts[: len(parts) - (level - 1)] if level > 1 else parts
    base = ".".join(parts)
    return f"{base}.{module}" if module else base


class _SettingsReadFinder(ast.NodeVisitor):
    """Collect ``<settings>.get("<key>")`` and ``<settings>["<key>"]`` reads.

    Also records, for call-site propagation, which callee parameters receive
    the settings object (``call_args``, keyed by the callee's defining module
    as resolved through this module's imports). With ``settings_params``
    given, a matching function's parameters are bound as receivers in its
    scope.
    """

    def __init__(self, label: str, settings_params: SettingsParams | None = None) -> None:
        self.label = label
        self.reads: dict[str, list[str]] = defaultdict(list)
        self.call_args: dict[tuple, set] = defaultdict(set)
        self._settings_params = settings_params or {}
        self._scopes: list[set[str]] = []
        self._imports: dict[str, str] = {}

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._imports[alias.asname or alias.name.partition(".")[0]] = (
                alias.name if alias.asname else alias.name.partition(".")[0]
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = _resolve_relative(self.label, node.module or "", node.level)
        for alias in node.names:
            self._imports[alias.asname or alias.name] = f"{base}.{alias.name}"

    def _callee_target(self, func: ast.AST) -> tuple[str | None, str] | None:
        """(defining module or None for this file, function name) of a call."""
        if isinstance(func, ast.Name):
            imported = self._imports.get(func.id)
            if imported is None:
                return None, func.id
            module, _, name = imported.rpartition(".")
            return (module or None), name
        if isinstance(func, ast.Attribute):
            head, _, rest = ast.unparse(func.value).partition(".")
            imported = self._imports.get(head)
            if imported is None:
                # self.m(...) / obj.m(...): a method, resolved within this file.
                return None, func.attr
            return imported + (f".{rest}" if rest else ""), func.attr
        return None

    def _is_settings(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Attribute) and node.attr == SETTINGS_ATTRIBUTE:
            return True
        if isinstance(node, ast.Name):
            if node.id == SETTINGS_ATTRIBUTE:
                return True
            return bool(self._scopes) and node.id in self._scopes[-1]
        if isinstance(node, ast.Call):
            name = _callee_name(node.func)
            if name == SETTINGS_LOADER:
                return True
            return (
                name == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == SETTINGS_ATTRIBUTE
            )
        if isinstance(node, ast.BoolOp):
            return any(self._is_settings(value) for value in node.values)
        if isinstance(node, ast.IfExp):
            return self._is_settings(node.body) or self._is_settings(node.orelse)
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

    def _bind_parameters(self, node) -> None:
        """Bind the parameters a propagated call site feeds the settings into."""
        received = self._settings_params.get((self.label, node.name))
        if not received:
            return
        positional = [a.arg for a in node.args.posonlyargs + node.args.args]
        by_name = positional + [a.arg for a in node.args.kwonlyargs]
        # A method called as ``obj.m(x)`` has ``self`` in front of ``x``; the
        # call site cannot tell, so both candidates are bound (one extra
        # receiver inside a method at worst -- an over-approximation).
        offset = 1 if positional and positional[0] in {"self", "cls"} else 0
        for item in received:
            if isinstance(item, int):
                for index in (item, item + offset):
                    if index < len(positional):
                        self._scopes[-1].add(positional[index])
            elif item in by_name:
                self._scopes[-1].add(item)

    def visit_Module(self, node: ast.Module) -> None:
        self._scopes.append(set())
        self._bind_aliases(node.body)
        self.generic_visit(node)
        self._scopes.pop()

    def _visit_scope(self, node) -> None:
        self._scopes.append(set(self._scopes[0]) if self._scopes else set())
        self._bind_parameters(node)
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
        target = self._callee_target(func)
        if target is not None and target[1] != "get":
            for index, arg in enumerate(node.args):
                if self._is_settings(arg):
                    self.call_args[target].add(index)
            for keyword in node.keywords:
                if keyword.arg and self._is_settings(keyword.value):
                    self.call_args[target].add(keyword.arg)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # Load context only: ``s["x"] = v`` is a write, not a read; the inner
        # ``s["x"]`` of ``s["x"]["y"] = v`` is still a load and still counts.
        if (
            isinstance(node.ctx, ast.Load)
            and self._is_settings(node.value)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            self._record(node.slice.value, node.lineno)
        self.generic_visit(node)


def _reads_from_trees(
    trees: Mapping[str, ast.AST], deferred_sources: Mapping[str, str] | None = None
) -> dict[str, list[str]]:
    """Two passes over parsed modules: direct reads, then one propagation level.

    Pass 1 collects reads and every call site that feeds the settings object
    into a parameter. Pass 2 re-visits only the modules that DEFINE such a
    function, with its parameters bound as receivers -- including modules
    from ``deferred_sources`` (not parsed for pass 1) that textually define
    one of those functions.
    """
    deferred = dict(deferred_sources or {})
    reads: dict[str, list[str]] = defaultdict(list)
    settings_params: dict[tuple[str, str], set] = defaultdict(set)
    parsed: dict[str, ast.AST] = dict(trees)
    for label, tree in trees.items():
        finder = _SettingsReadFinder(label)
        finder.visit(tree)
        for key, sites in finder.reads.items():
            reads[key].extend(sites)
        for (module, name), received in finder.call_args.items():
            # A call resolves to the caller's own file, or to the file its
            # import names; a name that resolves nowhere in the package is
            # not propagated (so a same-named function elsewhere is not).
            candidates = (label,) if module is None else _module_labels(module)
            for candidate in candidates:
                if candidate in parsed or candidate in deferred:
                    settings_params[(candidate, name)] |= received
    for label in {label for label, _ in settings_params}:
        if label not in parsed:
            parsed[label] = ast.parse(deferred[label], filename=label)
        finder = _SettingsReadFinder(label, settings_params)
        finder.visit(parsed[label])
        for key, sites in finder.reads.items():
            reads[key].extend(sites)
    return {key: sorted(set(sites)) for key, sites in reads.items()}


def settings_reads_in(source: str, label: str = "<source>") -> dict[str, list[str]]:
    """Top-level settings keys one module's source reads, with read sites."""
    return _reads_from_trees({label: ast.parse(source, filename=label)})


def derive_settings_reads(package_root: Path = PACKAGE_ROOT) -> dict[str, list[str]]:
    """Every top-level key the package reads from the loaded settings.

    Returns:
        Mapping of key -> sorted ``path:line`` read sites (paths relative to
        the repository root).
    """
    trees: dict[str, ast.AST] = {}
    deferred: dict[str, str] = {}
    for path in sorted(package_root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        label = str(path.relative_to(package_root.parent))
        # Sound prefilter: every receiver form the finder accepts spells one
        # of these two names, an alias is bound from one of them in the same
        # file, and a propagated call site spells one of them too. The
        # callee's own module need not: those files are parsed below, once
        # pass 1 knows which function names receive the settings (the
        # ``webhooks`` reader lives in such a file). Parsing only these is
        # the difference between ~15 s and ~40 s over this 1.9M-line package.
        if SETTINGS_ATTRIBUTE in source or SETTINGS_LOADER in source:
            trees[label] = ast.parse(source, filename=label)
        else:
            deferred[label] = source
    return _reads_from_trees(trees, deferred)


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
    assert "analysis_defaults" in loaded_settings, (
        "the L-02 table is already missing; see "
        "test_every_table_the_app_reads_survives_load_settings"
    )
    dropped = dict(loaded_settings)
    dropped.pop("analysis_defaults")

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
    """The receiver forms the package uses; losing one would shrink the guard silently.

    Each ``t_*`` key is one seen shape; each ``n_*`` key is one unseen shape
    (the documented boundary) and must stay absent.
    """
    source = (
        "from tldw_chatbook.config import load_settings\n"
        "class W:\n"
        "    def a(self):\n"
        "        return self.app_config.get('t_attr')\n"
        "    def b(self):\n"
        "        return self.app_instance.app_config['t_chain']\n"
        "    def m(self, cfg):\n"
        "        return cfg.get('t_method_param')\n"
        "    def mapping(self):\n"
        "        return getattr(self.app, 'app_config', {})\n"
        "    def via_return(self):\n"
        "        cfg = self.mapping()\n"
        "        return cfg.get('n_helper_return')\n"
        "def c(app_config):\n"
        "    return app_config.get('t_param', {})\n"
        "def d():\n"
        "    return load_settings().get('t_call')\n"
        "def e():\n"
        "    cfg = load_settings()\n"
        "    loaded = cfg\n"
        "    return loaded['t_alias']\n"
        "def f(other):\n"
        "    return other.get('n_not_settings')\n"
        "def g():\n"
        "    cfg = load_settings()\n"
        "    def inner(cfg):\n"
        "        return cfg\n"
        "    return cfg.get('t_outer')\n"
        "def h(cfg):\n"
        "    return cfg.get('n_other_scope')\n"
        "def boolop(app_config):\n"
        "    return (app_config or {}).get('t_boolop')\n"
        "def ifexp(app_config, flag):\n"
        "    return (app_config if flag else {})['t_ifexp']\n"
        "def by_getattr(app):\n"
        "    return getattr(app, 'app_config', {}).get('t_getattr')\n"
        "def writes(app_config):\n"
        "    app_config['n_store'] = 1\n"
        "    app_config['t_nested_store']['y'] = 1\n"
        "def helper(settings):\n"
        "    return settings.get('t_helper_positional')\n"
        "def helper_kw(*, settings):\n"
        "    return settings.get('t_helper_keyword')\n"
        "def second_level(settings):\n"
        "    return settings.get('n_second_level')\n"
        "def never_fed(settings):\n"
        "    return settings.get('n_never_fed')\n"
        "def callers(self):\n"
        "    helper(load_settings())\n"
        "    helper_kw(settings=self.app_config)\n"
        "    W().m(self.app_config)\n"
        "    second_level(helper(load_settings()))\n"
        "    never_fed({})\n"
    )
    reads = settings_reads_in(source)
    expected = {
        "t_attr", "t_chain", "t_method_param", "t_param", "t_call", "t_alias",
        "t_outer", "t_boolop", "t_ifexp", "t_getattr", "t_nested_store",
        "t_helper_positional", "t_helper_keyword",
    }
    assert set(reads) == expected, sorted(set(reads) ^ expected)


def test_propagation_follows_imports_not_names():
    """A helper's parameter is bound only in the module the caller imports it from.

    ``webhook_config_from_settings(load_settings())`` lives in one module and
    its reader in another; a same-named function in a third module (which
    nobody feeds the settings) must stay unseen -- name-only matching bound
    ``Backup_Recovery/credentials.py``'s ``record`` as a receiver.
    """
    caller = (
        "from tldw_chatbook.config import load_settings\n"
        "from tldw_chatbook.pkg.reader import helper, relative_target\n"
        "from tldw_chatbook.pkg import reader as mod\n"
        "import tldw_chatbook.pkg.reader as aliased\n"
        "def run(self):\n"
        "    helper(load_settings())\n"
        "    mod.by_module(self.app_config)\n"
        "    aliased.by_alias(settings=load_settings())\n"
        "    relative_target(load_settings())\n"
    )
    reader = (
        "def helper(settings):\n"
        "    return settings.get('t_imported')\n"
        "def by_module(settings):\n"
        "    return settings.get('t_module_attr')\n"
        "def by_alias(*, settings):\n"
        "    return settings.get('t_alias_attr')\n"
        "def relative_target(settings):\n"
        "    return settings.get('t_relative')\n"
    )
    unrelated = (
        "def helper(record):\n"
        "    return record.get('n_same_name_elsewhere')\n"
    )
    sibling = (
        "from .reader import relative_target\n"
        "def run(cfg):\n"
        "    relative_target(cfg)\n"  # cfg is not a settings receiver here
    )
    reads = _reads_from_trees(
        {
            "tldw_chatbook/pkg/caller.py": ast.parse(caller),
            "tldw_chatbook/pkg/sibling.py": ast.parse(sibling),
        },
        {
            "tldw_chatbook/pkg/reader.py": reader,
            "tldw_chatbook/other/credentials.py": unrelated,
        },
    )
    assert set(reads) == {"t_imported", "t_module_attr", "t_alias_attr", "t_relative"}, (
        sorted(reads)
    )
    assert _resolve_relative("tldw_chatbook/pkg/sibling.py", "reader", 1) == (
        "tldw_chatbook.pkg.reader"
    )
    assert _resolve_relative("tldw_chatbook/pkg/sub/__init__.py", "", 2) == (
        "tldw_chatbook.pkg"
    )
