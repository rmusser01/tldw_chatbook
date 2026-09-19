"""Opt-in TASK-32813 qualification plugin; compare real mounted consumer trees.

Run selected consumer tests with this directory on PYTHONPATH and
PYTEST_PLUGINS=migration_parity. This is an audit of the pre-migration CSS,
not a replacement for the consumers' behavior assertions. Only base computed
stylesheet rules are compared; inline styles are unchanged by this migration.
"""

import json
import os
from dataclasses import replace
from pathlib import Path
from weakref import WeakKeyDictionary

from textual.css.stylesheet import CssSource, Stylesheet
from textual.layout import Layout
from textual.pilot import Pilot

from tldw_chatbook.css import build_css, widget_css

HERE = Path(__file__).parent
PACKAGE = HERE.parents[3] / "tldw_chatbook"
ORIGINAL = {
    row["class_name"]: row
    for row in json.loads((HERE / "original-defaults.json").read_text())
}
OLD_BUNDLED = {
    row["class_name"]: row
    for row in json.loads((HERE / "original-toolpack-defaults.json").read_text())
}
TARGETS = ORIGINAL.keys() | OLD_BUNDLED.keys()
BLOCKS = [
    replace(block, css=OLD_BUNDLED[block.class_name]["css"])
    if block.class_name in OLD_BUNDLED
    else block
    for block in widget_css.iter_blocks(PACKAGE, widget_css.WIDGET_ATTR)
    if block.class_name not in ORIGINAL
]
OLD_STREAMS = dict(
    zip(
        (
            build_css.WIDGET_DEFAULTS_SELF_FILENAME,
            build_css.WIDGET_DEFAULTS_SCOPED_FILENAME,
        ),
        widget_css.render_stylesheets(BLOCKS, "baseline", scope_every_selector=True),
        strict=True,
    )
)
_cache = WeakKeyDictionary()
_records = {}
_mismatches = []
_cases = []
_outcomes = []
_original_pause = Pilot.pause


def _snapshot_sheet(sheet, node):
    """Use upstream cascade resolution without replacing the node's styles."""
    captured = {}
    sheet.replace_rules = lambda _node, rules, **_kwargs: captured.update(rules)
    sheet._process_component_classes = lambda _node: None
    upstream = getattr(Stylesheet.apply, "__wrapped__", Stylesheet.apply)
    flags = {
        key: getattr(node, key)
        for key in (
            "_has_hover_style",
            "_has_focus_within",
            "_has_order_style",
            "_has_odd_or_even",
        )
    }
    try:
        upstream(sheet, node, animate=False)
    finally:
        for key, value in flags.items():
            setattr(node, key, value)
    # Textual constructs a fresh layout strategy while parsing each sheet;
    # those objects compare by identity. Compare the CSS strategy itself, not
    # its incidental instance identity or arrangement cache.
    layout = captured.get("layout")
    if isinstance(layout, Layout):
        captured["layout"] = (
            type(layout).__module__,
            type(layout).__qualname__,
            layout.name,
        )
    return captured


def compare(app):
    nodes = list(app.query("*"))
    for screen in app.screen_stack:
        nodes.extend([screen, *screen.query("*")])
    nodes = list(dict.fromkeys(nodes))
    present = {
        base.__name__: -depth
        for node in nodes
        for depth, base in enumerate(node._node_bases)
        if base.__name__ in ORIGINAL
    }
    signature = (
        tuple(app.stylesheet.source.items()),
        tuple(sorted(present.items())),
        app.theme,
    )
    cached = _cache.get(app)
    if cached is None or cached[0] != signature:
        actual = app.stylesheet.copy()
        old = app.stylesheet.copy()
        for key, source in tuple(old.source.items()):
            filename = Path(key[0]).name
            if filename in OLD_STREAMS:
                old.source[key] = source._replace(content=OLD_STREAMS[filename])
        for name, tie_breaker in present.items():
            row = ORIGINAL[name]
            key = (str(PACKAGE / row["module"]), f"{name}.DEFAULT_CSS")
            old.source[key] = CssSource(row["css"], True, tie_breaker, name)
        actual.parse()
        old.parse()
        _cache[app] = (signature, actual, old)
    else:
        _, actual, old = cached
    for node in nodes:
        owners = {
            base.__name__
            for parent in [node, *node.ancestors]
            for base in parent._node_bases
            if base.__name__ in TARGETS
        }
        if not owners:
            continue
        before = _snapshot_sheet(old, node)
        after = _snapshot_sheet(actual, node)
        state = tuple(sorted(node.get_pseudo_classes()))
        for owner in owners:
            record = _records.setdefault(owner, set())
            record.add((type(node).__name__, node.id or "", state))
        if before != after:
            differences = {
                key: [repr(before.get(key)), repr(after.get(key))]
                for key in before.keys() | after.keys()
                if before.get(key) != after.get(key)
            }
            mismatch = {
                "owners": sorted(owners),
                "node": f"{type(node).__name__}#{node.id}",
                "state": state,
                "differences": differences,
            }
            if mismatch not in _mismatches:
                _mismatches.append(mismatch)


async def _pause(self, *args, **kwargs):
    result = await _original_pause(self, *args, **kwargs)
    compare(self.app)
    return result


def pytest_configure(config):
    Pilot.pause = _pause


def pytest_collection_modifyitems(items):
    _cases.extend(item.nodeid for item in items)
    if os.environ.get("TLDW_CSS_PARITY_ISOLATED_NODE"):
        node = os.environ["TLDW_TEST_PRIVATE_PROFILE_NODE"]
        assert len(items) == 1 and items[0].nodeid == node
        # Qualification driver selected this node's profile before collection.
        # Retain it through the existing source-bound fixture; no fixture or
        # original test body is replaced and no profile is shared across cases.
        items[0].obj._private_profile_test = True


def pytest_runtest_logreport(report):
    _outcomes.append(
        {
            "node": report.nodeid,
            "phase": report.when,
            "outcome": report.outcome,
            "wasxfail": getattr(report, "wasxfail", None),
        }
    )


def pytest_sessionfinish(session, exitstatus):
    Pilot.pause = _original_pause
    directory = Path(os.environ["TLDW_CSS_PARITY_REPORT_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{os.getpid()}.json").write_text(
        json.dumps(
            {
                "exitstatus": 1 if _mismatches else int(exitstatus),
                "cases": _cases,
                "outcomes": _outcomes,
                "classes": {
                    name: sorted(states) for name, states in sorted(_records.items())
                },
                "mismatches": _mismatches,
            },
            indent=2,
        )
    )
    if _mismatches:
        session.exitstatus = 1
