"""Interleave exact previous/current CSS on one mounted screen."""

from pathlib import Path
from statistics import median
import subprocess
import time
from types import ModuleType

from textual.css.stylesheet import Stylesheet
from textual.css import stylesheet
from textual.css.match import _check_selectors as upstream_match


def measure(app, baseline_revision="f152dfc4a6c4fbf2fcd277543757a46a0d2874e2"):
    from subject_observer import measure as observe
    from tldw_chatbook.Utils import textual_css_fastpath as current

    root = Path(__file__).resolve().parents[4]
    source = subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={root.as_posix()}",
            "show",
            f"{baseline_revision}:tldw_chatbook/Utils/textual_css_fastpath.py",
        ],
        cwd=root,
    )
    baseline = ModuleType("matching_css_baseline")
    exec(compile(source, "matching_css_baseline.py", "exec"), baseline.__dict__)
    patched = Stylesheet.apply
    patched_match = stylesheet._check_selectors
    Stylesheet.apply = patched.__wrapped__
    stylesheet._check_selectors = upstream_match
    baseline.install_stylesheet_fastpath()
    original = Stylesheet.apply
    original_match = stylesheet._check_selectors
    sheet = app.stylesheet
    nodes = list(app.screen.walk_children(with_self=True))
    results = {"before": [], "after": []}
    cold = {"before": [], "after": []}
    observed = {}
    mismatches = 0
    try:
        for pair in range(4):
            expected = None
            arms = [("before", original), ("after", patched)]
            if pair % 2:
                arms.reverse()
            for variant, apply in arms:
                Stylesheet.apply = apply
                stylesheet._check_selectors = (
                    original_match if variant == "before" else patched_match
                )
                if hasattr(sheet, current._CACHE_ATTR):
                    delattr(sheet, current._CACHE_ATTR)
                start = time.perf_counter()
                sheet.update(app.screen)
                cold[variant].append((time.perf_counter() - start) * 1000)
                durations = []
                for _ in range(5):
                    start = time.perf_counter()
                    sheet.update(app.screen)
                    durations.append((time.perf_counter() - start) * 1000)
                results[variant].append(median(durations))
                resolved = [
                    (
                        dict(node.styles.base.get_rules()),
                        {
                            key: value.get_rules()
                            for key, value in node._component_styles.items()
                        },
                    )
                    for node in nodes
                ]
                if expected is None:
                    expected = resolved
                else:
                    mismatches += sum(a != b for a, b in zip(expected, resolved))
                if pair == 3:
                    observed[variant] = observe(app)
    finally:
        Stylesheet.apply = patched
        stylesheet._check_selectors = patched_match
        if hasattr(sheet, current._CACHE_ATTR):
            delattr(sheet, current._CACHE_ATTR)
    return {
        "nodes": len(nodes),
        "warm_ms": results,
        "cold_ms": cold,
        "style_mismatches": mismatches,
        "matcher": observed,
    }
