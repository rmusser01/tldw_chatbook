"""Interleave old/new CSS application on the same mounted Console (diagnostic)."""

from pathlib import Path
from statistics import median
import subprocess
import time
from types import ModuleType


def measure(app):
    from textual.css.stylesheet import Stylesheet
    from tldw_chatbook.Utils import textual_css_fastpath as current

    root = Path(__file__).resolve().parents[4]
    source = subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={root.as_posix()}",
            "show",
            "ac0c13581a54327b1fd38a3b6ddcce45cdb416e0:tldw_chatbook/Utils/textual_css_fastpath.py",
        ],
        cwd=root,
    )
    baseline = ModuleType("css_baseline")
    exec(compile(source, "css_baseline.py", "exec"), baseline.__dict__)
    patched = Stylesheet.apply
    Stylesheet.apply = patched.__wrapped__
    baseline.install_stylesheet_fastpath()
    original = Stylesheet.apply
    sheet = app.stylesheet
    nodes = list(app.screen.walk_children(with_self=True))
    results = {"before": [], "after": [], "after_1024": []}
    saved_limit = current._MAX_CANDIDATE_KEYS
    mismatches = 0
    try:
        for pair in range(4):
            expected = None
            for variant, apply, limit in (
                ("before", original, 256),
                ("after", patched, 256),
                ("after_1024", patched, 1024),
            ):
                Stylesheet.apply = apply
                current._MAX_CANDIDATE_KEYS = limit
                if hasattr(sheet, current._CACHE_ATTR):
                    delattr(sheet, current._CACHE_ATTR)
                sheet.update(app.screen)  # Warm each arm's own cache.
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
    finally:
        current._MAX_CANDIDATE_KEYS = saved_limit
        Stylesheet.apply = patched
        if hasattr(sheet, current._CACHE_ATTR):
            delattr(sheet, current._CACHE_ATTR)
    return {
        "nodes": len(nodes),
        "subject_ancestor_shapes": len(
            {
                (
                    frozenset(node._selector_names),
                    frozenset(current._ancestor_names(node)),
                )
                for node in nodes
            }
        ),
        "update_median_ms_by_pair": results,
        "style_mismatches": mismatches,
    }
