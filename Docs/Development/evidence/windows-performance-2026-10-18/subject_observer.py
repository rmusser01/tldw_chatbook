"""Count remaining real selector matching, separate from plain timings."""

from collections import Counter
import time

from textual.css.stylesheet import Stylesheet


def measure(app):
    original_descriptor = Stylesheet.__dict__["_check_rule"]
    original = Stylesheet._check_rule
    calls, matches, durations = Counter(), Counter(), Counter()

    def observed(cls, rule, nodes):
        key = ", ".join(
            " ".join(
                f"{selector.combinator.name}/{selector.type.name}:{selector.name}"
                + "".join(f":{pseudo}" for pseudo in sorted(selector.pseudo_classes))
                for selector in alternative.selectors
            )
            for alternative in rule.selector_set
        )
        start = time.perf_counter()
        result = tuple(original(rule, nodes))
        durations[key] += (time.perf_counter() - start) * 1000
        calls[key] += 1
        matches[key] += bool(result)
        return result

    try:
        Stylesheet._check_rule = classmethod(observed)
        app.stylesheet.update(app.screen)
    finally:
        Stylesheet._check_rule = original_descriptor
    return {
        "calls": sum(calls.values()),
        "matches": sum(matches.values()),
        "rules": [
            {
                "selector": key,
                "calls": calls[key],
                "matches": matches[key],
                "ms": elapsed,
            }
            for key, elapsed in durations.most_common()
        ],
    }
