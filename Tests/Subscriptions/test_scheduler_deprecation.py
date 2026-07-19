"""Tests that the legacy Subscriptions scheduler modules emit deprecation warnings."""

import importlib
import warnings

from tldw_chatbook.Subscriptions import scheduler as scheduler_module
from tldw_chatbook.Subscriptions import textual_scheduler_worker as worker_module


def _reload_and_capture(module):
    """Reload ``module`` and return warnings emitted during import."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(module)
    return [w for w in caught if issubclass(w.category, DeprecationWarning)]


def test_subscription_scheduler_module_emits_deprecation_warning():
    """Importing the legacy scheduler module must emit a DeprecationWarning."""
    deprecation_warnings = _reload_and_capture(scheduler_module)
    messages = [str(w.message) for w in deprecation_warnings]
    assert any(
        "SubscriptionScheduler is deprecated" in msg for msg in messages
    ), f"Expected deprecation warning not found in: {messages}"


def test_textual_scheduler_worker_module_emits_deprecation_warning():
    """Importing the legacy worker module must emit a DeprecationWarning."""
    deprecation_warnings = _reload_and_capture(worker_module)
    messages = [str(w.message) for w in deprecation_warnings]
    assert any(
        "SubscriptionSchedulerWorker is deprecated" in msg for msg in messages
    ), f"Expected deprecation warning not found in: {messages}"


def test_package_import_of_non_deprecated_symbol_does_not_emit_scheduler_warning():
    """Importing a non-deprecated symbol from the package must not trigger the scheduler deprecation warning."""
    import tldw_chatbook.Subscriptions as subscriptions_module

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(subscriptions_module)
        # Access a non-deprecated symbol that is eagerly exported.
        _ = subscriptions_module.LocalWatchlistsService

    scheduler_warnings = [
        w for w in caught
        if issubclass(w.category, DeprecationWarning)
    ]
    assert not scheduler_warnings, (
        f"Unexpected deprecation warnings on package import: "
        f"{[str(w.message) for w in scheduler_warnings]}"
    )


def test_lazy_deprecated_symbol_is_accessible():
    """Deprecated scheduler symbols remain accessible via PEP 562 lazy loading."""
    import tldw_chatbook.Subscriptions as subscriptions_module

    importlib.reload(subscriptions_module)
    cls = subscriptions_module.SubscriptionScheduler
    assert cls.__name__ == "SubscriptionScheduler"
