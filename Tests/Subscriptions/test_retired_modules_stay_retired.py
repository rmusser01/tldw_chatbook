"""TASK-1211: the retired subscription scheduler and briefing island stays retired.

Roughly 7,750 LOC of scheduling, briefing, aggregation and ingest code was
removed because it was unreachable: `textual_scheduler_worker` and
`subscription_backend_controller` imported each other, forming a closed cycle
with no external entry point, and `SubscriptionBackendController` drove a
`SubscriptionWindow` class that no longer existed.

The cycle is the point. A grep sees four importers and reads the code as live,
which is how it survived long enough to be mistaken for the foundation of a
briefing feature. These tests assert the modules are gone and, more usefully,
that nothing quietly reintroduces an import of them.
"""

from __future__ import annotations

import importlib

import pytest

RETIRED_MODULES = [
    "tldw_chatbook.Subscriptions.briefing_generator",
    "tldw_chatbook.Subscriptions.briefing_templates",
    "tldw_chatbook.Subscriptions.recursive_summarizer",
    "tldw_chatbook.Subscriptions.aggregation_engine",
    "tldw_chatbook.Subscriptions.rss_feed_generator",
    "tldw_chatbook.Subscriptions.export_manager",
    "tldw_chatbook.Subscriptions.distribution_manager",
    "tldw_chatbook.Subscriptions.textual_scheduler_worker",
    "tldw_chatbook.Subscriptions.website_monitor",
    "tldw_chatbook.Subscriptions.scheduler",
    "tldw_chatbook.UI.Subscription_Modules.subscription_backend_controller",
    "tldw_chatbook.Event_Handlers.subscription_events",
    "tldw_chatbook.Event_Handlers.subscription_ingest_worker",
    # TASK-1220: lost its only caller when subscription_ingest_worker went, and
    # was reachable afterwards only through the package's eager re-export --
    # present in sys.modules at startup, called by nothing.
    "tldw_chatbook.Subscriptions.content_processor",
    "tldw_chatbook.Internal_Prompts.subscriptions_prompts",
]

RETIRED_SYMBOLS = [
    "BriefingGenerator",
    "BriefingSchedule",
    "SubscriptionScheduler",
    "TextualSchedulerWorker",
    "create_scheduler",
]


@pytest.mark.parametrize("module_name", RETIRED_MODULES)
def test_retired_module_is_gone(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("symbol", RETIRED_SYMBOLS)
def test_retired_symbol_is_not_reachable_from_the_package(symbol: str) -> None:
    """The package used to serve these through a PEP 562 ``__getattr__`` shim.

    That shim existed to keep the deprecated names importable during a dual-run
    validation period which, per the ADR-019 amendment, never happened.
    """
    import tldw_chatbook.Subscriptions as subscriptions

    assert not hasattr(subscriptions, symbol)


def test_settings_no_longer_offers_inert_subscription_prompts() -> None:
    """TASK-1220: Settings listed five prompts that could not affect anything.

    `Settings > Internal Prompts` groups specs by subsystem and badges them as
    customizable. The `subscriptions` group described `ContentProcessor`, whose
    only caller was retired in TASK-1211 -- so a user could carefully edit the
    feed-analysis prompt and change nothing. A false affordance is worse than an
    absent one.
    """
    from tldw_chatbook.Internal_Prompts.catalog import CATALOG

    subscription_specs = [key for key in CATALOG if key.startswith("subscriptions.")]
    assert not subscription_specs, (
        f"Settings still offers inert subscription prompts: {subscription_specs}"
    )


def test_live_monitoring_seam_is_untouched() -> None:
    """`monitoring_engine` is what `WatchlistCheckHandler` actually calls.

    It sat one import away from the retired modules -- `website_monitor` imported
    from it -- so this pins the boundary the removal had to respect.
    """
    from tldw_chatbook.Subscriptions.monitoring_engine import FeedMonitor, URLMonitor

    assert FeedMonitor is not None
    assert URLMonitor is not None


def test_app_import_pulls_in_no_retired_module() -> None:
    """The regression that matters: an import path creeping back in.

    Every retired module was reachable from `tldw_chatbook.app` only through
    modules that are themselves gone, so any future appearance here means a new
    edge was added to a graph that should have no edges left.
    """
    import sys

    import tldw_chatbook.app  # noqa: F401

    resurrected = [name for name in RETIRED_MODULES if name in sys.modules]
    assert not resurrected, f"retired modules imported at app startup: {resurrected}"
