"""Lazy watchlist exports; importing recovery never composes monitoring."""

from importlib import import_module

_BASE = {
    "LocalWatchlistsService": (".local_watchlists_service", "LocalWatchlistsService"),
    "ServerWatchlistsService": (
        ".server_watchlists_service",
        "ServerWatchlistsService",
    ),
    "build_watchlist_item_id": (".watchlist_normalizers", "build_watchlist_item_id"),
    "normalize_local_subscription_row": (
        ".watchlist_normalizers",
        "normalize_local_subscription_row",
    ),
    "normalize_server_delete_response": (
        ".watchlist_normalizers",
        "normalize_server_delete_response",
    ),
    "normalize_server_watchlist_source": (
        ".watchlist_normalizers",
        "normalize_server_watchlist_source",
    ),
    "normalize_watchlist_alert_rule": (
        ".watchlist_normalizers",
        "normalize_watchlist_alert_rule",
    ),
    "normalize_watchlist_run": (".watchlist_normalizers", "normalize_watchlist_run"),
    "WatchlistBackend": (".watchlist_scope_service", "WatchlistBackend"),
    "WatchlistScopeService": (".watchlist_scope_service", "WatchlistScopeService"),
}
_OPTIONAL = {
    "FeedMonitor": (".monitoring_engine", "FeedMonitor"),
    "URLMonitor": (".monitoring_engine", "URLMonitor"),
    "RateLimiter": (".monitoring_engine", "RateLimiter"),
    "CircuitBreaker": (".monitoring_engine", "CircuitBreaker"),
    "ContentExtractor": (".monitoring_engine", "ContentExtractor"),
    "CredentialEncryptor": (".security", "CredentialEncryptor"),
    "InputValidator": (".security", "InputValidator"),
}
__version__ = "1.0.0"
__author__ = "TLDW ChatBook Team"


def _load_optional():
    if "_CORE_AVAILABLE" not in globals():
        try:
            for name, (module, symbol) in _OPTIONAL.items():
                globals()[name] = getattr(import_module(module, __name__), symbol)
            globals()["_CORE_AVAILABLE"] = True
        except ImportError:
            globals()["_CORE_AVAILABLE"] = False
    return globals()["_CORE_AVAILABLE"]


def __getattr__(name):
    if name == "__all__":
        value = (list(_OPTIONAL) if _load_optional() else []) + list(_BASE)
    elif name == "_CORE_AVAILABLE":
        return _load_optional()
    elif name in _OPTIONAL:
        _load_optional()
        if name not in globals():
            raise AttributeError(name)
        return globals()[name]
    elif name in _BASE:
        module, symbol = _BASE[name]
        value = getattr(import_module(module, __name__), symbol)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
