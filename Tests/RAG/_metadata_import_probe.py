"""Finite cold metadata/received-capture probe; no engine/package stubs or App."""

from __future__ import annotations

import argparse
import builtins
import importlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys
import sysconfig

RECEIVED_NODE = "Tests/Chat/test_console_configuration_preparation.py::test_received_profile_depth_capture_uses_lightweight_metadata"
ENGINE_PREFIXES = (
    "tldw_chatbook.RAG_Search.simplified.embeddings_wrapper",
    "tldw_chatbook.RAG_Search.simplified.vector_store",
    "tldw_chatbook.RAG_Search.simplified.rag_service",
    "tldw_chatbook.RAG_Search.simplified.enhanced_rag_service",
    "tldw_chatbook.RAG_Search.simplified.enhanced_rag_service_v2",
    "tldw_chatbook.RAG_Search.simplified.rag_factory",
    "tldw_chatbook.RAG_Search.reranker",
    "tldw_chatbook.RAG_Search.parallel_processor",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--original-home", type=Path, required=True)
    parser.add_argument("--mode", choices=("library", "received"), required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    source, private, original_home = (
        args.source_root.resolve(),
        args.private_root.resolve(),
        args.original_home.resolve(),
    )
    assert Path(__file__).resolve() == source / "Tests/RAG/_metadata_import_probe.py"
    assert not any(
        name.startswith(("tldw_chatbook", "tldw_profile_core")) for name in sys.modules
    )
    core = source / "packages/tldw_profile_core/src"
    sys.path[:0] = [str(source), str(core)]
    expected = {
        "Tests": source / "Tests/__init__.py",
        "tldw_chatbook": source / "tldw_chatbook/__init__.py",
        "tldw_profile_core": core / "tldw_profile_core/__init__.py",
    }
    for name, path in expected.items():
        assert Path(importlib.util.find_spec(name).origin).resolve() == path.resolve()
    assert private != original_home and not original_home.is_relative_to(private)
    os.environ["TLDW_TEST_REAL_HOME"] = str(original_home)
    from Tests import real_profile_guard

    real_profile_guard.install()  # Original profile protection BEFORE HOME/private/product setup.
    assert real_profile_guard._real_home().resolve() == original_home
    assert real_profile_guard.protected_roots()
    for name in tuple(os.environ):
        if name.upper() not in {
            "SYSTEMROOT",
            "WINDIR",
            "COMSPEC",
            "PATH",
            "PATHEXT",
            "SYSTEMDRIVE",
            "TEMP",
            "TMP",
            "PROCESSOR_ARCHITECTURE",
            "NUMBER_OF_PROCESSORS",
            "TLDW_TEST_REAL_HOME",
        }:
            os.environ.pop(name, None)
    private.mkdir(parents=True, exist_ok=False)
    for name in ("home", "config", "data", "temp"):
        (private / name).mkdir()
    for key, relative in {
        "HOME": "home",
        "USERPROFILE": "home",
        "XDG_CONFIG_HOME": "config",
        "XDG_DATA_HOME": "data",
        "TLDW_TEST_CONFIG_ROOT": ".",
        "TLDW_CONFIG_PATH": "config/config.toml",
        "TEMP": "temp",
        "TMP": "temp",
    }.items():
        os.environ[key] = str(private / relative)
    os.environ.update(
        TLDW_TEST_MODE="1",
        PYTHONUTF8="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    network_denials, selfpipes, imports = [], [], []
    network_refusal_frames = []
    refused_import_probes = []
    assert "urllib3.util.connection" not in sys.modules
    urllib3_spec = importlib.util.find_spec("urllib3")
    expected_ipv6_source = (
        Path(urllib3_spec.origin).resolve().parent / "util/connection.py"
    )
    assert "urllib3.util.connection" not in sys.modules
    stdlib = Path(sysconfig.get_path("stdlib")).resolve()
    from asyncio import proactor_events, selector_events

    selfpipe_bindings = []
    for module, owner in (
        (proactor_events, proactor_events.BaseProactorEventLoop),
        (selector_events, selector_events.BaseSelectorEventLoop),
    ):
        function = owner._make_self_pipe
        expected_file = stdlib / "asyncio" / (module.__name__.rsplit(".", 1)[1] + ".py")
        assert Path(module.__file__).resolve() == expected_file
        assert Path(function.__code__.co_filename).resolve() == expected_file
        assert function.__globals__ is vars(module)
        selfpipe_bindings.append(
            (function.__code__, function.__globals__, expected_file)
        )

    def audit(event, values):
        if event not in {
            "socket.connect",
            "socket.connect_ex",
            "socket.bind",
            "socket.getaddrinfo",
            "socket.gethostbyname",
            "socket.gethostbyname_ex",
            "socket.gethostbyaddr",
            "socket.getnameinfo",
            "socket.sendto",
            "socket.sendmsg",
        }:
            return
        frame = sys._getframe(1)
        direct_frame = frame
        canonical_selfpipe = False
        code_frames = []
        while frame is not None:
            if len(code_frames) < 24:
                code_frames.append(
                    {
                        "file": frame.f_code.co_filename,
                        "function": frame.f_code.co_name,
                        "line": getattr(frame, "f_lineno", None),
                    }
                )
            if any(
                frame.f_code is code
                and frame.f_globals is globals_owner
                and Path(frame.f_code.co_filename).resolve() == expected_file
                for code, globals_owner, expected_file in selfpipe_bindings
            ):
                canonical_selfpipe = True
            frame = frame.f_back
        address = values[1] if len(values) > 1 else None
        if (
            canonical_selfpipe
            and event in {"socket.connect", "socket.bind"}
            and isinstance(address, tuple)
            and address[0] in ("127.0.0.1", "::1")
        ):
            selfpipes.append(event)
            return
        # This genuine import-time capability probe remains REFUSED. Its
        # original OSError fallback must not become a network permission.
        partial = sys.modules.get("urllib3.util.connection")
        probe = getattr(partial, "_has_ipv6", None)
        if (
            event == "socket.bind"
            and address == ("::1", 0)
            and partial is not None
            and direct_frame.f_code is getattr(probe, "__code__", None)
            and direct_frame.f_globals is vars(partial)
            and Path(direct_frame.f_code.co_filename).resolve() == expected_ipv6_source
            and Path(partial.__file__).resolve() == expected_ipv6_source
        ):
            if len(refused_import_probes) < 8:
                refused_import_probes.append("urllib3._has_ipv6")
            raise OSError("Known import-time IPv6 probe refused by test")
        network_denials.append(event)
        if len(network_refusal_frames) < 8:
            network_refusal_frames.append({"event": event, "code_frames": code_frames})
        raise AssertionError(
            "Fresh metadata probe refuses non-selfpipe network activity"
        )

    sys.addaudithook(audit)

    def blocked(name):
        if name.split(".", 1)[0] in {"numpy", "torch"}:
            imports.append(name)
            raise ImportError("Heavy ML import attempted by metadata probe: " + name)

    original_import, original_import_module = (
        builtins.__import__,
        importlib.import_module,
    )

    def checked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0:
            blocked(name)
        return original_import(name, globals, locals, fromlist, level)

    def checked_import_module(name, package=None):
        blocked(name)
        return original_import_module(name, package)

    class EngineGuard(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if any(
                fullname == prefix or fullname.startswith(prefix + ".")
                for prefix in ENGINE_PREFIXES
            ):
                imports.append(fullname)
                raise ImportError(
                    "RAG engine import attempted by metadata probe: " + fullname
                )
            return None

    assert not any(
        name.split(".", 1)[0] in {"numpy", "torch"}
        or any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in ENGINE_PREFIXES
        )
        for name in sys.modules
    ), "Cold heavy-import guard would be vacuous"
    builtins.__import__ = checked_import
    importlib.import_module = checked_import_module
    sys.meta_path.insert(0, EngineGuard())
    receipt = {
        "mode": args.mode,
        "source_origins_pinned": True,
        "original_home_guard_before_redirection": True,
        "engine_packages_stubbed": False,
        "import_attempts": imports,
        "network_denials": network_denials,
        "network_refusal_frames": network_refusal_frames,
        "refused_import_probes": refused_import_probes,
        "canonical_asyncio_selfpipe_events": selfpipes,
        "qualified": False,
        "product_samples": 0,
    }
    status = 1
    try:
        from tldw_chatbook.Backup_Recovery import bootstrap

        config_path, bootstrap_root = (
            bootstrap.effective_config_path(),
            bootstrap.default_bootstrap_root(),
        )
        assert config_path.is_relative_to(private) and bootstrap_root.is_relative_to(
            private
        )
        admission = bootstrap.startup_permission(config_path, bootstrap_root)
        receipt["startup_admitted"] = bool(admission[0])
        assert admission[0], admission
        if args.mode == "library":
            from tldw_chatbook.Library.library_rag_state import (
                library_rag_profile_top_k,
            )

            receipt["default_depth"] = library_rag_profile_top_k()
            assert receipt["default_depth"] == 15
            assert not imports
            from tldw_chatbook.RAG_Search.config_profiles import get_profile_manager
            from tldw_chatbook.RAG_Search.simplified.active_config import (
                resolve_active_rag_config,
                resolve_active_rag_top_k,
                set_active_profile,
            )
            from tldw_chatbook.RAG_Search.simplified.config import EmbeddingConfig
            from tldw_chatbook.config import save_setting_to_cli_config
            from tldw_chatbook.RAG_Search.config_metadata import (
                RerankingConfig,
                ProcessingConfig,
            )
            import pickle

            assert EmbeddingConfig().device == "auto"  # Preserve the real default.
            manager = get_profile_manager()
            custom = manager.clone_profile("hybrid_basic", "Cold metadata live profile")
            custom.rag_config.embedding.device = (
                "cpu"  # Full resolver parity fixture; no device probe.
            )
            custom.rag_config.search.default_top_k = 23
            manager.save_profile(custom)
            set_active_profile(custom.id)
            assert library_rag_profile_top_k() == resolve_active_rag_top_k() == 23
            assert resolve_active_rag_config().search.default_top_k == 23
            custom.rag_config.search.default_top_k = 31
            manager.save_profile(custom)
            assert (
                library_rag_profile_top_k()
                == resolve_active_rag_config().search.default_top_k
                == 31
            )
            os.environ["RAG_TOP_K"] = "7"
            assert (
                library_rag_profile_top_k()
                == resolve_active_rag_config().search.default_top_k
                == 7
            )
            os.environ["RAG_TOP_K"] = "0"
            assert (
                resolve_active_rag_top_k()
                == resolve_active_rag_config().search.default_top_k
                == 0
            )
            assert library_rag_profile_top_k() == 5
            os.environ["RAG_TOP_K"] = "invalid-depth"
            assert library_rag_profile_top_k() == 5
            for resolve in (resolve_active_rag_top_k, resolve_active_rag_config):
                try:
                    resolve()
                except ValueError:
                    pass
                else:
                    raise AssertionError(
                        "Original invalid environment refusal disappeared"
                    )
            os.environ.pop("RAG_TOP_K")
            assert save_setting_to_cli_config(
                "rag.service", "profile", "missing-private-profile"
            )
            os.environ["RAG_DEVICE"] = "cpu"
            assert (
                library_rag_profile_top_k()
                == resolve_active_rag_config().search.default_top_k
                == 15
            )
            for cls in (RerankingConfig, ProcessingConfig):
                value = cls()
                assert type(pickle.loads(pickle.dumps(value))) is cls
                assert cls.__module__ == "tldw_chatbook.RAG_Search.config_metadata"
            receipt["live_parity"] = {
                "default": 15,
                "custom": 23,
                "updated": 31,
                "env": 7,
                "zero_env_library_fallback": 5,
                "invalid_env_library_fallback": 5,
                "missing_profile": 15,
                "full_resolver_device_fixture": "cpu",
                "default_device_unchanged": "auto",
                "canonical_pickle_without_engine": True,
            }
        else:
            import pytest

            class BootstrapProfile:
                def pytest_configure(self, config):
                    config.addinivalue_line(
                        "markers",
                        "bootstrap_profile: single cold child keeps its admitted private collection profile",
                    )

                def pytest_collection_modifyitems(self, items):
                    assert len(items) == 1 and items[0].nodeid == RECEIVED_NODE
                    assert items[0].get_closest_marker("bootstrap_profile") is not None

                def pytest_runtest_teardown(self):
                    assert bootstrap.effective_config_path() == config_path
                    assert bootstrap.default_bootstrap_root() == bootstrap_root
                    assert not real_profile_guard.take_violations()
                    assert not network_denials

            # Avoid ONLY the original conftest's eager config_profiles preload.
            # Source/home/storage/socket guards and bootstrap_profile identity are
            # explicitly installed above; imported fixture dependencies stay real.
            status = pytest.main(
                [
                    RECEIVED_NODE,
                    "--noconftest",
                    "-q",
                    "-p",
                    "pytest_asyncio.plugin",
                    "-p",
                    "pytest_timeout",
                    "--basetemp=" + str(private / "pytest"),
                ],
                plugins=[BootstrapProfile()],
            )
            assert status == 0
        assert not imports
        assert not any(
            name == "numpy" or name.startswith(("numpy.", "torch"))
            for name in sys.modules
        )
        status = 0
    except BaseException as error:
        receipt["exception_type"] = type(error).__name__
        raise
    finally:
        receipt["real_profile_refusals"] = real_profile_guard.take_violations()
        receipt["status"] = status
        receipt["loaded_source_modules"] = {
            name: str(Path(module.__file__).resolve())
            for name, module in tuple(sys.modules.items())
            if name.startswith(("tldw_chatbook", "tldw_profile_core", "Tests"))
            and getattr(module, "__file__", None)
        }
        assert all(
            Path(path).is_relative_to(source)
            for path in receipt["loaded_source_modules"].values()
        )
        (private / "probe-results.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(receipt))
        assert not receipt["real_profile_refusals"] and not network_denials
    return status


if __name__ == "__main__":
    raise SystemExit(main())
