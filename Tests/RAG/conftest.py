# Re-export the canonical scratch_config fixture from Tests/Internal_Prompts.conftest.
#
# We use a plain import (not pytest_plugins) because declaring pytest_plugins =
# ["Tests.Internal_Prompts.conftest"] — either in the test module or here —
# collides with pytest's implicit auto-load of Tests/Internal_Prompts/conftest.py
# under --import-mode=importlib when both test directories are collected in the
# same session: "ValueError: Plugin already registered under a different name".
# See Tests/Web_Scraping/test_websearch_internal_prompts.py.
#
# pytest discovers fixtures present in a conftest's namespace, so a plain import
# sidesteps this collision and gives us exactly ONE canonical definition.
from Tests.Internal_Prompts.conftest import scratch_config  # noqa: F401
