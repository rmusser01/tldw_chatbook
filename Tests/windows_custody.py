"""Explicit eligibility policy for the elevated native custody CI job."""

import os
from typing import NoReturn

import pytest


REQUIRED_CUSTODY_ENV = "TLDW_REQUIRE_ELEVATED_CUSTODY"


def elevated_custody_required() -> bool:
    return os.environ.get(REQUIRED_CUSTODY_ENV) == "1"


def unavailable_custody(reason: str) -> NoReturn:
    if elevated_custody_required():
        pytest.fail(f"{REQUIRED_CUSTODY_ENV}=1: {reason}", pytrace=False)
    pytest.skip(reason)
