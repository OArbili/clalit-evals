import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "tests", ROOT / "tests" / "data", ROOT / "getting_started", ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Live tests need EVALS_LIVE=1 and an API key; otherwise they are skipped, never failed."""
    if os.environ.get("EVALS_LIVE") == "1" and os.environ.get("ANTHROPIC_API_KEY"):
        return
    skip = pytest.mark.skip(reason="live judge tests run only with EVALS_LIVE=1 and ANTHROPIC_API_KEY set")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
