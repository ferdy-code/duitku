import importlib

import pytest

PACKAGES = [
    "domain",
    "db",
    "db.repositories",
    "memory",
    "tools",
    "agent",
    "interfaces",
    "interfaces.telegram",
    "observability",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_importable(name: str) -> None:
    importlib.import_module(f"duitku.{name}")
