"""Auto-discovery for registered datamodules and models.

Exposes `supported_datamodules` and `supported_models` which lazily populate
on first access by importing all submodules and collecting from factory registries.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Sequence
from functools import lru_cache
from typing import Any


class _LazyList(Sequence[Any]):
    """Sequence that materializes on first access."""

    def __init__(self, loader):
        self._loader = loader
        self._items = None

    def _load(self):
        if self._items is None:
            self._items = self._loader()
        return self._items

    def __getitem__(self, index):
        return self._load()[index]

    def __len__(self):
        return len(self._load())

    def __iter__(self):
        return iter(self._load())

    def __repr__(self):
        return repr(self._load())


@lru_cache(maxsize=None)
def _import_all_submodules(package_name: str) -> None:
    """Recursively import all submodules to trigger @register decorators."""
    package = importlib.import_module(package_name)
    for module_info in pkgutil.iter_modules(getattr(package, '__path__', [])):
        importlib.import_module(f'{package_name}.{module_info.name}')


def _load_datamodules() -> list[type[Any]]:
    """Import all datamodules and return sorted list from registry."""
    from src.data.datamodules.base_datamodule import DataModuleFactory

    _import_all_submodules('src.data.datamodules')
    return sorted(DataModuleFactory.datamodules.values(), key=lambda x: x.__name__)


def _load_models() -> list[type[Any]]:
    """Import all models and return sorted list from registry."""
    from src.models.base_model import ModelFactory

    _import_all_submodules('src.models')
    return sorted(ModelFactory.models.values(), key=lambda x: x.__name__)


supported_datamodules: Sequence[type[Any]] = _LazyList(_load_datamodules)
supported_models: Sequence[type[Any]] = _LazyList(_load_models)

__all__ = ['supported_datamodules', 'supported_models']
