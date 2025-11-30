"""
@file: utils.py
@description: Utility functions for registering configuration classes with Hydra's ConfigStore.
"""

from typing import Callable, TypeVar

from hydra.core.config_store import ConfigStore

from src.configs.constants import ConfigName

T = TypeVar('T')


def register_config(group: ConfigName) -> Callable:
    """
    Decorator to register a configuration class with Hydra's ConfigStore.

    Args:
        group (ConfigName): The group name to register the configuration class under.

    Returns:
        Callable: The decorator function that registers the class.
    """

    def decorator(cls):
        cs = ConfigStore.instance()
        cs.store(
            group=group.value,
            name=cls.__name__,
            node=cls,
        )
        return cls

    return decorator


def register_trainer(cls: type[T]) -> type[T]:
    """
    Wrapper to register a trainer configuration class with the specified group.

    Args:
        cls (type): The trainer configuration class to register.

    Returns:
        type: The registered trainer configuration class.
    """
    return register_config(group=ConfigName.TRAINER)(cls)


def register_model_config(cls: type[T]) -> type[T]:
    """
    Wrapper to register a model configuration class with the specified group.

    Args:
        cls (type): The model configuration class to register.

    Returns:
        type: The registered model configuration class.
    """
    return register_config(group=ConfigName.MODEL)(cls)


def register_data(cls: type[T]) -> type[T]:
    """
    Wrapper to register a data configuration class with the specified group.

    Args:
        cls (type): The data configuration class to register.

    Returns:
        type: The registered data configuration class.
    """
    return register_config(group=ConfigName.DATA)(cls)
