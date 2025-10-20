"""Typer utilities for generating CLI options from dataclasses."""

import functools
import inspect
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from typing import Annotated, get_origin, get_type_hints

import typer


def process_dataclass_fields[T](cls: type[T]) -> list[inspect.Parameter]:
    parameters = []

    for field in fields(cls):
        if is_dataclass(field.type):
            nested_parameters = process_dataclass_fields(field.type)
            parameters.extend(nested_parameters)
        else:
            annotation = Annotated[
                field.type,
                typer.Option(help=field.metadata.get("help", f"help of {field.name}")),
            ]
            param = inspect.Parameter(
                name=field.name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=field.default,
                annotation=annotation,
            )
            parameters.append(param)

    return parameters


def build_cls[T](cls: type[T], kwargs: dict) -> T:
    params = {}
    for field in fields(cls):
        if is_dataclass(field.type):
            params[field.name] = build_cls(field.type, kwargs)
        else:
            if field.name in kwargs:
                params[field.name] = kwargs[field.name]
                kwargs.pop(field.name)
    return cls(**params)


def options[T](cls: type[T]):
    assert is_dataclass(cls), "options decorator can only be applied to dataclass types"

    def decorator[R](func: Callable[..., type[R]]) -> Callable[..., type[R]]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            new_kwargs = kwargs.copy()
            cls_instance = build_cls(cls, new_kwargs)
            type_hints = get_type_hints(func, include_extras=True)
            for name in inspect.signature(func).parameters.keys():
                if type_hints.get(name) == cls:
                    new_kwargs[name] = cls_instance
            return func(*args, **new_kwargs)

        sig = inspect.signature(func)
        type_hints = get_type_hints(func, include_extras=True)
        new_params = []

        # First skip the original dataclass parameter
        for name, param in sig.parameters.items():
            if get_origin(type_hints.get(name)) is None and type_hints.get(name) == cls:
                continue
            new_params.append(param)

        # Then add parameters from the dataclass fields
        new_params.extend(process_dataclass_fields(cls))
        wrapper.__signature__ = sig.replace(parameters=new_params)
        return wrapper

    return decorator
