from collections.abc import Callable
from dataclasses import MISSING, Field
from functools import wraps
from pathlib import Path
from typing import Protocol, get_type_hints

import click

from evalhub.inference.schemas import GenerationConfig, SampleParams


class DataclassProtocol(Protocol):
    __dataclass_fields__: dict


TYPE_MAP = {
    bool: click.BOOL,
    int: click.INT,
    float: click.FLOAT,
    str: click.STRING,
    Path: click.Path(),
}


def _get_click_type(field_type: type) -> click.ParamType:
    r"""Convert Python type to Click type."""
    if str(field_type).startswith("pathlib."):
        return click.Path()
    if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
        return click.STRING
    return TYPE_MAP.get(field_type, click.STRING)


def _create_click_option(field_name: str, field_info: Field, field_type: type) -> click.Option | None:
    r"""Create a click option from field metadata."""
    metadata = field_info.metadata
    if metadata.get("hidden", False):
        return None

    cli_name = metadata.get("cli_name", f"--{field_name.replace('_', '-')}")
    click_type = _get_click_type(field_type)

    option_kwargs = {
        "default": field_info.default,
        "help": metadata.get("help", ""),
        "type": click_type,
        "required": metadata.get("required", False),
    }

    if metadata.get("multiple", False):
        option_kwargs["multiple"] = True

    if "validate" in metadata:
        validate_func = metadata["validate"]

        def validation_callback(ctx, param, value):
            if value is not None:
                if isinstance(value, list | tuple):
                    for v in value:
                        if not validate_func(v):
                            raise click.BadParameter(f"Invalid value: {v}")
                else:
                    if not validate_func(value):
                        raise click.BadParameter(f"Invalid value: {value}")
            return value

        option_kwargs["callback"] = validation_callback

    return click.option(cli_name, **option_kwargs)


def options(cls: DataclassProtocol) -> Callable[[Callable], Callable]:
    r"""Decorator to automatically generate click options from dataclass metadata."""

    def decorator(func: Callable):
        type_hints = get_type_hints(cls)

        def apply_options(fields: list[tuple[str, Field]], skip_type: type = None):
            for field_name, field_info in fields:
                field_type = type_hints.get(field_name, type(field_info.default))
                if field_type != skip_type and (opt := _create_click_option(field_name, field_info, field_type)):
                    nonlocal func
                    func = opt(func)

        # Handle nested SampleParams if present
        if SampleParams in type_hints.values():
            apply_options(list(SampleParams.__dataclass_fields__.items()))

        # Handle main class fields
        apply_options(list(cls.__dataclass_fields__.items()), SampleParams)

        @wraps(func)
        def wrapper(*args, **kwargs):
            print(args, kwargs)
            # Create SampleParams from extracted kwargs
            sample_param_names = set(SampleParams.__dataclass_fields__.keys())
            sample_kwargs = {k: v for k, v in kwargs.items() if k in sample_param_names}
            config_kwargs = {k: v for k, v in kwargs.items() if k not in sample_param_names}

            # Clean up None values and convert types
            sample_kwargs = {k: v for k, v in sample_kwargs.items() if v is not None}
            config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}

            # Parse tasks
            config_kwargs["tasks"] = [task.strip().lower() for task in config_kwargs["tasks"].split(",")]

            # Create config objects
            sample_params = SampleParams(**sample_kwargs)
            config = GenerationConfig(sample_params=sample_params, **config_kwargs)
            return func(config, *args)

        return wrapper

    return decorator


def _extract_field_info(cls: DataclassProtocol, include_nested: bool = True) -> list[dict]:
    r"""Extract field information from dataclass for display."""
    fields_info = []
    type_hints = get_type_hints(cls)
    fields: list[tuple[str, Field]] = list(cls.__dataclass_fields__.items())
    for field_name, field_info in fields:
        field_type = type_hints.get(field_name, type(field_info.default))
        metadata = field_info.metadata

        if include_nested and field_type == SampleParams:
            nested_fields = _extract_field_info(SampleParams, include_nested=False)
            fields_info.extend(nested_fields)
            continue

        if metadata.get("hidden", False):
            continue

        fields_info.append(
            {
                "name": metadata.get("cli_name", f"--{field_name.replace('_', '-')}"),
                "type": field_type.__name__ if hasattr(field_type, "__name__") else str(field_type),
                "default": field_info.default if field_info.default is not MISSING else field_info.default_factory(),
                "help": metadata.get("help", "No description available"),
                "required": metadata.get("required", False),
            }
        )

    return fields_info
