"""Generate argparse arguments from a pydantic config class.

Before this bridge, every hyperparameter lived in up to five places — the
pydantic field default, an ``add_argument(default=...)`` mirror, the
``args.x`` → ``Config(x=...)`` transcription, a config→kwargs explosion, and
the train-function signature default — with nothing enforcing agreement
("defaults = best-known settings" had two silently divergeable sources of
truth). With it, the config class is the single source of truth for names,
types, defaults, and help text.

Flag conventions match the repo's existing CLI surface, so adopting the
bridge for a config whose fields already mirror its flags does not change
that experiment's command lines:

- ``field_name`` → ``--field-name``
- ``bool`` field defaulting to ``False`` → ``--field-name`` (store_true)
- ``bool`` field defaulting to ``True`` named ``use_x`` → ``--no-x``
  (store_false into ``use_x``); other default-``True`` bools →
  ``--no-field-name``
- ``Literal[...]`` → ``choices``
- ``X | None`` → parsed as ``X``
- ``Field(description=...)`` → ``--help`` text

Typical usage::

    parser = argparse.ArgumentParser()
    add_config_args(parser, MyTrainingConfig)
    parser.add_argument("--save-checkpoint", action="store_true")  # non-config args
    args = parser.parse_args()
    config = config_from_args(MyTrainingConfig, args)
"""

import argparse
import types
import typing

from pydantic import BaseModel


def _unwrap_optional(annotation: object) -> object:
    """``X | None`` → ``X``; anything else unchanged."""
    if typing.get_origin(annotation) in (types.UnionType, typing.Union):
        members = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(members) == 1:
            return members[0]
    return annotation


def _flag_for(name: str, *, negated: bool) -> str:
    if not negated:
        return "--" + name.replace("_", "-")
    stem = name.removeprefix("use_")
    return "--no-" + stem.replace("_", "-")


def add_config_args(
    parser: argparse.ArgumentParser,
    config_cls: type[BaseModel],
    *,
    exclude: tuple[str, ...] = (),
) -> None:
    """Add one CLI argument per field of ``config_cls`` to ``parser``.

    ``exclude`` skips fields the caller handles specially (or that make no
    sense on this script's command line).

    Raises ``TypeError`` for a field type the bridge can't map (a container,
    a nested model, ...) — exclude it and add the argument by hand.
    """
    for name, field in config_cls.model_fields.items():
        if name in exclude:
            continue
        annotation = _unwrap_optional(field.annotation)
        required = field.is_required()
        default = None if required else field.get_default(call_default_factory=True)
        help_text = field.description or ""
        if annotation is bool:
            if required or default is None:
                raise TypeError(
                    f"{config_cls.__name__}.{name}: a bool field needs a "
                    f"True/False default to pick a flag convention (tri-state "
                    f"bools don't map to a flag) — pass exclude=({name!r},) "
                    f"and add it by hand"
                )
            if default is True:
                parser.add_argument(
                    _flag_for(name, negated=True),
                    dest=name,
                    action="store_false",
                    help=help_text or f"disable {name}",
                )
            else:
                parser.add_argument(
                    _flag_for(name, negated=False),
                    action="store_true",
                    help=help_text,
                )
        elif annotation in (int, float, str):
            assert isinstance(annotation, type)
            parser.add_argument(
                _flag_for(name, negated=False),
                type=annotation,
                default=default,
                required=required,
                help=help_text or ("required" if required else f"default: {default}"),
            )
        elif typing.get_origin(annotation) is typing.Literal:
            choices = typing.get_args(annotation)
            parser.add_argument(
                _flag_for(name, negated=False),
                type=type(choices[0]),
                choices=choices,
                default=default,
                required=required,
                help=help_text or ("required" if required else f"default: {default}"),
            )
        else:
            raise TypeError(
                f"{config_cls.__name__}.{name}: can't map {field.annotation!r} "
                f"to an argparse argument — pass exclude=({name!r},) and add "
                f"it by hand"
            )


def config_from_args[C: BaseModel](
    config_cls: type[C],
    args: argparse.Namespace,
    **overrides: object,
) -> C:
    """Build a config from parsed args, for fields present on the namespace.

    ``overrides`` wins over the namespace — use it for fields that were
    excluded from :func:`add_config_args` or computed from other args.
    """
    values: dict[str, object] = {
        name: getattr(args, name)
        for name in config_cls.model_fields
        if hasattr(args, name)
    }
    values.update(overrides)
    return config_cls(**values)
