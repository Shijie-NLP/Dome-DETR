"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy
import os
from typing import Any

import yaml

from .workspace import GLOBAL_CONFIG

__all__ = [
    "load_config",
    "merge_config",
    "merge_dict",
    "parse_cli",
]


INCLUDE_KEY = "__include__"


def load_config(file_path, cfg=None) -> dict:
    """
    A yaml file as a dict, with the files it lists under ``__include__`` merged in first, in
    order, so that the file's own keys win over its includes and later includes win over earlier
    ones. Include paths are relative to the including file unless absolute.
    """
    cfg = {} if cfg is None else cfg
    _, ext = os.path.splitext(file_path)
    assert ext in [".yml", ".yaml"], "only support yaml files"

    with open(file_path) as f:
        file_cfg = yaml.safe_load(f) or {}

    for base_yaml in file_cfg.pop(INCLUDE_KEY, []):
        base_yaml = os.path.expanduser(base_yaml)
        if not os.path.isabs(base_yaml):
            base_yaml = os.path.join(os.path.dirname(file_path), base_yaml)
        merge_dict(cfg, load_config(base_yaml))

    return merge_dict(cfg, file_cfg)


def _deep_merge(dct: dict, another: dict, overwrite: bool) -> dict:
    """
    Merge ``another`` into ``dct`` in place: keys missing from ``dct`` are added and dicts are
    merged recursively. Where both hold a non-dict value, ``another`` wins if ``overwrite`` is
    set and ``dct`` keeps its value otherwise.
    """
    for k, v in another.items():
        if k in dct and isinstance(dct[k], dict) and isinstance(v, dict):
            _deep_merge(dct[k], v, overwrite)
        elif k not in dct or overwrite:
            dct[k] = v
    return dct


def merge_dict(dct, another_dct, inplace=True) -> dict:
    """Merge ``another_dct`` into ``dct`` recursively; ``another_dct`` wins where both hold a value."""
    if not inplace:
        dct = copy.deepcopy(dct)
    return _deep_merge(dct, another_dct, overwrite=True)


def dictify(s: str, v: Any) -> dict:
    """``dictify('a.b.c', 3)`` is ``{'a': {'b': {'c': 3}}}``."""
    if "." not in s:
        return {s: v}
    key, rest = s.split(".", 1)
    return {key: dictify(rest, v)}


def parse_cli(nargs: list[str]) -> dict:
    """
    Command-line overrides as a nested dict: ``['a.c=3', 'b=10']`` is ``{'a': {'c': 3}, 'b': 10}``.
    Values are parsed as yaml, so ``x=1e-4``, ``x=true`` and ``x=[1, 2]`` keep their types.
    """
    cfg = {}
    for s in nargs or []:
        s = s.strip()
        if "=" not in s:
            raise ValueError(f"expected key=value, got {s!r}")
        k, v = s.split("=", 1)
        cfg = merge_dict(cfg, dictify(k, yaml.safe_load(v)))
    return cfg


def merge_config(cfg, another_cfg=GLOBAL_CONFIG, inplace: bool = False, overwrite: bool = False) -> dict:
    """
    Merge ``another_cfg`` (the registry by default) into ``cfg``: keys missing from ``cfg`` are
    added, dicts are merged recursively, and values present in both are kept from ``cfg`` unless
    ``overwrite`` is set. This is how a yaml's ``HGNetv2: {name: B0}`` ends up on top of the
    registered defaults of ``HGNetv2``.

    Example:

        cfg1 = load_config('./dfine_r18vd_6x_coco.yml')
        cfg1 = merge_config(cfg1, inplace=True)

        cfg2 = load_config('./dfine_r50vd_6x_coco.yml')
        cfg2 = merge_config(cfg2, inplace=True)

        model1 = create(cfg1['model'], cfg1)
        model2 = create(cfg2['model'], cfg2)
    """
    if not inplace:
        cfg = copy.deepcopy(cfg)
    return _deep_merge(cfg, another_cfg, overwrite=overwrite)
