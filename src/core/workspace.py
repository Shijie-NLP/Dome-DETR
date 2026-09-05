"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.

The registry behind the yaml configs.

``register`` records a class's constructor signature under its name; ``create`` builds an
instance from a config entry, resolving two kinds of entry:

- a class entry, keyed by the class name (``HGNetv2: {name: B0, ...}``): the registered
  defaults with the yaml's overrides merged on top by ``merge_config``;
- an alias entry, keyed by a role (``optimizer: {type: AdamW, lr: 1e-4}``): the named class's
  defaults plus the arguments listed next to ``type``.

Two class attributes shape what a constructor receives. ``__share__`` names arguments that are
read from the top level of the config when present there (``num_classes``, say), so they need
stating only once. ``__inject__`` names arguments that are themselves built by ``create``: their
config value is either the name of another entry, or an inline ``{type: ..., ...}`` dict.
"""

import importlib
import inspect
from typing import Any

# name -> schema (see extract_schema) for every registered class, plus registered functions
GLOBAL_CONFIG: dict[str, Any] = {}


def register(dct: Any = GLOBAL_CONFIG, name=None, force=False):
    """
    Register a class or function under ``name`` (its own name by default). ``dct`` is the
    registry dict, or a class on which the object is set as an attribute. Registering a name
    twice is an error unless ``force`` is set.
    """

    def decorator(obj):
        register_name = obj.__name__ if name is None else name
        if inspect.isclass(dct):
            assert force or not hasattr(dct, register_name), f"{dct.__name__} already has {register_name}"
            setattr(dct, register_name, obj)
            return obj

        assert force or register_name not in dct, f"{register_name} has already been registered"
        if inspect.isclass(obj):
            dct[register_name] = extract_schema(obj)
        elif inspect.isfunction(obj):
            dct[register_name] = obj
        else:
            raise ValueError(f"Do not support {type(obj)} register")
        return obj

    return decorator


def extract_schema(module: type) -> dict:
    """
    What ``create`` needs to build ``module`` from a config: where to import it from, its
    ``__inject__`` and ``__share__`` lists, and one key per constructor argument holding the
    default (``None`` for required arguments). The defaults are kept a second time under
    ``_kwargs`` so an alias entry can start from a clean set after the yaml has overridden the
    top-level ones.
    """
    argspec = inspect.getfullargspec(module.__init__)
    arg_names = [arg for arg in argspec.args if arg != "self"]
    defaults = argspec.defaults or ()
    num_required = len(arg_names) - len(defaults)

    schema = {
        "_name": module.__name__,
        "_pymodule": importlib.import_module(module.__module__),
        "_inject": list(getattr(module, "__inject__", [])),
        "_share": list(getattr(module, "__share__", [])),
    }
    kwargs = {}
    for i, arg in enumerate(arg_names):
        if arg in schema["_share"]:
            assert i >= num_required, f"{module.__name__}.{arg}: a shared argument must have a default"
        kwargs[arg] = defaults[i - num_required] if i >= num_required else None
    schema.update(kwargs)
    schema["_kwargs"] = kwargs
    return schema


def create(type_or_name, global_cfg=GLOBAL_CONFIG, **kwargs):
    """
    Build the object a config entry describes. ``kwargs`` are arguments supplied by the caller
    (an optimizer's ``params``, an evaluator's ``coco_gt``) and win over the config.
    """
    name = type_or_name if isinstance(type_or_name, str) else type_or_name.__name__
    if name not in global_cfg:
        raise ValueError(f"The module {name} is not registered")

    cfg = global_cfg[name]
    if not isinstance(cfg, dict):
        # an object placed in the config directly, already built
        return cfg

    if "type" in cfg:
        schema = _schema(cfg["type"], global_cfg)
        args = {k: v for k, v in cfg.items() if k != "type"}
        return _instantiate(schema, {**schema["_kwargs"], **args, **kwargs}, global_cfg)

    schema = _schema(name, global_cfg)
    args = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return _instantiate(schema, {**args, **kwargs}, global_cfg)


def _schema(name, global_cfg) -> dict:
    schema = global_cfg.get(name)
    if not isinstance(schema, dict) or "_pymodule" not in schema:
        raise ValueError(f"{name} is not a registered class")
    return schema


def _instantiate(schema: dict, args: dict, global_cfg):
    """Resolve the shared and injected arguments of ``args`` and call the constructor."""
    module = getattr(schema["_pymodule"], schema["_name"])
    args = dict(args)

    for k in schema["_share"]:
        if k in global_cfg:
            args[k] = global_cfg[k]

    for k in schema["_inject"]:
        spec = args.get(k)
        if spec is None:
            continue
        if isinstance(spec, str):
            if spec not in global_cfg:
                raise ValueError(f"Missing inject config of {spec}.")
            args[k] = create(spec, global_cfg)
        elif isinstance(spec, dict):
            if "type" not in spec:
                raise ValueError(f"Missing `type` in the inject config of {k}.")
            inject_schema = _schema(spec["type"], global_cfg)
            inject_args = {key: v for key, v in spec.items() if key != "type"}
            args[k] = _instantiate(inject_schema, {**inject_schema["_kwargs"], **inject_args}, global_cfg)
        else:
            raise ValueError(f"Inject does not support {spec}")

    return module(**args)
