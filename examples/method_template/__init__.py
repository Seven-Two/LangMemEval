"""Copy this directory to src/langmem_eval/methods/my_method and rename it."""
from langmem_eval.registry import MethodOption, register_method


def public_config():
    from .config import Settings
    return Settings.from_env().public_config()


@register_method("my_method", architecture="Example: recent historical turns",
                 infrastructure="In-memory, no model calls during write/retrieve",
                 options=(MethodOption("my-method-window", "MY_METHOD_WINDOW", int,
                                       help="Maximum retained historical turns"),),
                 public_config=public_config)
def create(model):
    from .backend import MyMethodBackend
    return MyMethodBackend(model)
