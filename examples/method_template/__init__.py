"""Copy this directory to src/langmem_eval/methods/my_method and rename it."""
from langmem_eval.registry import register_method


@register_method("my_method", architecture="Example: recent historical turns",
                 infrastructure="In-memory, no model calls during write/retrieve")
def create(model):
    from .backend import MyMethodBackend
    return MyMethodBackend(model)
