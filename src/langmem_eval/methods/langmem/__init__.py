"""LangMem registration; defer SDK and service imports until construction."""
from ...registry import register_method


def create_aml(*args):
    from ...aml.backend import AMLLangMemBackend
    return AMLLangMemBackend(*args)


@register_method("langmem", architecture="LangMem session extraction/update + dense retrieval; read-only QA",
                 infrastructure="langmem, OpenAI, in-process LangGraph store", aml_factory=create_aml)
def create(model):
    from .backend import LangMemBackend
    return LangMemBackend(model)
