from ..registry import register_method


@register_method("langmem", architecture="LangMem session extraction/update + dense retrieval; read-only QA",
                 infrastructure="langmem, OpenAI, in-process LangGraph store")
def create(model):
    from ..backend import LangMemBackend
    return LangMemBackend(model)
