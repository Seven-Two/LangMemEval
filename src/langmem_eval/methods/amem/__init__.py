"""Register A-Mem without importing torch, loading weights, or calling APIs."""
from ...registry import register_method


@register_method("amem", architecture="A-Mem notes, links and neighbor evolution; keyword query + graph expansion",
                 infrastructure="Pinned A-Mem JSON prompts, local or API embeddings, unified answer protocol")
def create(model):
    from .backend import AMemBackend
    return AMemBackend(model)
