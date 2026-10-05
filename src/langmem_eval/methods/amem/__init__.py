"""Register A-Mem without importing torch, loading weights, or calling APIs."""
from ...registry import MethodOption, register_method


def public_config():
    from .config import AMemSettings
    return AMemSettings.from_env().public_config()


@register_method("amem", architecture="A-Mem notes, links and neighbor evolution; keyword query + graph expansion",
                 infrastructure="Pinned A-Mem JSON prompts, local or API embeddings, unified answer protocol",
                 public_config=public_config, options=(
                     MethodOption("amem-response-format", "AMEM_RESPONSE_FORMAT",
                                  choices=("json_schema", "json_object", "prompt")),
                     MethodOption("amem-neighbor-k", "AMEM_NEIGHBOR_K", int),
                     MethodOption("amem-evolution-threshold", "AMEM_EVOLUTION_THRESHOLD", int),
                     MethodOption("amem-temperature", "AMEM_TEMPERATURE", float),
                     MethodOption("amem-max-output-tokens", "AMEM_MAX_OUTPUT_TOKENS", int),
                     MethodOption("amem-cache-mode", "AMEM_CACHE_MODE",
                                  choices=("off", "reuse", "refresh", "require"),
                                  help="Reuse completed A-Mem history; default off. require fails on cache miss."),
                     MethodOption("amem-cache-dir", "AMEM_CACHE_DIR",
                                  help="Completed memory cache directory; default data/amem-cache"),
                 ))
def create(model):
    from .backend import AMemBackend
    return AMemBackend(model)
