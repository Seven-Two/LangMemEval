"""Compatibility imports for the A-Mem prompts now colocated with the method."""
from .methods.amem.prompts import (
    ANALYSIS_PROMPT, ANALYSIS_SCHEMA, EVOLUTION_PROMPT, EVOLUTION_SCHEMA,
    QUERY_PROMPT, QUERY_SCHEMA,
)

__all__ = ["ANALYSIS_PROMPT", "ANALYSIS_SCHEMA", "EVOLUTION_PROMPT", "EVOLUTION_SCHEMA",
           "QUERY_PROMPT", "QUERY_SCHEMA"]
