"""Method packages: register a lazy factory in each public package's __init__.

Keep algorithms, settings and prompts inside that package. Discovery imports
only this level; private modules and backend modules are not auto-imported.
Single-file registrations remain supported for small experiments.
"""
