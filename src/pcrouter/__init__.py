"""pcrouter: KV-cache-aware request routing for LLM serving.

One routing core (:mod:`pcrouter.router`, :mod:`pcrouter.policies`, :mod:`pcrouter.index`)
drives both a trace-driven cluster simulator (:mod:`pcrouter.sim`) and an OpenAI-compatible
proxy (:mod:`pcrouter.server`).
"""

__version__ = "0.2.0"
