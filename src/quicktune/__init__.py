from .core import registry

__all__ = ["registry", "register_default_recipes"]

_RECIPES_LOADED = False


def register_default_recipes():
    global _RECIPES_LOADED
    if _RECIPES_LOADED:
        return
    from . import recipes as _

    _RECIPES_LOADED = True
