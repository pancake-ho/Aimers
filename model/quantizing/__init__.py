"""Quantizing package exports.

Keep imports lazy so `python main.py --help` does not require optional
AutoRound dependencies at module import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["AutoRoundQuantizer", "GPTQquantize"]

if TYPE_CHECKING:
    from .quantize import GPTQquantize


def __getattr__(name: str):
    if name in __all__:
        try:
            from .quantize import AutoRoundQuantizer, GPTQquantize
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise ImportError(
                "AutoRound utilities require optional dependency 'auto_round'."
            ) from exc
        return {
            "GPTQquantize": GPTQquantize,
        }[name]
    raise AttributeError(f"module 'quantizing' has no attribute '{name}'")
