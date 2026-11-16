"""Format-agnostic document conversion.

Public surface:

``default_registry()`` returns the built-in converters, ``ConversionLimits``
bounds a single conversion, and ``render_markdown()`` turns the shared model
into GitHub-flavored Markdown.
"""

from .model import Block, Document
from .registry import (
    ConversionContext,
    ConversionLimits,
    ConverterRegistry,
    default_registry,
)
from .render import render_markdown

__all__ = [
    "Block",
    "ConversionContext",
    "ConversionLimits",
    "ConverterRegistry",
    "Document",
    "default_registry",
    "render_markdown",
]
