"""
Calibrated Urban Street Network Generator

A system for generating planar street networks that match real-world
urban morphology and space syntax patterns.
"""

__version__ = "0.1.0"

from .config import GeneratorConfig
from .generator import StreetNetworkGenerator
from .reference import ReferenceExtractor

__all__ = [
    "GeneratorConfig",
    "StreetNetworkGenerator",
    "ReferenceExtractor",
]
