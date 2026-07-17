"""Neural-network model definitions.

This module is the public place to import model classes from. The underlying
implementations currently live in ``crystallographic_benchmark`` for backwards
compatibility with earlier notebooks.
"""

from .crystallographic_benchmark import ResNet18Gray, ResNetCrystal

__all__ = [
    "ResNet18Gray",
    "ResNetCrystal",
]

