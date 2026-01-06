"""
Gemma-2-2B Python Code Generation with LoRA/QLoRA

Utilities for training, inference, and evaluation.
"""

__version__ = "1.0.0"

from .inference import load_model, generate_code

__all__ = ['load_model', 'generate_code']
