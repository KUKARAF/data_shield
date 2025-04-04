"""
Text anonymization package using NLTK for grammatically correct substitutions.
Supports name and ID anonymization with extensible filter system.
"""
from anonymizer.core import Anonymizer
from anonymizer.base import BaseFilter

__version__ = "0.1.0"
__all__ = ["Anonymizer", "BaseFilter"]