from typing import List
from abc import ABC, abstractmethod

class BaseFilter(ABC):
    """
    Base class for all data filters.
    """

    def __init__(self):
        """Initialize base filter."""
        pass

    @abstractmethod
    def find(self, text: str) -> List[str]:
        """
        Find all instances of the target data type in the text.

        Args:
            text: Input text to search

        Returns:
            List of matched strings
        """
        pass