from typing import List
import nltk
from datetime import datetime
from anonymizer.base import BaseFilter

class DateFilter(BaseFilter):
    """
    Filter for detecting and anonymizing dates using NLTK.
    """

    def __init__(self):
        """Initialize the date filter."""
        super().__init__()

    def find(self, text: str) -> List[str]:
        """
        Find dates in text using NLTK temporal expression recognition.

        Args:
            text: Input text to search

        Returns:
            List of detected dates
        """
        if not text:
            return []

        try:
            # Tokenize and tag the text
            tokens = nltk.word_tokenize(text)
            tagged = nltk.pos_tag(tokens)

            dates = []
            current_date = []

            for i, (word, tag) in enumerate(tagged):
                # Look for cardinal numbers and month names
                if (tag == 'CD' and word.isdigit()) or \
                   (tag == 'NNP' and word.lower() in self._get_month_names()):
                    current_date.append(word)
                elif current_date:
                    # Check if we have a valid date expression
                    date_str = ' '.join(current_date)
                    if self._is_valid_date(date_str):
                        dates.append(date_str)
                    current_date = []

            # Check final date expression
            if current_date:
                date_str = ' '.join(current_date)
                if self._is_valid_date(date_str):
                    dates.append(date_str)

            return dates
        except Exception as e:
            print(f"Error in date detection: {e}")
            return []

    def _get_month_names(self) -> List[str]:
        """Get list of month names and abbreviations."""
        return ['january', 'february', 'march', 'april', 'may', 'june',
                'july', 'august', 'september', 'october', 'november', 'december',
                'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    def _is_valid_date(self, date_str: str) -> bool:
        """
        Check if a string represents a valid date expression.
        Uses NLTK's temporal expression analysis.
        """
        try:
            tokens = nltk.word_tokenize(date_str.lower())
            tagged = nltk.pos_tag(tokens)

            # Check for patterns like "January 1st, 2024" or "2024 January 1"
            has_month = any(word in self._get_month_names() for word, _ in tagged)
            has_number = any(tag == 'CD' for _, tag in tagged)

            return has_month and has_number
        except Exception:
            return False