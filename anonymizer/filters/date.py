from typing import List
import nltk
from datetime import datetime
import re
from anonymizer.base import BaseFilter

class DateFilter(BaseFilter):
    """
    Filter for detecting and anonymizing dates using NLTK.
    """

    def __init__(self):
        """Initialize the date filter."""
        super().__init__()
        self._verify_resources()
        self._compile_patterns()

    def _verify_resources(self):
        """Verify required NLTK resources are available."""
        required = [
            ('tokenizers/punkt', 'punkt'),
            ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
        ]
        for path, resource in required:
            try:
                nltk.data.find(path)
            except LookupError:
                print(f"Downloading required resource: {resource}")
                nltk.download(resource, quiet=True)

    def _compile_patterns(self):
        """Compile regex patterns for date formats."""
        # Base patterns
        year = r'\d{4}'
        day = r'\d{1,2}(?:st|nd|rd|th)?'
        month = r'\d{1,2}'
        month_names = '|'.join(self._get_month_names())

        self.date_patterns = [
            # Full written dates with ordinals
            fr'\b(?:{month_names})\s+{day}(?:\s*,\s*{year})?\b',  # December 25th, 2023
            fr'\b{day}\s+(?:{month_names})(?:\s*,?\s*{year})?\b',  # 25th December, 2023

            # Numerical formats with different separators
            fr'\b{year}[-/]{month}[-/]{day}\b',  # 2023/12/25 or 2023-12-25
            fr'\b{month}[-/]{day}[-/]{year}\b',  # 12/25/2023 or 12-25-2023

            # Additional numerical variations
            fr'\b{day}[-/]{month}[-/]{year}\b',  # 25/12/2023 or 25-12-2023
            fr'\b{year}\.{month}\.{day}\b',  # 2023.12.25
        ]
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.date_patterns]

    def find(self, text: str) -> List[str]:
        """
        Find dates in text using regex patterns and NLTK temporal expression recognition.

        Args:
            text: Input text to search

        Returns:
            List of detected dates
        """
        if not text:
            return []

        try:
            dates = set()  # Using set to avoid duplicates

            # First pass: Check for formatted dates using regex
            for pattern in self.compiled_patterns:
                matches = pattern.finditer(text)
                for match in matches:
                    date_str = match.group().strip()
                    clean_date = self._clean_date(date_str)
                    if clean_date:
                        dates.add(clean_date)

            # Second pass: Use NLTK for natural language dates
            tokens = nltk.word_tokenize(text)
            tagged = nltk.pos_tag(tokens)

            current_date = []
            for i, (word, tag) in enumerate(tagged):
                word_lower = word.lower()

                # Check for date components
                if ((tag == 'CD' and self._could_be_date_number(word)) or 
                    word_lower in self._get_month_names() or 
                    word in [',', 'of'] or
                    word_lower in {'yesterday', 'today', 'tomorrow'}):
                    current_date.append(word)
                elif current_date:
                    date_str = ' '.join(current_date)
                    clean_date = self._clean_date(date_str)
                    if clean_date:
                        dates.add(clean_date)
                    current_date = []

            # Process final date if any
            if current_date:
                date_str = ' '.join(current_date)
                clean_date = self._clean_date(date_str)
                if clean_date:
                    dates.add(clean_date)

            # Sort by length (descending) to prioritize full date matches
            return sorted(list(dates), key=len, reverse=True)

        except Exception as e:
            print(f"Error in date detection: {str(e)}")
            if isinstance(e, LookupError):
                print("Attempting to download missing NLTK resources...")
                self._verify_resources()
                return self.find(text)
            return []

    def _get_month_names(self) -> List[str]:
        """Get list of month names and abbreviations."""
        return [
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december',
            'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
        ]

    def _could_be_date_number(self, word: str) -> bool:
        """Check if a number could be part of a date."""
        try:
            # Remove ordinal indicators and commas
            clean = word.lower().replace(',', '')
            for suffix in ['th', 'st', 'nd', 'rd']:
                if clean.endswith(suffix):
                    clean = clean[:-len(suffix)]

            num = int(clean)
            # Valid for day of month or year
            return 1 <= num <= 31 or (1900 <= num <= 2100)
        except ValueError:
            return False

    def _clean_date(self, date_str: str) -> str:
        """
        Clean and validate a date string, preserving the original format.
        """
        try:
            # Handle relative dates
            date_lower = date_str.lower()
            if date_lower in {'yesterday', 'today', 'tomorrow'}:
                return date_str

            # Keep original format for formatted dates
            for pattern in self.compiled_patterns:
                if pattern.match(date_str):
                    return date_str.strip()

            # For natural language dates, ensure proper formatting
            parts = date_str.replace(',', ' , ').split()
            cleaned_parts = []
            has_month = False
            has_day = False
            has_year = False

            for part in parts:
                part_lower = part.lower()
                if part_lower in self._get_month_names():
                    has_month = True
                    cleaned_parts.append(part)
                elif self._could_be_date_number(part):
                    if len(part) == 4 and 1900 <= int(part) <= 2100:
                        has_year = True
                        cleaned_parts.append(part)
                    else:
                        has_day = True
                        cleaned_parts.append(part)
                elif part == ',':
                    cleaned_parts.append(part)

            # Only return if we have at least month and day
            if has_month and has_day:
                return ' '.join(cleaned_parts)
            return ''

        except Exception:
            return ''

    def _is_valid_date(self, date_str: str) -> bool:
        """
        Check if a string represents a valid date expression.
        """
        try:
            # Handle relative dates
            if date_str.lower() in {'yesterday', 'today', 'tomorrow'}:
                return True

            # Try regex patterns first
            for pattern in self.compiled_patterns:
                if pattern.match(date_str):
                    return True

            # Check for valid month name and day combination
            tokens = nltk.word_tokenize(date_str.lower())
            has_month = any(token in self._get_month_names() for token in tokens)
            has_day = any(self._could_be_date_number(token) for token in tokens)

            return has_month and has_day

        except Exception:
            return False