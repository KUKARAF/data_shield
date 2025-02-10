from typing import List
import nltk
import re
from anonymizer.base import BaseFilter

class IdFilter(BaseFilter):
    """
    Filter for detecting and anonymizing ID numbers using NLTK.
    """

    def __init__(self):
        """Initialize the ID filter."""
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
        """Compile regex patterns for common ID formats."""
        self.id_patterns = [
            # Generic ID patterns
            r'\b\d{5,12}\b',  # 5-12 digit numbers
            r'\b[A-Z]{1,3}-\d{3,8}\b',  # Letter-number combinations
            r'\b\d{2,4}-\d{2,4}-\d{2,4}\b',  # Segmented numbers
            # Add more patterns as needed
        ]
        self.compiled_patterns = [re.compile(pattern) for pattern in self.id_patterns]

    def find(self, text: str) -> List[str]:
        """
        Find IDs in text using NLTK's POS tagging and pattern matching.

        Args:
            text: Input text to search

        Returns:
            List of detected IDs
        """
        if not text:
            return []

        try:
            # Tokenize and tag the text
            tokens = nltk.word_tokenize(text)
            tagged = nltk.pos_tag(tokens)

            ids = []
            for i, (word, tag) in enumerate(tagged):
                # Check for numbers and potential IDs
                if (tag == 'CD' or self._matches_id_pattern(word)) and self._is_id_context(tagged, i):
                    # Validate and clean the ID
                    clean_id = self._clean_id(word)
                    if clean_id and clean_id not in ids:
                        ids.append(clean_id)

            return ids

        except Exception as e:
            print(f"Error in ID detection: {e}")
            if isinstance(e, LookupError):
                print("Attempting to download missing NLTK resources...")
                self._verify_resources()
                # Retry once after downloading resources
                return self.find(text)
            return []

    def _matches_id_pattern(self, text: str) -> bool:
        """Check if text matches any ID pattern."""
        return any(pattern.match(text) for pattern in self.compiled_patterns)

    def _clean_id(self, id_text: str) -> str:
        """Clean and validate an ID string."""
        # Remove common separators and whitespace
        clean = re.sub(r'[\s\-_.]', '', id_text)
        # Ensure the cleaned ID still looks valid
        if len(clean) >= 5 and (clean.isalnum() or clean.isdigit()):
            return id_text  # Return original format if valid
        return ''

    def _is_id_context(self, tagged_tokens: List[tuple], position: int) -> bool:
        """
        Check if a token appears in a context that suggests it's an ID.
        Uses surrounding words and their POS tags to make the determination.

        Args:
            tagged_tokens: List of (word, tag) tuples from NLTK POS tagger
            position: Position of the token to check

        Returns:
            True if the context suggests an ID, False otherwise
        """
        id_indicators = {
            'id', 'number', 'no', 'reference', 'case', '#', 'code',
            'identifier', 'registration', 'serial', 'account'
        }

        try:
            context_window = 3  # Check 3 tokens before and after
            start = max(0, position - context_window)
            end = min(len(tagged_tokens), position + context_window + 1)

            # Check surrounding context
            for i in range(start, end):
                if i == position:
                    continue

                word = tagged_tokens[i][0].lower()
                tag = tagged_tokens[i][1]

                # Check for ID indicators
                if word in id_indicators:
                    return True

                # Check for possessive patterns
                if tag in ['PRP$', 'POS'] and i < position:
                    next_word = tagged_tokens[i + 1][0].lower()
                    if next_word in id_indicators:
                        return True

            return False

        except Exception as e:
            print(f"Error in ID context detection: {e}")
            return False