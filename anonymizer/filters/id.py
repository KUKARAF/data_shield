from typing import List
import nltk
from anonymizer.base import BaseFilter

class IdFilter(BaseFilter):
    """
    Filter for detecting and anonymizing ID numbers using NLTK.
    """

    def __init__(self):
        """Initialize the ID filter."""
        super().__init__()

    def find(self, text: str) -> List[str]:
        """
        Find IDs in text using NLTK's POS tagging and chunking.

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
                # Look for cardinal numbers in appropriate contexts
                if tag == 'CD':
                    # Check surrounding context for ID indicators
                    if self._is_id_context(tagged, i):
                        ids.append(word)

            return ids
        except Exception as e:
            print(f"Error in ID detection: {e}")
            return []

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
        id_indicators = {'id', 'number', 'no', 'reference', 'case', '#'}

        try:
            # Check previous token
            if position > 0:
                prev_word = tagged_tokens[position - 1][0].lower()
                if prev_word in id_indicators:
                    return True

            # Check next token
            if position < len(tagged_tokens) - 1:
                next_word = tagged_tokens[position + 1][0].lower()
                if next_word in id_indicators:
                    return True

            return False
        except Exception as e:
            print(f"Error in ID context detection: {e}")
            return False