from typing import Dict, List, Optional
import nltk
from anonymizer.base import BaseFilter
from anonymizer.utils import load_filters, ensure_nltk_resources

class Anonymizer:
    """
    Main anonymization class that handles text processing and filter management.
    """

    def __init__(
        self,
        filters: Optional[List[str]] = None,
        preserve_grammar: bool = True
    ):
        """
        Initialize the anonymizer with specified filters.

        Args:
            filters: List of filter names to apply (e.g., ["name", "date", "id"])
            preserve_grammar: Whether to maintain grammatical correctness
        """
        self.preserve_grammar = preserve_grammar
        self._filters: Dict[str, BaseFilter] = {}
        self._substitutions: Dict[str, str] = {}
        self._reverse_substitutions: Dict[str, str] = {}

        # Ensure NLTK resources are available
        ensure_nltk_resources()

        # Load specified filters or all available ones
        available_filters = load_filters()
        if filters is None:
            self._filters = available_filters
        else:
            self._filters = {
                f: filt for f, filt in available_filters.items()
                if f.lower() in [x.lower() for x in filters]
            }

    def hide_personal_data(self, text: str) -> str:
        """
        Replace personal data with placeholders.

        Args:
            text: Input text containing personal data

        Returns:
            Anonymized text with placeholders
        """
        if not text:
            return text

        self._substitutions.clear()
        self._reverse_substitutions.clear()

        # First collect all matches from each filter
        all_matches = {}
        for filter_name, filter_obj in self._filters.items():
            matches = filter_obj.find(text)
            if matches:
                all_matches[filter_name] = sorted(matches, key=len, reverse=True)

        # Process matches in order (longer matches first)
        result = text
        # Sort filters to ensure consistent ordering (name > date > id)
        for filter_name in sorted(all_matches.keys()):
            for idx, match in enumerate(all_matches[filter_name], 1):
                if not match or match in self._reverse_substitutions:
                    continue

                placeholder = f"<{filter_name.upper()}_{idx}>"
                self._substitutions[placeholder] = match
                self._reverse_substitutions[match] = placeholder
                # Replace all occurrences of this match
                result = result.replace(match, placeholder)

        if self.preserve_grammar:
            result = self._ensure_grammar(result)

        return result

    def fill_personal_data(self, text: str) -> str:
        """
        Replace placeholders with original personal data.

        Args:
            text: Text with placeholders

        Returns:
            Text with restored personal data
        """
        if not text:
            return text

        result = text
        # Sort substitutions by length of placeholder (longest first)
        # to avoid partial replacements
        sorted_subs = sorted(
            self._substitutions.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        for placeholder, original in sorted_subs:
            result = result.replace(placeholder, original)

        return result

    def _ensure_grammar(self, text: str) -> str:
        """
        Ensure grammatical correctness of the anonymized text using NLTK.

        Args:
            text: Text to process

        Returns:
            Grammatically corrected text
        """
        try:
            # Tokenize and tag the text
            doc = nltk.word_tokenize(text)
            tagged = nltk.pos_tag(doc)

            result = text
            for i, (word, tag) in enumerate(tagged):
                if word.startswith('<') and word.endswith('>'):
                    # Adjust articles based on pronunciation
                    if i > 0:
                        prev_word, prev_tag = tagged[i-1]
                        if prev_word.lower() in ['a', 'an']:
                            # Determine correct article based on pronunciation
                            if self._starts_with_vowel_sound(word):
                                result = result.replace(f"{prev_word} {word}", f"an {word}")
                            else:
                                result = result.replace(f"{prev_word} {word}", f"a {word}")

            return result

        except Exception as e:
            print(f"Error in grammar correction: {str(e)}")
            return text

    def _starts_with_vowel_sound(self, word: str) -> bool:
        """
        Determine if a word starts with a vowel sound.
        Special handling for placeholders.
        """
        # For placeholders, use the type of entity
        if word.startswith('<') and '_' in word:
            entity_type = word[1:word.find('_')].lower()
            return entity_type[0] in 'aeiou'
        return word[0].lower() in 'aeiou'