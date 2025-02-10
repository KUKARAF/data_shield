import re
from typing import Dict, List, Optional
import nltk
from anonymizer.base import BaseFilter
from anonymizer.utils import load_filters, ensure_nltk_resources

class Anonymizer:
    """Main anonymization class that handles text processing and filter management."""

    def __init__(self, filters: Optional[List[str]] = None, preserve_grammar: bool = True):
        """Initialize the anonymizer with specified filters."""
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
        """Replace personal data with placeholders while preserving context."""
        if not text:
            return text

        self._substitutions.clear()
        self._reverse_substitutions.clear()

        # First collect all matches from each filter
        all_matches = {}
        for filter_name, filter_obj in self._filters.items():
            matches = filter_obj.find(text)
            if matches:
                all_matches[filter_name] = matches

        # Process matches in reverse order of length to avoid partial replacements
        result = text
        for filter_name, matches in all_matches.items():
            if filter_name == 'name':
                # Special handling for names with titles
                for idx, name_info in enumerate(matches, 1):
                    if not name_info or name_info['full'] in self._reverse_substitutions:
                        continue

                    # Build placeholder preserving structure
                    components = []

                    # Keep title if present
                    if name_info['title']:
                        components.append(name_info['title'])

                    # Add first name placeholder
                    first_placeholder = f"<FIRST_NAME_{idx}>"
                    components.append(first_placeholder)
                    self._substitutions[first_placeholder] = name_info['first']

                    # Add last name placeholder if present
                    if name_info['last']:
                        last_placeholder = f"<LAST_NAME_{idx}>"
                        components.append(last_placeholder)
                        self._substitutions[last_placeholder] = name_info['last']

                    # Create the full replacement preserving surrounding context
                    placeholder = ' '.join(components)
                    self._reverse_substitutions[name_info['full']] = placeholder

                    # Replace the name while preserving surrounding context
                    pattern = rf"\b{re.escape(name_info['full'])}\b"
                    result = re.sub(pattern, placeholder, result)

            else:
                # Standard handling for other filters
                for idx, match in enumerate(matches, 1):
                    if not match or match in self._reverse_substitutions:
                        continue

                    placeholder = f"<{filter_name.upper()}_{idx}>"
                    self._substitutions[placeholder] = match
                    self._reverse_substitutions[match] = placeholder

                    # Use word boundaries for replacement
                    pattern = rf"\b{re.escape(match)}\b"
                    result = re.sub(pattern, placeholder, result)

        if self.preserve_grammar:
            result = self._ensure_grammar(result)

        return result

    def fill_personal_data(self, text: str) -> str:
        """Replace placeholders with original personal data."""
        if not text:
            return text

        result = text
        # Sort substitutions by length of placeholder (longest first)
        sorted_subs = sorted(
            [(k, v) for k, v in self._substitutions.items()],
            key=lambda x: len(x[0]),
            reverse=True
        )

        # Restore all placeholders
        for placeholder, original in sorted_subs:
            result = result.replace(placeholder, original)

        return result

    def _ensure_grammar(self, text: str) -> str:
        """Ensure grammatical correctness of the anonymized text."""
        try:
            tokens = nltk.word_tokenize(text)
            tagged = nltk.pos_tag(tokens)

            result = text
            for i, (word, tag) in enumerate(tagged):
                if word.startswith('<') and word.endswith('>'):
                    if i > 0:
                        prev_word = tagged[i-1][0].lower()
                        if prev_word in ['a', 'an']:
                            if self._starts_with_vowel_sound(word):
                                result = result.replace(f"{prev_word} {word}", f"an {word}")
                            else:
                                result = result.replace(f"{prev_word} {word}", f"a {word}")

            return result

        except Exception as e:
            print(f"Error in grammar correction: {str(e)}")
            return text

    def _starts_with_vowel_sound(self, word: str) -> bool:
        """Determine if a word starts with a vowel sound."""
        if word.startswith('<') and '_' in word:
            entity_type = word[1:word.find('_')].lower()
            return entity_type[0] in 'aeiou'
        return word[0].lower() in 'aeiou'