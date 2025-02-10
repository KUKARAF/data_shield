from typing import List, Dict
import nltk
import re
from anonymizer.base import BaseFilter

class NameFilter(BaseFilter):
    """Filter for detecting and anonymizing names using NLTK."""

    def __init__(self):
        """Initialize the name filter with NLTK resources."""
        super().__init__()
        self._verify_resources()
        self._init_name_lists()

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

    def _init_name_lists(self):
        """Initialize lists of common names and titles."""
        self.honorifics = {
            'mr', 'mrs', 'ms', 'dr', 'prof', 'professor',
            'pan', 'pani', 'sir', 'madam', 'lord', 'lady',
            'rev', 'reverend', 'captain', 'major', 'colonel'
        }

    def find(self, text: str) -> List[Dict[str, str]]:
        """Find names in text using NLTK and additional heuristics."""
        if not text:
            return []

        try:
            # Normalize honorifics first
            normalized_text = self._normalize_honorifics(text)

            # Process text sentence by sentence
            sentences = nltk.sent_tokenize(normalized_text)
            names = []

            for sentence in sentences:
                tokens = nltk.word_tokenize(sentence)
                tagged = nltk.pos_tag(tokens)

                i = 0
                while i < len(tagged):
                    word, tag = tagged[i]
                    word_lower = word.lower().rstrip('.')

                    # Handle honorifics followed by proper nouns
                    if word_lower in self.honorifics:
                        # Normalize title
                        title = word if word.endswith('.') else (f"{word}." if len(word) <= 3 else word)

                        # Look ahead for name parts
                        name_parts = []
                        j = i + 1
                        while j < len(tagged):
                            next_word, next_tag = tagged[j]
                            if next_tag.startswith('NNP') or next_word in ['Jr.', 'Sr.']:
                                name_parts.append(next_word)
                                j += 1
                            else:
                                break

                        if name_parts:
                            name_info = self._create_name_info(title, name_parts)
                            if name_info:
                                names.append(name_info)
                            i = j
                            continue

                    # Handle proper nouns (potential names without titles)
                    elif tag.startswith('NNP'):
                        name_parts = []
                        j = i
                        while j < len(tagged):
                            curr_word, curr_tag = tagged[j]
                            if curr_tag.startswith('NNP') or curr_word in ['Jr.', 'Sr.']:
                                name_parts.append(curr_word)
                                j += 1
                            else:
                                break

                        if len(name_parts) > 0:
                            # Check if first word is actually a title
                            first_word = name_parts[0].lower().rstrip('.')
                            if first_word in self.honorifics:
                                title = name_parts[0]
                                if not title.endswith('.') and len(title) <= 3:
                                    title = f"{title}."
                                name_info = self._create_name_info(title, name_parts[1:])
                            else:
                                name_info = self._create_name_info(None, name_parts)

                            if name_info:
                                names.append(name_info)
                            i = j
                            continue

                    i += 1

            return names

        except Exception as e:
            print(f"Error in name detection: {str(e)}")
            if isinstance(e, LookupError):
                print("Attempting to download missing NLTK resources...")
                self._verify_resources()
                return self.find(text)
            return []

    def _create_name_info(self, title: str, name_parts: List[str]) -> Dict[str, str]:
        """Create a name info dictionary from name components."""
        if not name_parts:
            return None

        # Process title
        if title:
            title = title.rstrip('.')
            if len(title) <= 3:
                title = f"{title}."

        # Split into components
        first_name = name_parts[0]
        remaining_parts = name_parts[1:]

        # Process last name and suffixes
        last_parts = []
        for part in remaining_parts:
            if part in ['Jr.', 'Sr.']:
                last_parts.append(part)
            elif part.rstrip('.') in ['Jr', 'Sr']:
                last_parts.append(f"{part}.")
            else:
                last_parts.append(part)

        last_name = ' '.join(last_parts) if last_parts else ''

        # Build full name with proper spacing
        parts = []
        if title:
            parts.append(title)
        parts.append(first_name)
        if last_name:
            parts.append(last_name)

        # Return the name components
        return {
            'full': ' '.join(parts),
            'title': title if title else '',
            'first': first_name,
            'last': last_name
        }

    def _normalize_honorifics(self, text: str) -> str:
        """Normalize honorifics to ensure proper detection."""
        words = text.split()
        for i, word in enumerate(words):
            word_lower = word.lower().rstrip('.')
            if word_lower in self.honorifics:
                if not word.endswith('.') and len(word) <= 3:
                    words[i] = f"{word}."
        return ' '.join(words)