from typing import List
import nltk
import re
from anonymizer.base import BaseFilter

class NameFilter(BaseFilter):
    """
    Filter for detecting and anonymizing names using NLTK's NER.
    """

    def __init__(self):
        """Initialize the name filter with NLTK resources."""
        super().__init__()
        self._verify_resources()
        self._init_name_lists()

    def _verify_resources(self):
        """Verify that all required NLTK resources are available."""
        required_resources = [
            ('tokenizers/punkt', 'punkt'),
            ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
            ('chunkers/maxent_ne_chunker', 'maxent_ne_chunker'),
            ('corpora/words', 'words')
        ]

        for path, resource in required_resources:
            try:
                nltk.data.find(path)
            except LookupError:
                print(f"Downloading required resource: {resource}")
                nltk.download(resource, quiet=True)

    def _init_name_lists(self):
        """Initialize lists of common names and titles."""
        self.common_names = set(nltk.corpus.words.words())
        self.honorifics = {
            'mr', 'mrs', 'ms', 'dr', 'prof', 'professor',
            'pan', 'pani', 'sir', 'madam', 'lord', 'lady',
            'rev', 'reverend', 'captain', 'major', 'colonel'
        }

    def find(self, text: str) -> List[str]:
        """
        Find names in text using NLTK NER and additional heuristics.

        Args:
            text: Input text to search

        Returns:
            List of detected names
        """
        if not text:
            return []

        try:
            # Tokenize and process the text
            tokens = nltk.word_tokenize(text)
            tagged = nltk.pos_tag(tokens)

            # First pass: Get named entities
            entities = nltk.chunk.ne_chunk(tagged)
            names = []  # Using list to maintain order

            # Process named entities and build name sequences
            current_name = []
            for i, node in enumerate(entities):
                if isinstance(node, nltk.Tree):
                    if node.label() == 'PERSON':
                        name = ' '.join(leaf[0] for leaf in node.leaves())
                        # Check for preceding honorific
                        if current_name and current_name[-1].lower() in self.honorifics:
                            name = ' '.join(current_name + [name])
                        names.append(name)
                        current_name = []
                else:
                    word, tag = node
                    word_lower = word.lower()

                    # Handle titles and proper nouns
                    if word_lower in self.honorifics:
                        if current_name:
                            # Add accumulated name if exists
                            full_name = ' '.join(current_name)
                            if self._is_valid_name(full_name):
                                names.append(full_name)
                        current_name = [word]
                    elif tag.startswith('NNP'):
                        if not current_name and self._is_valid_name(word):
                            names.append(word)
                        current_name.append(word)
                    elif current_name:
                        # Check if we have a valid name sequence
                        full_name = ' '.join(current_name)
                        if self._is_valid_name(full_name):
                            names.append(full_name)
                        current_name = []

            # Add final name if exists
            if current_name:
                full_name = ' '.join(current_name)
                if self._is_valid_name(full_name):
                    names.append(full_name)

            # Remove duplicates while preserving order
            return list(dict.fromkeys(names))

        except Exception as e:
            print(f"Error in name detection: {str(e)}")
            if isinstance(e, LookupError):
                print("Attempting to download missing NLTK resources...")
                self._verify_resources()
                # Retry once after downloading resources
                return self.find(text)
            return []

    def _is_valid_name(self, name: str) -> bool:
        """
        Check if a word or phrase is likely to be a name.
        """
        words = name.split()
        if not words:
            return False

        # Handle honorific + name combinations
        if words[0].lower() in self.honorifics:
            words = words[1:]  # Skip honorific for validation
            if not words:  # Don't allow bare honorifics
                return False

        # Single word case (after removing honorific)
        if len(words) == 1:
            word = words[0]
            if word[0].isupper():
                tagged = nltk.pos_tag([word])
                return (word in self.common_names or 
                       tagged[0][1].startswith('NNP'))

        # Multi-word case (after removing honorific)
        return all(word[0].isupper() for word in words)