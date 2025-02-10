from typing import List
import nltk
from anonymizer.base import BaseFilter

class NameFilter(BaseFilter):
    """
    Filter for detecting and anonymizing names using NLTK's NER.
    """

    def __init__(self):
        """Initialize the name filter with NLTK resources."""
        super().__init__()
        self._verify_resources()

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

    def find(self, text: str) -> List[str]:
        """
        Find names in text using NLTK NER.

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
            names = set()  # Using set to avoid duplicates

            # Define honorifics and titles
            honorifics = {
                'mr', 'mrs', 'ms', 'dr', 'prof', 'professor',
                'pan', 'pani', 'sir', 'madam', 'lord', 'lady',
                'rev', 'reverend', 'captain', 'major', 'colonel'
            }

            # Process named entities first
            current_name = []
            for node in entities:
                if isinstance(node, nltk.Tree):
                    if node.label() == 'PERSON':
                        name = ' '.join(leaf[0] for leaf in node.leaves())
                        if len(name.split()) > 1:  # Only add multi-word names
                            names.add(name)
                    current_name = []
                else:
                    word, tag = node
                    word_lower = word.lower()

                    # Handle titles, honorifics, and proper nouns
                    if (tag.startswith('NNP') or 
                        word_lower in honorifics or 
                        (current_name and tag in ['CC', 'IN', 'DT'])):  # Include connecting words
                        current_name.append(word)
                    elif current_name:
                        name = ' '.join(current_name)
                        if len(name.split()) > 1:  # Only add multi-word names
                            names.add(name)
                        current_name = []

            # Add final name if exists
            if current_name and len(' '.join(current_name).split()) > 1:
                names.add(' '.join(current_name))

            # Second pass: Look for title patterns
            i = 0
            while i < len(tagged):
                word, tag = tagged[i]
                word_lower = word.lower()

                # Check for title patterns
                if word_lower in honorifics or (tag.startswith('NNP') and word_lower in honorifics):
                    name_parts = [word]
                    j = i + 1

                    # Collect subsequent proper nouns and allowed connectors
                    while j < len(tagged) and (
                        tagged[j][1].startswith('NNP') or 
                        tagged[j][0] in ['.', 'and', 'of', 'the'] or
                        tagged[j][1] in ['CC', 'IN', 'DT']
                    ):
                        if tagged[j][0] != '.':  # Don't include periods
                            name_parts.append(tagged[j][0])
                        j += 1

                    if len(name_parts) > 1:
                        names.add(' '.join(name_parts))
                    i = j
                else:
                    i += 1

            return sorted(list(names))  # Convert set back to sorted list

        except Exception as e:
            print(f"Error in name detection: {str(e)}")
            if isinstance(e, LookupError):
                print("Attempting to download missing NLTK resources...")
                self._verify_resources()
                return self.find(text)  # Retry once after downloading resources
            return []