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

        # Use NLTK for named entity recognition
        try:
            tokens = nltk.word_tokenize(text)
            tagged = nltk.pos_tag(tokens)
            entities = nltk.chunk.ne_chunk(tagged)

            names = []
            current_name = []

            for node in entities:
                if isinstance(node, nltk.Tree):
                    if node.label() == 'PERSON':
                        # Extract full name from the tree node
                        name = ' '.join([leaf[0] for leaf in node.leaves()])
                        if name not in names:
                            names.append(name)
                elif current_name:
                    # Handle multi-token names and honorifics
                    if node[1].startswith('NNP') or node[0].lower() in {'mr', 'mrs', 'ms', 'dr', 'prof', 'pan', 'pani'}:
                        current_name.append(node[0])
                    else:
                        if len(current_name) > 0:
                            name = ' '.join(current_name)
                            if name not in names:
                                names.append(name)
                        current_name = []
                elif node[1].startswith('NNP') or node[0].lower() in {'mr', 'mrs', 'ms', 'dr', 'prof', 'pan', 'pani'}:
                    current_name.append(node[0])

            # Add final multi-token name if exists
            if len(current_name) > 0:
                name = ' '.join(current_name)
                if name not in names:
                    names.append(name)

            return names
        except Exception as e:
            print(f"Error in name detection: {e}")
            return []